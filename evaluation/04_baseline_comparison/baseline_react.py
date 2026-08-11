# -*- coding: utf-8 -*-
"""
Strong Baseline 4: ReAct Single Agent Framework Baseline
==================================================================
Represents the "general-purpose single agent framework" baseline: using
Yao et al. (2023)'s ReAct (Reasoning + Acting) paradigm, providing the agent
with the same extraction tools as the multi-agent approach, letting the agent
autonomously decide processing order, retry strategy, and termination condition.

Fairness:
  - Input: the same table_with_title-new1.json (load_clean_input)
  - Output: the same schema, saved by save_pred
  - Underlying LLM: Claude 4.5 Opus (consistent with RAG / Codegen baselines, single variable comparison)
  - Single-chunk extraction prompt is identical to the RAG baseline, ensuring differences
    come only from the "agent framework"

★ Safety: API key is read from environment variable only, never hardcoded.

Dependencies:
  pip install requests  # Only needs requests; ReAct loop is hand-written to avoid langchain version coupling

Usage:
  set ANTHROPIC_API_KEY=your_key
  python baseline_react.py                    # full run
  python baseline_react.py --limit 50         # only first 50 chunks (functional verification)
  python baseline_react.py --max-retries 2    # max retries per chunk
"""
from __future__ import annotations

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Optional

import requests

from baselines_common import (
    load_clean_input, code_from_title, save_pred,
)

CLAUDE_URL = "https://api.anthropic.com/v1/messages"
MODEL_NAME = "claude-opus-4-5-20250929"

# Single extraction prompt (identical to RAG baseline, ensuring differences come only from the agent framework)
EXTRACT_PROMPT = r"""Task: Extract structured JSON from the following single airway HTML table/text. Output only one JSON object, no explanation.
Airway code: {code}
Content:
{content}

Strict output format:
{{"airway_code":"{code}","MEA":"number or none","MAA":"number or none",
"airway_point":[{{"name":"point name","region":"two-letter region code or none","type":"VOR/NDB/FIX/WP/etc","position":["none","none"]}}]}}

Rules: MEA=3rd column/first altitude, MAA=4th column; fill "none" for missing; keep waypoints in order, remove consecutive duplicates only."""


def _require_key():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY not found. Set ANTHROPIC_API_KEY=your_key first")


def call_llm(messages: List[Dict[str, str]], temperature: float = 0, timeout: int = 120) -> str:
    # Anthropic API: system message is a separate field, rest go into messages
    system_msg = ""
    user_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_msgs.append(m)
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
               "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    body = {"model": MODEL_NAME, "temperature": temperature, "max_tokens": 8192,
            "messages": user_msgs}
    if system_msg:
        body["system"] = system_msg
    r = requests.post(CLAUDE_URL, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def _extract_json_obj(text: str):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _extract_one(code: str, content: str) -> Optional[Dict[str, Any]]:
    """Low-level single extraction (called by the ReAct agent's extract_chunk tool)."""
    prompt = EXTRACT_PROMPT.format(code=code, content=content[:8000])
    obj = _extract_json_obj(call_llm([{"role": "user", "content": prompt}]))
    if obj and obj.get("airway_point"):
        return obj
    return None


# ---------------------------------------------------------------------------
# ReAct Implementation: Hand-written Thought-Action-Observation loop
# ---------------------------------------------------------------------------
# Reasons for hand-writing instead of using langchain.create_react_agent:
#   1) Avoid reproducibility issues caused by API differences across langchain/langgraph versions
#   2) Fully controllable prompt and termination conditions, facilitating paper reproduction and review
#   3) Still faithfully follows the ReAct paradigm of Yao et al. (2023) (Thought/Action/Observation)
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are a ReAct agent that exhaustively extracts airway records from FAA NASR documents.

You operate in a strict Thought -> Action -> Observation loop:
- Thought: reason about which chunk to process next, whether a retry is needed, and whether you are done.
- Action: call exactly ONE of the available tools, in the form `Action: tool_name(args)`.
- Observation: you will receive the tool's output; incorporate it into your next Thought.

Available tools:
  extract_chunk(title)
      Extract the airway record from the chunk with the given title (string).
      Returns: {"ok": true, "record": {...}} on success,
               {"ok": false, "error": "..."} on failure.
  list_pending()
      Returns a JSON list of titles that have NOT been successfully extracted yet
      (and have not exhausted their retry budget).
  done()
      Call this ONLY when list_pending() returns "[]" to terminate the loop.

Rules:
  - Process chunks one at a time. After each extract_chunk, you MAY retry once
    if it failed; otherwise move to the next pending title.
  - Do NOT invent records; if a chunk fails twice, leave it and continue.
  - When list_pending() returns "[]", immediately call done().
  - Always output exactly one Thought block and one Action line per step.

Example step:
Thought: I should check what chunks still need processing.
Action: list_pending()
"""

ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", re.S)


class ReActAgent:
    """Single-agent ReAct loop.

    State:
      processed: title -> record dict  (successfully extracted)
      failed:    title -> retry count  (accumulated failures, skipped after reaching max_retries)
    """

    def __init__(self, data: Dict[str, str], max_retries: int = 2):
        self.data = data
        self.processed: Dict[str, Dict[str, Any]] = {}
        self.failed: Dict[str, int] = {}
        self.max_retries = max_retries
        self.n_llm_calls = 0
        self.n_tool_calls = 0

    # ----- tools -----
    def _tool_extract_chunk(self, title: str) -> str:
        self.n_tool_calls += 1
        title = title.strip().strip('"').strip("'")
        if title not in self.data:
            return json.dumps({"ok": False, "error": f"title not found: {title[:60]}"}, ensure_ascii=False)
        if title in self.processed:
            return json.dumps({"ok": True, "record": self.processed[title], "cached": True}, ensure_ascii=False)
        code, _cont = code_from_title(title)
        content = self.data[title]
        try:
            obj = _extract_one(code, content)
            if obj:
                self.processed[title] = obj
                return json.dumps({"ok": True, "record": obj}, ensure_ascii=False)
            self.failed[title] = self.failed.get(title, 0) + 1
            return json.dumps({"ok": False, "error": "empty airway_point"})
        except Exception as e:
            self.failed[title] = self.failed.get(title, 0) + 1
            return json.dumps({"ok": False, "error": str(e)[:120]}, ensure_ascii=False)

    def _tool_list_pending(self) -> str:
        titles = [t for t in self.data.keys()
                  if t not in self.processed and self.failed.get(t, 0) < self.max_retries]
        return json.dumps(titles, ensure_ascii=False)

    def _dispatch(self, action_name: str, action_arg: str) -> Optional[str]:
        action_name = action_name.strip()
        action_arg = action_arg.strip()
        try:
            if action_name == "extract_chunk":
                # Support both extract_chunk("title") and extract_chunk(title) syntax
                arg = action_arg.strip().strip('"').strip("'")
                return self._tool_extract_chunk(arg)
            if action_name == "list_pending":
                return self._tool_list_pending()
            if action_name == "done":
                return "__DONE__"
        except Exception as e:
            return json.dumps({"ok": False, "error": f"dispatch error: {e}"}, ensure_ascii=False)
        return None

    def _llm_step(self, history: List[Dict[str, str]]) -> str:
        """Call LLM to generate next Thought+Action."""
        self.n_llm_calls += 1
        return call_llm(
            [{"role": "system", "content": REACT_SYSTEM_PROMPT}] + history,
            temperature=0,
        )

    def run(self, limit: Optional[int] = None, step_cap: Optional[int] = None) -> List[Dict[str, Any]]:
        """Drive the ReAct loop.

        Args:
          limit: Max chunks to process (for functional verification), None for all.
          step_cap: Max total steps (to prevent infinite loops), None estimates as 4*n_target+50.
        """
        n_target = len(self.data) if limit is None else min(limit, len(self.data))
        if step_cap is None:
            step_cap = 4 * n_target + 50
        print(f"[react] Processing {n_target}/{len(self.data)} chunks "
              f"(max_retries={self.max_retries}, step_cap={step_cap})")

        history: List[Dict[str, str]] = [{
            "role": "user",
            "content": (
                f"Please exhaustively extract all {n_target} airway chunks. "
                "Start by calling list_pending() to see what needs to be done, "
                "then process chunks one by one via extract_chunk(title). "
                "When list_pending() returns [], call done()."
            ),
        }]

        n_steps = 0
        last_done = -1
        while n_steps < step_cap:
            n_steps += 1
            try:
                reply = self._llm_step(history)
            except Exception as e:
                print(f"  [react] LLM step failed: {e}, retrying once")
                time.sleep(2)
                continue
            history.append({"role": "assistant", "content": reply})

            m = ACTION_RE.search(reply)
            if not m:
                history.append({"role": "user", "content":
                    "Invalid format. You MUST output 'Thought: ...' then 'Action: tool_name(args)'. Try again."})
                continue

            action_name = m.group(1).strip()
            action_arg = m.group(2).strip()
            obs = self._dispatch(action_name, action_arg)
            if obs == "__DONE__":
                print(f"[react] agent called done(), terminating loop")
                break
            if obs is None:
                obs = json.dumps({"ok": False, "error": f"unknown action: {action_name}"})
            history.append({"role": "user", "content": f"Observation: {obs}"})

            done = len(self.processed)
            if done != last_done and (done % 20 == 0 or done >= n_target):
                print(f"  [react] step={n_steps} done={done}/{n_target} "
                      f"llm_calls={self.n_llm_calls} tool_calls={self.n_tool_calls}")
                last_done = done

            # Termination condition: reached limit or no pending chunks
            if limit is not None and done >= n_target:
                print(f"[react] reached limit={limit}, terminating loop")
                break
            try:
                pending = json.loads(self._tool_list_pending())
                if not pending:
                    print(f"[react] list_pending() is empty, all processable chunks done")
                    break
            except Exception:
                pass

        records = list(self.processed.values())
        print(f"[react] Completed: {len(records)} airways, {n_steps} steps, "
              f"LLM calls {self.n_llm_calls}, tool calls {self.n_tool_calls}")
        return records


def run_react(limit: Optional[int] = None, max_retries: int = 2) -> str:
    _require_key()
    t0 = time.time()
    data = load_clean_input()
    agent = ReActAgent(data, max_retries=max_retries)
    records = agent.run(limit=limit)
    path = save_pred(records, "react")
    print(f"[react] Total {len(records)} records, elapsed {time.time()-t0:.1f}s")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N chunks (for functional verification); omit for full run")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Max retries per chunk (skip after reaching limit)")
    args = ap.parse_args()
    run_react(limit=args.limit, max_retries=args.max_retries)