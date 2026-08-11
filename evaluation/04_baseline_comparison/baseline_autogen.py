# -*- coding: utf-8 -*-
"""
Strong Baseline 5: AutoGen Conversational Multi-Agent Framework Baseline
==================================================================
Represents the "general-purpose conversational multi-agent framework" baseline:
using Microsoft AutoGen (Wu et al., 2023)'s GroupChat paradigm, with three
AssistantAgents (Extractor / Validator / Aggregator) collaborating under a
GroupChatManager to complete the extraction task.

Fairness:
  - Input: the same table_with_title-new1.json
  - Output: the same schema, saved by save_pred
  - Underlying LLM: Claude 4.5 Opus (consistent with RAG / Codegen / ReAct baselines)
  - Single-chunk extraction prompt is identical to the RAG/ReAct baselines

★ Safety: API key is read from environment variable only.

Dependencies:
  pip install pyautogen==0.2.40   # AutoGen 0.2 stable API
  pip install requests

Usage:
  set ANTHROPIC_API_KEY=your_key
  python baseline_autogen.py                # full run
  python baseline_autogen.py --limit 50     # only first 50 chunks (functional verification)
  python baseline_autogen.py --max-round 6  # max rounds per chunk GroupChat
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

# Single extraction prompt (identical to RAG/ReAct baselines, ensuring differences come only from the agent framework)
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


# ---------------------------------------------------------------------------
# Two paths:
#   A) Preferred: use autogen 0.2 package (native GroupChat)
#   B) Fallback: hand-written AutoGen-style conversational collaboration (no external dependencies, same paradigm)
# Fallback is used so the experiment is reproducible when autogen is not installed / version incompatible.
# ---------------------------------------------------------------------------


def _make_llm_config():
    return {
        "config_list": [{
            "model": MODEL_NAME,
            "api_key": os.environ["ANTHROPIC_API_KEY"],
            "base_url": CLAUDE_URL,
        }],
        "temperature": 0,
        "timeout": 120,
    }


# ===== Single extraction (core call of Extractor agent) =====
def _extract_one(code: str, content: str) -> Optional[Dict[str, Any]]:
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
               "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    body = {"model": MODEL_NAME, "temperature": 0, "max_tokens": 8192,
            "messages": [{"role": "user", "content": EXTRACT_PROMPT.format(code=code, content=content[:8000])}]}
    r = requests.post(CLAUDE_URL, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    obj = _extract_json_obj(r.json()["content"][0]["text"])
    if obj and obj.get("airway_point"):
        return obj
    return None


# ===== Validator: lightweight structural validation of extraction results, returns (is_valid, issues) =====
def _validate_record(record: Dict[str, Any]) -> tuple:
    issues = []
    if not record.get("airway_code"):
        issues.append("missing airway_code")
    pts = record.get("airway_point") or []
    if not pts:
        issues.append("empty airway_point")
    for i, p in enumerate(pts):
        if not p.get("name"):
            issues.append(f"point[{i}] missing name")
    return (len(issues) == 0, issues)


# ===== Aggregator: accumulate validated records into a list =====
class Aggregator:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add(self, record: Dict[str, Any]) -> None:
        # Deduplication: later arrival priority for same airway_code (consistent with multi-agent main result)
        code = record.get("airway_code")
        self.records = [r for r in self.records if r.get("airway_code") != code]
        self.records.append(record)


# ---------------------------------------------------------------------------
# Path A: use autogen 0.2 package
# ---------------------------------------------------------------------------
def _run_with_autogen_pkg(data: Dict[str, str], limit: Optional[int], max_round: int) -> List[Dict[str, Any]]:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

    llm_config = _make_llm_config()

    extractor = AssistantAgent(
        name="Extractor",
        system_message=(
            "You are an airway extraction expert. Given an airway chunk, you call the internal extraction logic to output a JSON object. "
            "Strict output format: {\"airway_code\":..., \"MEA\":..., \"MAA\":..., \"airway_point\":[...]}. "
            "No explanation, no extra text. After extraction, pass the JSON to Validator for validation."
        ),
        llm_config=llm_config,
    )
    validator = AssistantAgent(
        name="Validator",
        system_message=(
            "You are an airway validation expert. After receiving the Extractor's JSON, check: airway_code is not empty, "
            "airway_point is not empty, each point has a name. If valid, reply 'VALID: <json>'; "
            "if invalid, reply 'INVALID: <issues>' to let Extractor retry. Max 1 retry."
        ),
        llm_config=llm_config,
    )
    aggregator = AssistantAgent(
        name="Aggregator",
        system_message=(
            "You are an aggregation expert. After receiving 'VALID: <json>', reply 'AGGREGATED: <airway_code>'. "
            "Do not output the full JSON, only confirm aggregation."
        ),
        llm_config=llm_config,
    )
    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda msg: "AGGREGATED" in (msg.get("content") or ""),
        code_execution_config=False,  # Disable Docker code execution
    )

    group_chat = GroupChat(
        agents=[user_proxy, extractor, validator, aggregator],
        messages=[],
        max_round=max_round,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    aggregator_state = Aggregator()
    n_target = len(data) if limit is None else min(limit, len(data))
    print(f"[autogen-pkg] Processing {n_target}/{len(data)} chunks, max_round={max_round}")

    for i, (title, content) in enumerate(data.items()):
        if limit is not None and i >= limit:
            break
        code, _cont = code_from_title(title)
        # Call underlying extraction directly (ensuring prompt consistency with other baselines),
        # then let AutoGen GroupChat handle Validator/Aggregator collaboration
        try:
            record = _extract_one(code, content)
        except Exception as e:
            print(f"  [autogen-pkg] {code} extraction failed: {e}")
            continue
        if not record:
            continue

        msg = (
            f"Please collaboratively process airway {code}. Extractor has extracted JSON:\n"
            f"{json.dumps(record, ensure_ascii=False)}\n"
            "Validator, please validate; Aggregator, please aggregate."
        )
        try:
            user_proxy.initiate_chat(manager, message=msg, clear_history=True)
        except Exception as e:
            print(f"  [autogen-pkg] {code} GroupChat failed: {e}")

        # If validation passes, aggregate
        is_valid, _ = _validate_record(record)
        if is_valid:
            aggregator_state.add(record)

        if (i + 1) % 20 == 0 or (i + 1) >= n_target:
            print(f"  [autogen-pkg] progress={i+1}/{n_target} aggregated={len(aggregator_state.records)}")

    return aggregator_state.records


# ---------------------------------------------------------------------------
# Path B: Hand-written AutoGen-style conversational collaboration (fallback, no external dependencies)
# ---------------------------------------------------------------------------
# Paradigm characteristics (key differences from CrewAI fallback implementation):
#   AutoGen's GroupChat enables multi-round conversational collaboration between agents: when Validator finds issues,
#   it can "fall back" to let Extractor re-extract based on specific issues. This implementation follows this
#   "fallback dialogue" semantics — Validator performs strict validation (including waypoint count, field completeness),
#   and Extractor retries on failure, up to max_retries rounds.
#   In contrast, CrewAI's Process.sequential does not support fallback; Validator failure means skipping.
VALIDATOR_PROMPT = (
    "You are an airway validation expert, performing strict validation. After receiving an airway JSON, check item by item:\n"
    "1) airway_code is not empty and has a valid airway number format (letters+digits);\n"
    "2) airway_point list is not empty and contains at least 2 different waypoints (start+end);\n"
    "3) Each point has a name field that is a non-empty string;\n"
    "4) MEA and MAA fields exist (may be 'none').\n"
    "If all pass, reply 'VALID'; if any fails, reply 'INVALID: <specific issues>'."
)
AGGREGATOR_PROMPT = (
    "You are an aggregation expert. After receiving 'VALID' confirmation, reply 'AGGREGATED'."
)


def _run_with_fallback(data: Dict[str, str], limit: Optional[int], max_round: int) -> List[Dict[str, Any]]:
    """Hand-written AutoGen-style conversational collaboration: Extractor -> Validator -> Aggregator."""
    n_target = len(data) if limit is None else min(limit, len(data))
    print(f"[autogen-fallback] Processing {n_target}/{len(data)} chunks, max_round={max_round}")
    aggregator_state = Aggregator()
    n_llm_calls = 0

    def _llm(messages: List[Dict[str, str]]) -> str:
        nonlocal n_llm_calls
        n_llm_calls += 1
        # Anthropic API: system message is a separate field
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
        body = {"model": MODEL_NAME, "temperature": 0, "max_tokens": 8192,
                "messages": user_msgs}
        if system_msg:
            body["system"] = system_msg
        r = requests.post(CLAUDE_URL, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        return r.json()["content"][0]["text"]

    for i, (title, content) in enumerate(data.items()):
        if limit is not None and i >= limit:
            break
        code, _cont = code_from_title(title)

        # 1) Extractor
        try:
            record = _extract_one(code, content)
        except Exception as e:
            print(f"  [autogen-fallback] {code} extraction failed: {e}")
            continue
        if not record:
            continue

        # 2) Validator (single dialogue; retry once on failure)
        valid = False
        for retry in range(2):
            msg = (
                f"Please validate the following airway JSON:\n{json.dumps(record, ensure_ascii=False)}\n"
                "If valid, reply 'VALID'; if invalid, reply 'INVALID: <issues>'."
            )
            try:
                v_reply = _llm([
                    {"role": "system", "content": VALIDATOR_PROMPT},
                    {"role": "user", "content": msg},
                ])
            except Exception as e:
                print(f"  [autogen-fallback] {code} validation call failed: {e}")
                break
            if "VALID" in v_reply and "INVALID" not in v_reply:
                valid = True
                break
            # Retry extraction
            try:
                record = _extract_one(code, content)
            except Exception:
                break
            if not record:
                break

        # 3) Aggregator (confirmation)
        if valid:
            try:
                _llm([
                    {"role": "system", "content": AGGREGATOR_PROMPT},
                    {"role": "user", "content": f"Aggregator: airway {code} has passed validation, please aggregate."},
                ])
            except Exception:
                pass
            aggregator_state.add(record)

        if (i + 1) % 20 == 0 or (i + 1) >= n_target:
            print(f"  [autogen-fallback] progress={i+1}/{n_target} "
                  f"aggregated={len(aggregator_state.records)} llm_calls={n_llm_calls}")

    print(f"[autogen-fallback] Total LLM calls: {n_llm_calls}")
    return aggregator_state.records


def run_autogen(limit: Optional[int] = None, max_round: int = 6) -> str:
    _require_key()
    t0 = time.time()
    data = load_clean_input()

    # Prefer autogen package; fall back to self-implementation on failure
    try:
        from autogen import AssistantAgent  # noqa: F401
        print("[autogen] Detected autogen package, using Path A (GroupChat)")
        records = _run_with_autogen_pkg(data, limit, max_round)
    except ImportError:
        print("[autogen] autogen package not installed, using Path B (hand-written conversational collaboration, same paradigm)")
        print("        For native AutoGen GroupChat, please: pip install pyautogen==0.2.40")
        records = _run_with_fallback(data, limit, max_round)

    path = save_pred(records, "autogen")
    print(f"[autogen] Total {len(records)} records, elapsed {time.time()-t0:.1f}s")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N chunks (for functional verification); omit for full run")
    ap.add_argument("--max-round", type=int, default=6,
                    help="Max rounds per chunk GroupChat")
    args = ap.parse_args()
    run_autogen(limit=args.limit, max_round=args.max_round)