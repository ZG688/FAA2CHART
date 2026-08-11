# -*- coding: utf-8 -*-
"""
Strong Baseline 3: Direct Code Generation
==================================================================
Represents the "let LLM write the parsing code" baseline: instead of extracting
one by one, the LLM generates a Python parsing function from a few examples,
then uses that function to batch-parse all cleaned chunks.

Fairness: consumes the same table_with_title-new1.json, produces the same schema.

★ Safety:
  1) API key is read from environment variable only, never hardcoded.
  2) LLM-generated code is executed in a 【restricted namespace】, only exposing
     re / json, blocking os/sys/open etc., to mitigate arbitrary code execution risk.
     Even so, please only run in a trusted environment and manually review generated_parser.py.

Usage:
    set ANTHROPIC_API_KEY=your_key
    python baseline_codegen.py
"""
from __future__ import annotations

import os
import re
import json
import time
from typing import Any, Dict, List

import requests

from baselines_common import (
    load_clean_input, code_from_title, save_pred, OUT_ROOT,
)

CLAUDE_URL = "https://api.anthropic.com/v1/messages"
MODEL_NAME = "claude-opus-4-5-20250929"

CODEGEN_PROMPT = r"""You are a senior Python engineer. Please write a function:

    def parse_airway(code: str, content: str) -> dict

Input content is an FAA airway HTML table with the following format characteristics:
  <table><tr><td>start_name, RG type</td><td>end_name, RG type</td><td>MEA</td><td>MAA</td></tr>
  Where "start_name, RG type" looks like "ACTIVE PASS, CA NDB" or "ELFEE, AK NDB/DME".
  Each <tr> is a segment: column 1=start, column 2=end, column 3=MEA altitude, column 4=MAA altitude.
  Altitudes may have a * prefix (e.g., *3000), extract only the numeric part.
  Some rows are annotation rows: containing MOCA, MRA, GNSS MEA, HF COMMS, *FOR THAT AIRSPACE, etc., skip these rows.
  Some <td> elements have a rowspan attribute, containing multiple waypoint names (separated by \n), parse each line.
  Some <td> elements contain annotations like "6200 - MOCA", ignore the annotation part, extract only the waypoint name.

Strict output format:
{{"airway_code": code, "MEA": "number or none", "MAA": "number or none",
  "airway_point": [{{"name": str, "region": "two-letter or none", "type": str, "position": ["none","none"]}}]}}

Parsing rules (must follow):
1. Use re to extract all <tr>...</tr>, extract <td>...</td> row by row
2. Point cell format "NAME, RG TYPE": use regex r'^([^,]+),\s*([A-Z]{{2}})\s+(.+)$' to extract name/region/type
   【Important】Do NOT split cell text by comma! The comma is part of the "NAME, RG" format.
3. If a cell has multiple lines (\n separated), parse each line separately, but skip lines containing MOCA/MRA/GNSS/HF COMMS/*FOR THAT
4. Altitude columns: use r'\d{{3,5}}' to extract numbers, ignore * prefix and -MOCA/-GNSS suffixes
5. Waypoints in order, consecutive duplicates (same name) kept only once
6. If content does not contain <table> tags, treat as plain text: one waypoint per line

Output only the function code, no explanation, no markdown fences. Reference samples:
{samples}
"""


def _require_key():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY not found. Set ANTHROPIC_API_KEY=your_key first")


def call_llm(prompt: str) -> str:
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
               "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    body = {"model": MODEL_NAME, "temperature": 0, "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(CLAUDE_URL, headers=headers, json=body, timeout=180)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def _build_samples(data: Dict[str, str], k: int = 8) -> str:
    """Select representative samples from real data (skip directory pages)."""
    # Skip none_table and non-airway entries, select first k real airway pages
    items = [(t, c) for t, c in data.items()
             if not t.startswith("none") and "FEDERAL AIRWAY" in t][:k]
    # If not enough, supplement with arbitrary entries
    if len(items) < k:
        extra = [(t, c) for t, c in data.items()
                 if not t.startswith("none") and "FEDERAL AIRWAY" not in t][:k - len(items)]
        items.extend(extra)
    out = []
    for title, content in items:
        code, _ = code_from_title(title)
        out.append(f"# code={code}  title={title[:60]}\n# content={content[:1200]}")
    return "\n\n".join(out)


def _safe_exec_parser(code_src: str):
    """Exec the generated code in a restricted namespace, returning the parse_airway function."""
    # re / json injected by sandbox; strip import re/json from generated code,
    # otherwise __import__ (blocked by sandbox) would cause ImportError.
    code_src = re.sub(
        r"(?m)^\s*(?:import\s+(?:re|json)(?:\s*,\s*(?:re|json))*"
        r"|from\s+(?:re|json)\s+import\s+.*)\s*$",
        "", code_src)
    banned = ["import os", "import sys", "open(", "__import__", "subprocess",
              "eval(", "exec(", "socket", "shutil", "pathlib"]
    low = code_src.lower()
    for b in banned:
        if b in low:
            raise ValueError(f"Generated code contains banned call: {b}, execution rejected")
    safe_globals = {"re": re, "json": json, "__builtins__": {
        "len": len, "range": range, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "set": set, "enumerate": enumerate,
        "zip": zip, "min": min, "max": max, "sorted": sorted, "any": any,
        "all": all, "map": map, "filter": filter, "isinstance": isinstance,
        "True": True, "False": False, "None": None,
    }}
    local_ns: Dict[str, Any] = {}
    exec(code_src, safe_globals, local_ns)
    fn = local_ns.get("parse_airway") or safe_globals.get("parse_airway")
    if not callable(fn):
        raise ValueError("Generated code does not contain a callable parse_airway")
    return fn


def run_codegen() -> str:
    _require_key()
    t0 = time.time()
    data = load_clean_input()

    prompt = CODEGEN_PROMPT.format(samples=_build_samples(data))
    code_src = _strip_code_fence(call_llm(prompt))

    # Save the generated parser for paper appendix and manual review
    gen_path = os.path.join(OUT_ROOT, "code", "generated_parser.py")
    with open(gen_path, "w", encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n# LLM-generated parser (codegen baseline), please review manually\nimport re, json\n\n")
        f.write(code_src)
    print(f"[codegen] Saved generated code -> {gen_path}")

    parse_airway = _safe_exec_parser(code_src)

    records: List[Dict[str, Any]] = []
    n_fail = 0
    n_empty = 0
    first_err = None
    for title, content in data.items():
        code, _cont = code_from_title(title)
        try:
            obj = parse_airway(code, content)
            if isinstance(obj, dict) and obj.get("airway_point"):
                records.append(obj)
            else:
                n_empty += 1
        except Exception as e:
            n_fail += 1
            if first_err is None:
                first_err = (title[:80], str(e))
    if first_err:
        print(f"  [codegen] First crash: {first_err[0]}")
        print(f"  [codegen] Error message: {first_err[1]}")
    path = save_pred(records, "codegen")
    print(f"[codegen] Extracted {len(records)} records, empty {n_empty} chunks, parse crash {n_fail} chunks, elapsed {time.time()-t0:.1f}s")
    return path


if __name__ == "__main__":
    run_codegen()