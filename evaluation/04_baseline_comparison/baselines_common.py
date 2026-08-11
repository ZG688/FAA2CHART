# -*- coding: utf-8 -*-
"""
Strong baselines comparison — shared utilities
==================================================================
Five strong baselines (all targeting "extraction phase: cleaned PDF JSON → structured airway JSON"):

  Paradigm baselines (different paradigms):
    rule_based   Deterministic parser (regex + layout rules, no LLM)
    rag          Retrieval-augmented extraction (chunk + keyword retrieval + LLM single extraction)
    codegen      Direct code generation (LLM writes parser, then execute)

  Framework baselines (different general-purpose multi-agent frameworks):
    react        ReAct single agent (Reasoning+Acting loop, langchain implementation)
    autogen      AutoGen conversational multi-agent (GroupChat collaboration)

Fairness guarantees (critical):
  - Unified input: all consume the same cleaned result table_with_title-new1.json
  - Unified output: list[{airway_code, MEA, MAA, airway_point:[{name,region,type,position}]}]
  - Unified scoring: evaluated by evaluate_extraction_v2.py after production
  This allows direct comparison with multi-agent/single-model extraction results.

This module provides:
  - Input/output paths and schema constants
  - HTML table / plain text line parsing primitives (reused by rule_based)
  - Continued page (-CONTINUED) merging, same-block multi-airway splitting
  - JSON normalization and saving
"""
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------ Paths
DATA_ROOT = os.environ.get("FAA2CHART_DATA_ROOT", "PATH_TO_YOUR_DATA_ROOT")
CLEAN_INPUT = os.path.join(DATA_ROOT, "airway_data", "text", "table_with_title-new1.json")

# Output directory for this project
OUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 04 baseline comparison
PRED_OUT_DIR = os.path.join(DATA_ROOT, "airway_data", "text")  # Same directory as other predictions for evaluation

# Prediction output filenames for each baseline (consistent with PRED_FILES naming)
# Paradigm baselines
PRED_FILES = {
    "rule_based": "parsed_routes_baseline_rule_based.json",
    "rag":        "parsed_routes_baseline_rag.json",
    "codegen":    "parsed_routes_baseline_codegen.json",
    # Framework baselines (general-purpose multi-agent frameworks)
    "react":      "parsed_routes_baseline_react.json",
    "autogen":    "parsed_routes_baseline_autogen.json",
}

# Point type keywords (used to split type from "NAME, RG TYPE")
_TYPE_TOKENS = ["VOR/DME", "VORTAC", "NDB/DME", "MARINE NDB", "FAN MARKER",
                "TACAN", "VOR", "DME", "NDB", "FIX", "WP", "INTXN", "WAYPOINT"]
_TYPE_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in _TYPE_TOKENS) + r")\b")

# Noise rows (header/description)
_NOISE_ROW_RE = re.compile(
    r"^(AIRWAY SEGMENT|CHANGEOVER POINTS|FROM|TO|DISTANCE|COP\b|MOCA|MCA|MRA|"
    r"GNSS|MSL|HF COMMS|VHF|UHF|\*)", re.IGNORECASE)

# Airway code (used for same-block multi-airway splitting and title extraction)
_AIRWAY_CODE_RE = re.compile(r"\b([A-Z]{1,3}\d{1,4}[A-Z]?|RTE\d+|BR\d+V?|AR\d+)\b")


def load_clean_input(path: str = CLEAN_INPUT) -> Dict[str, str]:
    """Load cleaned {title: HTML/plain text} dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def code_from_title(title: str) -> str:
    """Extract airway code from title key.
    e.g. '95.116 AMBER FEDERAL AIRWAY A16' -> 'A16'
         'RTE12' -> 'RTE12'; 'BR53V' -> 'BR53V'
         Strip '- CONTINUED' suffix."""
    t = title.strip()
    cont = False
    if t.upper().endswith("- CONTINUED") or t.upper().endswith("-CONTINUED"):
        cont = True
        t = re.sub(r"\s*-\s*CONTINUED$", "", t, flags=re.IGNORECASE)
    # Normalize "R O U T E" -> "ROUTE" (OCR-introduced spaces)
    t = re.sub(r"\bR\s*O\s*U\s*T\s*E\b", "ROUTE", t, flags=re.IGNORECASE)
    # Normalize RNAV: ROUTET200 / ROUTE T200 / ROUTE T 200 → T200
    t = re.sub(r"\bROUTE\s*T\s*(\d{3,4})\b", r"T\1", t, flags=re.IGNORECASE)
    t = re.sub(r"\bROUTE\s*Q\s*(\d{3,4})\b", r"Q\1", t, flags=re.IGNORECASE)
    # Direct code match
    if re.fullmatch(r"[A-Z]{1,3}\d{1,4}[A-Z]?|RTE\d+|BR\d+V?|AR\d+", t):
        return (t, cont)
    # Take last matching code from long title
    hits = _AIRWAY_CODE_RE.findall(t)
    if hits:
        return (hits[-1], cont)
    return (t, cont)


def norm_alt(v: Optional[str]) -> str:
    """Altitude value -> pure digit string; 'none' if missing."""
    if v is None:
        return "none"
    m = re.search(r"\d{3,5}", str(v).replace(",", ""))
    return m.group(0) if m else "none"


def parse_point_cell(cell: str) -> Optional[Dict[str, Any]]:
    """Parse 'GRAND TURK, TC VORTAC' into point dict.
    Returns None if no valid point name found."""
    if not cell:
        return None
    # Take first line (rowspan cells may contain multi-line comments)
    line = cell.split("\n")[0].strip().lstrip("*")
    if not line or _NOISE_ROW_RE.match(line):
        return None
    # Pattern: "NAME, RG TYPE"
    m = re.match(r"^(.*?),\s*([A-Z]{2})\s+(.*)$", line)
    if m:
        name = m.group(1).strip().lstrip("*")
        region = m.group(2).strip()
        rest = m.group(3).strip()
        tm = _TYPE_RE.search(rest)
        ptype = tm.group(1) if tm else rest.split()[0] if rest else "none"
        if not name:
            return None
        return {"name": name, "region": region, "type": ptype,
                "position": ["none", "none"]}
    # No region code: "NAME TYPE"
    tm = _TYPE_RE.search(line)
    if tm:
        name = line[:tm.start()].strip().rstrip(",").lstrip("*")
        if name:
            return {"name": name, "region": "none", "type": tm.group(1),
                    "position": ["none", "none"]}
    return None


def dedup_consecutive(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only one instance of consecutive duplicate points (by name)."""
    out = []
    for p in points:
        if out and out[-1]["name"].upper() == p["name"].upper():
            continue
        out.append(p)
    return out


def normalize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Unify schema before output: fill fields, remove empty airways."""
    clean = []
    for r in records:
        code = str(r.get("airway_code", "")).strip()
        pts = r.get("airway_point") or []
        pts = [p for p in pts if p and p.get("name")]
        pts = dedup_consecutive(pts)
        if not code or not pts:
            continue
        clean.append({
            "airway_code": code,
            "MEA": norm_alt(r.get("MEA")) if r.get("MEA") not in (None, "none") else "none",
            "MAA": norm_alt(r.get("MAA")) if r.get("MAA") not in (None, "none") else "none",
            "airway_point": [{
                "name": p.get("name", ""),
                "region": p.get("region", "none"),
                "type": p.get("type", "none"),
                "position": p.get("position", ["none", "none"]),
            } for p in pts],
        })
    return clean


def merge_continued(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge continued pages by airway_code: same code → merge, dedup head/tail. Preserve first-appearance order."""
    order: List[str] = []
    merged: Dict[str, Dict[str, Any]] = {}
    for r in records:
        code = r["airway_code"]
        if code not in merged:
            merged[code] = {"airway_code": code, "MEA": r.get("MEA", "none"),
                            "MAA": r.get("MAA", "none"),
                            "airway_point": list(r.get("airway_point", []))}
            order.append(code)
        else:
            prev = merged[code]["airway_point"]
            add = list(r.get("airway_point", []))
            if prev and add and prev[-1]["name"].upper() == add[0]["name"].upper():
                add = add[1:]
            prev.extend(add)
            if merged[code].get("MEA", "none") == "none" and r.get("MEA", "none") != "none":
                merged[code]["MEA"] = r["MEA"]
            if merged[code].get("MAA", "none") == "none" and r.get("MAA", "none") != "none":
                merged[code]["MAA"] = r["MAA"]
    return [merged[c] for c in order]


def save_pred(records: List[Dict[str, Any]], baseline_name: str) -> str:
    """Save prediction results to evaluation-readable directory, return path."""
    records = normalize_records(records)
    out_path = os.path.join(PRED_OUT_DIR, PRED_FILES[baseline_name])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[{baseline_name}] saved {len(records)} airways -> {out_path}")
    return out_path


# ==========================================================================
# Framework baseline (react / autogen) execution notes
# ==========================================================================
# 1. Dependencies (install in a venv separate from QGIS to avoid conflicts):
#      pip install requests
#      pip install pyautogen==0.2.40   # optional: use real AutoGen GroupChat
#    ReAct baseline needs no extra packages, pure requests + hand-written loop, faithful to Yao et al. (2023).
#
# 2. Environment variables (same as RAG/Codegen):
#      PowerShell:  $env:ANTHROPIC_API_KEY="your_key"
#      cmd:         set ANTHROPIC_API_KEY=your_key
#
# 3. Run a single baseline (recommend --limit 50 to verify first):
#      python baseline_react.py --limit 50
#      python baseline_autogen.py --limit 50
#
# 4. Full run + evaluate + export table (merged with existing 3 paradigm baselines):
#      python run_baselines_and_compare.py --run react autogen
#      python run_baselines_and_compare.py --eval-only  # evaluate only, skip re-run
#
# 5. Fairness constraints (identical to RAG/Codegen):
#      - Same cleaned input table_with_title-new1.json
#      - Same gold (FAA NASR Feb 2025)
#      - Same evaluation pipeline evaluate_extraction_v2.py
#      - Same underlying LLM (Claude 4.5 Opus, temperature=0)
#      - Same per-chunk extraction prompt (EXTRACT_PROMPT is literally identical in RAG/ReAct/AutoGen)
#      - Same output schema (airway_code, MEA, MAA, airway_point)
#    Only variable: the agent framework itself (ReAct single / AutoGen conversational)