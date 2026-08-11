# -*- coding: utf-8 -*-
"""
Baseline 1: Rule-based deterministic parser (no LLM)
==================================================================
Uses regex + layout rules to parse airways directly from cleaned JSON.
Represents the "traditional deterministic parser" baseline.
Consumes table_with_title-new1.json, produces prediction JSON with same schema as multi-agent.

Design notes (matching FAA airway report layout):
  - HTML tables: parse row-by-row; col1=from, col2=to, col3=MEA, col4=MAA
  - colspan rows or bare code rows => new airway within same block (split)
  - Plain text: line-by-line "from  to  altitude" or paired names + trailing altitude block
  - Continued pages -CONTINUED => merge by airway code

Usage:
  python baseline_rule_based.py
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from baselines_common import (
    load_clean_input, code_from_title, parse_point_cell, norm_alt,
    merge_continued, save_pred, _AIRWAY_CODE_RE, _NOISE_ROW_RE,
)

_TR_RE = re.compile(r"<tr>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_COLSPAN_RE = re.compile(r'<td[^>]*colspan=', re.IGNORECASE)


def _clean_cell(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).replace("&nbsp;", " ").strip()


def _is_bare_code_row(cells: List[str]) -> str:
    """Check if a row is a 'bare airway code' (new airway start). Returns code or ''."""
    non_empty = [c for c in cells if c.strip()]
    if len(non_empty) == 1:
        m = _AIRWAY_CODE_RE.fullmatch(non_empty[0].strip())
        if m:
            return non_empty[0].strip()
    return ""


def parse_html_table(code: str, html: str) -> List[Dict[str, Any]]:
    """Parse an HTML table into [records...] (may contain multiple airways in same block)."""
    records: List[Dict[str, Any]] = []
    cur = {"airway_code": code, "MEA": "none", "MAA": "none", "airway_point": []}
    mea_set, maa_set = [], []

    def flush():
        nonlocal cur, mea_set, maa_set
        if cur["airway_point"]:
            cur["MEA"] = mea_set[0] if mea_set else "none"
            cur["MAA"] = maa_set[0] if maa_set else "none"
            records.append(cur)
        cur = {"airway_code": code, "MEA": "none", "MAA": "none", "airway_point": []}
        mea_set, maa_set = [], []

    for tr in _TR_RE.findall(html):
        tds = [_clean_cell(x) for x in _TD_RE.findall(tr)]
        if not tds:
            continue
        # colspan row or bare code row => new airway
        new_code = _is_bare_code_row(tds)
        if _COLSPAN_RE.search(tr) or new_code:
            colspan_code = new_code
            if not colspan_code:
                joined = " ".join(tds)
                hit = _AIRWAY_CODE_RE.search(joined)
                colspan_code = hit.group(1) if hit else ""
            if colspan_code:
                flush()
                cur["airway_code"] = colspan_code
                continue
        # Normal segment row
        from_pt = parse_point_cell(tds[0]) if len(tds) >= 1 else None
        to_pt = parse_point_cell(tds[1]) if len(tds) >= 2 else None
        mea = norm_alt(tds[2]) if len(tds) >= 3 else "none"
        maa = norm_alt(tds[3]) if len(tds) >= 4 else "none"
        if from_pt:
            if not cur["airway_point"] or cur["airway_point"][-1]["name"].upper() != from_pt["name"].upper():
                cur["airway_point"].append(from_pt)
        if to_pt:
            cur["airway_point"].append(to_pt)
        if mea != "none":
            mea_set.append(mea)
        if maa != "none":
            maa_set.append(maa)
    flush()
    return records


def parse_plain_text(code: str, text: str) -> List[Dict[str, Any]]:
    """Parse plain text block. Lines may be 'FROM ... TO ... MEA MAA' or paired point names."""
    records: List[Dict[str, Any]] = []
    cur = {"airway_code": code, "MEA": "none", "MAA": "none", "airway_point": []}
    mea_first = "none"
    maa_first = "none"

    for raw in text.split("\n"):
        line = raw.strip()
        if not line or _NOISE_ROW_RE.match(line):
            continue
        # Inline embedded new airway code
        bare = _AIRWAY_CODE_RE.fullmatch(line)
        if bare:
            if cur["airway_point"]:
                cur["MEA"], cur["MAA"] = mea_first, maa_first
                records.append(cur)
            cur = {"airway_code": line, "MEA": "none", "MAA": "none", "airway_point": []}
            mea_first = maa_first = "none"
            continue
        # Extract trailing altitudes
        alts = re.findall(r"\d{3,5}", line.split(",")[-1]) if "," in line else re.findall(r"\d{3,5}$", line)
        if mea_first == "none":
            am = re.search(r"(\d{3,5})", line)
            if am:
                mea_first = am.group(1)
        p = parse_point_cell(line)
        if p:
            if not cur["airway_point"] or cur["airway_point"][-1]["name"].upper() != p["name"].upper():
                cur["airway_point"].append(p)
    if cur["airway_point"]:
        cur["MEA"], cur["MAA"] = mea_first, maa_first
        records.append(cur)
    return records


def run_rule_based() -> str:
    t0 = time.time()
    data = load_clean_input()
    all_records: List[Dict[str, Any]] = []
    for title, content in data.items():
        code, _cont = code_from_title(title)
        if code.upper().startswith("NONE_TABLE") or code.upper().startswith("NONE_TEXT"):
            # Noise block: try to find code from content, else skip
            hit = _AIRWAY_CODE_RE.search(content or "")
            if not hit:
                continue
            code = hit.group(1)
        if "<table" in (content or "").lower():
            recs = parse_html_table(code, content)
        else:
            recs = parse_plain_text(code, content)
        all_records.extend(recs)
    merged = merge_continued(all_records)
    path = save_pred(merged, "rule_based")
    print(f"[rule_based] elapsed {time.time()-t0:.2f}s, {len(merged)} airways after merge")
    return path


if __name__ == "__main__":
    run_rule_based()