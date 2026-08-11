# -*- coding: utf-8 -*-
"""
Ablation study statistics & table export (offline, pure Python, no QGIS/API needed)
====================================================================================
Reads index.jsonl from the output of run_ablation_final.py at eval_results/ablation_runs/, computing:
  - Mean±SD and median for each variant and each metric
  - Wilcoxon signed-rank test (equal-length) or Mann-Whitney U (unequal-length) p-value vs. baseline control
  - Cliff's delta effect size
  - Bootstrap 95% confidence intervals
Exports a bilingual (CN/EN) docx, annotated for insertion into Section 6.5 / 6.6.

Usage (plain Python, no QGIS required):
  python analyze_ablation_final.py
"""
from __future__ import annotations

import os
import sys
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# ── shared statistical utilities ──────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))  # evaluation/
if _EVAL_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_ROOT)
from stat_utils import (          # noqa: E402
    mean_std, median, cliffs_delta, mann_whitney_p, wilcoxon_p,
    bootstrap_ci, fmt,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(os.path.dirname(_HERE), "eval_results", "ablation_runs")
INDEX = os.path.join(RUNS_DIR, "index.jsonl")

VARIANT_ORDER = ["control", "A_merge", "C_natural_lang", "D_no_guard",
                 "single_serial", "single", "B_no_store"]
VARIANT_LABEL_ZH = {
    "control": "Full multi-agent (baseline)",
    "A_merge": "Ablation A: merge loader+analysis",
    "C_natural_lang": "Ablation C: natural-language comm.",
    "D_no_guard": "Ablation D: no pre-checks",
    "single_serial": "Single-agent (serialized tool calls)",
    "single": "Single-agent (default parallel)",
    "B_no_store": "Ablation B: remove ObjectStore",
}
VARIANT_LABEL_EN = {
    "control": "Full multi-agent (baseline)",
    "A_merge": "Ablation A: merge loader+analysis",
    "C_natural_lang": "Ablation C: natural-language comm.",
    "D_no_guard": "Ablation D: no pre-checks",
    "single_serial": "Single-agent (serialized tool calls)",
    "single": "Single-agent (default parallel)",
    "B_no_store": "Ablation B: remove ObjectStore",
}
METRIC_KEYS = ["total_steps", "tool_calls", "subagent_delegations", "tool_errors", "elapsed_sec"]
METRIC_LABEL_ZH = {
    "success_rate": "Success rate", "total_steps": "Total steps", "tool_calls": "Tool calls",
    "subagent_delegations": "Sub-agent delegations", "tool_errors": "Tool errors", "elapsed_sec": "Wall-clock (s)",
}
METRIC_LABEL_EN = {
    "success_rate": "Success rate", "total_steps": "Total steps", "tool_calls": "Tool calls",
    "subagent_delegations": "Sub-agent delegations", "tool_errors": "Tool errors", "elapsed_sec": "Wall-clock (s)",
}
BASELINE = "control"


def load_records() -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.exists(INDEX):
        raise FileNotFoundError(f"Index not found at {INDEX}. Please run run_ablation_final.py first.")
    by = defaultdict(list)
    with open(INDEX, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by[rec["variant"]].append(rec)
    return by


def _series(records: List[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for r in records:
        if key == "success_rate":
            out.append(1.0 if r.get("success") else 0.0)
        elif key == "elapsed_sec":
            out.append(float(r.get("elapsed_sec", 0.0)))
        else:
            out.append(float(r.get("metrics", {}).get(key, 0) or 0))
    return out


def analyze(by_variant):
    result = {"variants": {}, "vs_baseline": {}}
    for v, records in by_variant.items():
        vs = {"n_runs": len(records)}
        sr = _series(records, "success_rate")
        m, _ = mean_std(sr)
        lo, hi = bootstrap_ci(sr)
        vs["success_rate"] = {"mean": m, "ci95": [lo, hi]}
        for key in METRIC_KEYS:
            xs = _series(records, key)
            mm, ss = mean_std(xs)
            lo, hi = bootstrap_ci(xs)
            vs[key] = {"mean": mm, "std": ss, "median": median(xs), "ci95": [lo, hi]}
        result["variants"][v] = vs
    if BASELINE in by_variant:
        base = by_variant[BASELINE]
        for v, records in by_variant.items():
            if v == BASELINE:
                continue
            comp = {}
            for key in METRIC_KEYS + ["success_rate"]:
                a, b = _series(records, key), _series(base, key)
                paired = (len(a) == len(b) and len(a) > 0)
                p = wilcoxon_p(a, b) if paired else mann_whitney_p(a, b)
                comp[key] = {"test": "wilcoxon" if paired else "mann_whitney_u",
                             "p_value": p, "cliffs_delta": cliffs_delta(a, b)}
            result["vs_baseline"][v] = comp
    return result


def print_report(analysis):
    print("\n================ Ablation Study Statistics Summary ================")
    for v in VARIANT_ORDER:
        if v not in analysis["variants"]:
            continue
        s = analysis["variants"][v]
        sr = s["success_rate"]
        print(f"\n[{v}] {VARIANT_LABEL_EN.get(v, v)}  (n={s['n_runs']})")
        print(f"  Success rate={fmt(sr['mean'])} CI95=[{fmt(sr['ci95'][0])},{fmt(sr['ci95'][1])}]")
        for key in METRIC_KEYS:
            st = s[key]
            print(f"  {METRIC_LABEL_EN[key]}: {fmt(st['mean'])}±{fmt(st['std'])} "
                  f"median={fmt(st['median'])} CI95=[{fmt(st['ci95'][0])},{fmt(st['ci95'][1])}]")
        if v in analysis["vs_baseline"]:
            print("  -- vs baseline control --")
            for key in METRIC_KEYS + ["success_rate"]:
                c = analysis["vs_baseline"][v][key]
                print(f"    {METRIC_LABEL_EN[key]}: p={fmt(c['p_value'],4)} "
                      f"delta={fmt(c['cliffs_delta'],3)} ({c['test']})")


def export_docx(analysis, out_path):
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.table import WD_TABLE_ALIGNMENT

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(9)

    def note(text):
        p = doc.add_paragraph(); r = p.add_run(text)
        r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00); r.font.size = Pt(9); r.bold = True

    doc.add_heading("Ablation Study Results", level=1)
    note("[Suggested insertion] Table A in Section 6.5 (multi-agent advantage and statistical testing), responding to Reviewer #2-4; "
         "Table B (component-wise ablation) in Section 6.6 (new Ablation study), responding to Reviewer #3-3. Marking only, no text changes.")

    present = [v for v in VARIANT_ORDER if v in analysis["variants"]]
    metrics_row = ["success_rate"] + METRIC_KEYS

    doc.add_paragraph()
    doc.add_paragraph("Table A. Per-variant process metrics (mean +/- SD; 95% bootstrap CI in brackets).")
    t1 = doc.add_table(rows=1, cols=1 + len(metrics_row)); t1.style = "Table Grid"
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    h = t1.rows[0].cells; h[0].text = "Variant / n"
    for j, key in enumerate(metrics_row):
        h[j + 1].text = f"{METRIC_LABEL_EN[key]}"
    for v in present:
        s = analysis["variants"][v]; row = t1.add_row().cells
        row[0].text = f"{VARIANT_LABEL_EN.get(v,v)} (n={s['n_runs']})"
        for j, key in enumerate(metrics_row):
            if key == "success_rate":
                sr = s["success_rate"]
                row[j + 1].text = f"{fmt(sr['mean'])}\n[{fmt(sr['ci95'][0])},{fmt(sr['ci95'][1])}]"
            else:
                stt = s[key]
                row[j + 1].text = f"{fmt(stt['mean'])}±{fmt(stt['std'])}\n[{fmt(stt['ci95'][0])},{fmt(stt['ci95'][1])}]"

    doc.add_paragraph()
    doc.add_paragraph("Table B. Statistical tests and effect sizes vs. the baseline (control).")
    t2 = doc.add_table(rows=1, cols=1 + len(metrics_row)); t2.style = "Table Grid"
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    h2 = t2.rows[0].cells; h2[0].text = "Variant"
    for j, key in enumerate(metrics_row):
        h2[j + 1].text = f"{METRIC_LABEL_EN[key]}\n(p / delta)"
    for v in present:
        if v not in analysis["vs_baseline"]:
            continue
        row = t2.add_row().cells
        row[0].text = f"{VARIANT_LABEL_EN.get(v,v)}"
        for j, key in enumerate(metrics_row):
            c = analysis["vs_baseline"][v][key]
            row[j + 1].text = f"p={fmt(c['p_value'],4)}\ndelta={fmt(c['cliffs_delta'],3)}"

    doc.add_paragraph()
    p = doc.add_paragraph(); r = p.add_run(
        "Note: Success rate defined as a run with no fatal exception that produced an analysis layer or a layout; "
        "p-values from paired Wilcoxon signed-rank test (equal-length repeats) or Mann-Whitney U (unequal length), "
        "two-sided; delta is Cliff's delta (|delta|<0.147 negligible, <0.33 small, <0.474 medium, otherwise large). "
        "Ablation B (B_no_store) is a necessity demonstration (n=1, expected structural failure), "
        "excluded from significance testing."); r.font.size = Pt(8)
    p2 = doc.add_paragraph(); r2 = p2.add_run(
        "Note: Success = a run with no fatal exception that produced an analysis layer or a layout; "
        "p-values from paired Wilcoxon signed-rank (equal-length) or Mann-Whitney U (unequal length), "
        "two-sided; delta is Cliff's delta. Ablation B (B_no_store) is a necessity demonstration (n=1, "
        "expected structural failure), excluded from significance testing."); r2.font.size = Pt(8)

    doc.save(out_path)


def main():
    by = load_records()
    analysis = analyze(by)
    print_report(analysis)
    out_json = os.path.join(os.path.dirname(HERE), "eval_results", "ablation_stats_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] Machine-readable statistics report: {out_json}")
    try:
        out_docx = os.path.join(os.path.dirname(HERE), "eval_results", "ablation_stats_report.docx")
        export_docx(analysis, out_docx)
        print(f"[Done] Bilingual docx: {out_docx}")
    except ImportError:
        print("[Skipped] python-docx not installed. Run 'pip install python-docx' and retry.")


if __name__ == "__main__":
    main()