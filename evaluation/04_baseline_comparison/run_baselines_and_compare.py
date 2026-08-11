# -*- coding: utf-8 -*-
"""
Strong baselines unified driver + comparison table export
==================================================================
1) Optionally run six baselines
   - Paradigm baselines: rule_based (no API) / rag / codegen (requires ANTHROPIC_API_KEY)
   - Framework baselines (general-purpose multi-agent frameworks):
     react / autogen (requires ANTHROPIC_API_KEY)
2) Reuse evaluate_extraction_v2 from 01_extraction evaluation, scoring each baseline
   and multi-agent primary result with the same gold and same metrics
3) Export bilingual comparison docx, annotated for insertion into Section 6.5 (strong baseline comparison)

Usage:
    # Evaluate existing predictions only (don't re-run baselines)
    python run_baselines_and_compare.py --eval-only
    # Run rule_based then evaluate
    python run_baselines_and_compare.py --run rule_based
    # Run all five baselines (needs API) then evaluate
    set ANTHROPIC_API_KEY=your_key
    python run_baselines_and_compare.py --run rule_based rag codegen react autogen
    # Framework baselines can add --limit 50 for smoke test
    python run_baselines_and_compare.py --run react --limit 50
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import importlib.util
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.dirname(HERE)  # 04 baseline comparison
sys.path.insert(0, HERE)

from baselines_common import PRED_FILES, PRED_OUT_DIR

# Evaluator path (reuse validated script, avoid duplicate implementation)
EVAL_SCRIPT = os.path.join(
    OUT_ROOT, "..", "01_extraction",
    "evaluate_extraction_v2.py")

# Multi-agent primary result (best-performing Claude4.5 as reference)
MULTIAGENT_REF = {
    "MultiAgent(Claude4.5)": "parsed_routes_newnew_5_Claude4.5_updated_merged.json",
}


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("eval_v2", os.path.abspath(EVAL_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _accelerate_to_id(mod)
    return mod


def _accelerate_to_id(mod):
    """Performance fix: evaluator to_id does full-library Levenshtein for every unmapped
    point, which is extremely slow when rule-based has many unmapped points.
    Same-name points recur across thousands of airways; memoizing by (name, gold_id)
    dramatically reduces fuzzy match calls without changing any results (pure cache)."""
    _orig_to_id = mod.to_id
    _cache: Dict[Any, Any] = {}

    def cached_to_id(name, cg):
        key = (str(name), id(cg))
        if key in _cache:
            return _cache[key]
        val = _orig_to_id(name, cg)
        _cache[key] = val
        return val

    mod.to_id = cached_to_id

    # Result-equivalent length filtering to accelerate fuzzy_match:
    # Levenshtein(a,b) >= |len(a)-len(b)|, so candidates with length difference > max_dist
    # cannot match. Skip them directly — no change to final result, just avoids wasted DP.
    _orig_lev = mod.levenshtein

    def fast_fuzzy_match(n, ids, name2id, max_dist=2):
        if max_dist <= 0:
            return None
        ln = len(n)
        best, best_d = None, max_dist + 1
        for key in name2id:
            if abs(len(key) - ln) >= best_d:
                continue
            d = _orig_lev(n, key)
            if d < best_d:
                best_d, best = d, key
        if best:
            return name2id[best]
        for key in ids:
            if abs(len(key) - ln) >= best_d:
                continue
            d = _orig_lev(n, key)
            if d < best_d:
                best_d, best = d, key
        return best

    mod.fuzzy_match = fast_fuzzy_match


def run_baseline(name: str, limit=None):
    if name == "rule_based":
        from baseline_rule_based import run_rule_based
        return run_rule_based()
    if name == "rag":
        from baseline_rag import run_rag
        return run_rag()
    if name == "codegen":
        from baseline_codegen import run_codegen
        return run_codegen()
    # Framework baselines (general-purpose multi-agent frameworks)
    if name == "react":
        from baseline_react import run_react
        return run_react(limit=limit)
    if name == "autogen":
        from baseline_autogen import run_autogen
        return run_autogen(limit=limit)

    raise ValueError(f"Unknown baseline: {name}")


def evaluate_all(targets: Dict[str, str], coord_km: float, boot: int) -> Dict[str, Any]:
    ev = _load_evaluator()
    feb = ev.load_airways(ev.FEB_CSV_DIR)
    nov_coord = ev.load_coords(ev.NOV_CSV_DIR)
    print(f"Feb airway gold: {len(feb['awy_seq'])}  Nov coords: {len(nov_coord['coord'])}")

    report = {}
    for name, fn in targets.items():
        path = os.path.join(PRED_OUT_DIR, fn)
        if not os.path.exists(path):
            print(f"  [skip] Prediction file not found: {path}")
            continue
        res = ev.evaluate(path, feb, nov_coord, coord_km)
        s = ev.summarize(res, n_boot=boot)
        report[name] = {
            "n_pred_codes": res["n_pred_codes"],
            "code_F1": s["code_F1"], "code_P": s["code_P"], "code_R": s["code_R"],
            "waypoint_P": s["waypoint_P"], "OSV": s["OSV"],
            "order_tau": s["order_tau"], "seq_ratio": s["seq_ratio"],
            "MEA_acc": s["MEA_acc"], "MAA_acc": s["MAA_acc"],
            "trunc_pct": s["trunc_pct"],
            "ci95": {k: list(v) for k, v in s["ci"].items()},
        }
        print(f"  [{name}] code_F1={s['code_F1']:.3f} wp_P={s['waypoint_P']:.3f} "
              f"OSV={s['OSV']:.3f} tau={s['order_tau']:.3f} "
              f"MEA={s['MEA_acc']:.3f} MAA={s['MAA_acc']:.3f} codes={res['n_pred_codes']}")
    return report


def export_docx(report: Dict[str, Any], out_path: str):
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.table import WD_TABLE_ALIGNMENT

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(9)
    p = doc.add_paragraph(); r = p.add_run(
        "[Insertion point] Insert into Section 6.5 (strong baseline comparison). "
        "Annotation only, no modifications to the original text.")
    r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00); r.bold = True

    doc.add_heading("Baseline Comparison for Extraction (Bilingual)", level=1)
    doc.add_paragraph(
        "Table. Extraction performance of all methods under the same FAA NASR (Feb 2025) "
        "ground truth and identical metrics.")

    cols = [("Method", None), ("#Codes", "n_pred_codes"),
            ("code_F1", "code_F1"), ("waypoint_P", "waypoint_P"),
            ("OSV", "OSV"), ("order_tau", "order_tau"),
            ("MEA_acc", "MEA_acc"), ("MAA_acc", "MAA_acc"),
            ("trunc%", "trunc_pct")]
    t = doc.add_table(rows=1, cols=len(cols)); t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, (label, _k) in enumerate(cols):
        t.rows[0].cells[j].text = label
    for name, s in report.items():
        row = t.add_row().cells
        row[0].text = name
        for j, (_label, key) in enumerate(cols[1:], start=1):
            v = s.get(key)
            if key == "n_pred_codes":
                row[j].text = str(v)
            elif key == "trunc_pct":
                row[j].text = f"{v:.1%}" if v is not None else "-"
            else:
                row[j].text = f"{v:.3f}" if v is not None else "-"

    note = doc.add_paragraph(); rr = note.add_run(
        "Note: All methods consume the same cleaned input and emit the same schema; content-level "
        "metrics use Feb 2025 NASR. code_F1/waypoint_P/OSV/order_tau/MEA/MAA are reliable extraction "
        "metrics (waypoint recall is excluded because FAA selective charting vs. the full database is "
        "not a comparable reference). rule_based uses no LLM; RAG only processes retrieved chunks; "
        "codegen executes an LLM-generated parser. "
        "react/autogen are general-purpose multi-agent framework baselines; all use "
        "Claude 4.5 Opus as the underlying LLM with the identical per-chunk extraction "
        "prompt as RAG/codegen, differing only in the agent framework itself.")
    rr.font.size = Pt(8)

    doc.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[],
                    help="Baselines to run: rule_based rag codegen react autogen")
    ap.add_argument("--eval-only", action="store_true", help="Skip running, only evaluate existing predictions")
    ap.add_argument("--coord-km", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None,
                    help="Framework baselines (react/autogen) process only first N chunks (smoke test)")
    args = ap.parse_args()

    for name in args.run:
        print(f"\n===== Running baseline {name} =====")
        run_baseline(name, limit=args.limit)

    # Evaluation targets: six baselines + multi-agent reference
    targets = dict(MULTIAGENT_REF)
    for name, fn in PRED_FILES.items():
        targets[f"Baseline({name})"] = fn

    print("\n===== Unified evaluation =====")
    report = evaluate_all(targets, args.coord_km, args.boot)

    out_json = os.path.join(OUT_ROOT, "eval_results", "baseline_compare_report.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] Machine-readable report: {out_json}")

    try:
        out_docx = os.path.join(OUT_ROOT, "eval_results", "baseline_comparison_bilingual.docx")
        export_docx(report, out_docx)
        print(f"[Done] Bilingual docx: {out_docx}")
    except ImportError:
        print("[Skip] python-docx not installed. Run: pip install python-docx")


if __name__ == "__main__":
    main()