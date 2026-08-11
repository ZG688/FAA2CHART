# -*- coding: utf-8 -*-
"""
QGIS Python Console one-click script: run RAG + Codegen baselines + unified evaluation + export comparison table
==================================================================
Usage: Paste the entire block into QGIS Python Console (Plugins -> Python Console) and run.

Notes:
  1. Replace ANTHROPIC_API_KEY below with your actual key
  2. If python-docx is not installed, DOCX export will be skipped (does not affect JSON evaluation)
"""
import os
import sys
import json
import importlib.util

# ============================================================
# Step 1: Set API Key — replace YOUR_KEY_HERE with your actual key
# ============================================================
os.environ["ANTHROPIC_API_KEY"] = "YOUR_KEY_HERE"

# ============================================================
# Step 2: Fixed paths (consistent with terminal version, does not depend on cwd in QGIS)
# ============================================================
SCRIPTS_DIR = os.environ.get("BASELINE_SCRIPTS_DIR", "PATH_TO_YOUR_BASELINE_CODE_DIR")
OUT_ROOT   = os.environ.get("BASELINE_OUT_ROOT", "PATH_TO_YOUR_BASELINE_ROOT")
DATA_ROOT  = os.environ.get("FAA2CHART_DATA_ROOT", "PATH_TO_YOUR_DATA_ROOT")

# Evaluator path
EVAL_SCRIPT = os.path.join(
    OUT_ROOT, "..", "01_extraction",
    "evaluate_extraction_v2.py")

# Prediction output directory and filenames (consistent with baselines_common.py)
PRED_OUT_DIR = os.path.join(DATA_ROOT, "airway_data", "text")
PRED_FILES = {
    "rule_based": "parsed_routes_baseline_rule_based.json",
    "rag":        "parsed_routes_baseline_rag.json",
    "codegen":    "parsed_routes_baseline_codegen.json",
}

# Multi-agent reference
MULTIAGENT_REF = {
    "MultiAgent(Claude4.5)": "parsed_routes_newnew_5_Claude4.5_updated_merged.json",
}

# Add script directory to sys.path for importing submodules
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

print("=" * 60)
print("Baseline Comparison Pipeline (QGIS)")
print(f"Scripts dir: {SCRIPTS_DIR}")
print(f"Evaluator:   {EVAL_SCRIPT}")
print(f"Pred output: {PRED_OUT_DIR}")
print("=" * 60)

# ============================================================
# Step 3: Run RAG and Codegen baselines
# ============================================================

# --- RAG ---
print("\n===== Running baseline RAG =====")
import baseline_rag
baseline_rag.run_rag(top_k=40)

# --- Codegen ---
print("\n===== Running baseline Codegen =====")
import baseline_codegen
baseline_codegen.run_codegen()

# ============================================================
# Step 4: Performance fix (accelerate evaluator fuzzy matching)
# ============================================================
def _accelerate_to_id(mod):
    """Cache to_id + length-filter fuzzy_match to avoid O(N²) Levenshtein"""
    _orig_to_id = mod.to_id
    _cache = {}

    def cached_to_id(name, cg):
        key = (str(name), id(cg))
        if key in _cache:
            return _cache[key]
        val = _orig_to_id(name, cg)
        _cache[key] = val
        return val
    mod.to_id = cached_to_id

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


# ============================================================
# Step 5: Load evaluator
# ============================================================
print("\n===== Loading evaluator =====")
spec = importlib.util.spec_from_file_location("eval_v2", os.path.abspath(EVAL_SCRIPT))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
_accelerate_to_id(ev)

# Load gold data
feb = ev.load_airways(ev.FEB_CSV_DIR)
nov_coord = ev.load_coords(ev.NOV_CSV_DIR)
print(f"Feb airway gold: {len(feb['awy_seq'])}  Nov coords: {len(nov_coord['coord'])}")

# ============================================================
# Step 6: Unified evaluation
# ============================================================
print("\n===== Unified evaluation =====")
COORD_KM = 2.0
BOOT = 1000

# Build evaluation target list
targets = dict(MULTIAGENT_REF)
for name, fn in PRED_FILES.items():
    targets[f"Baseline({name})"] = fn

report = {}
for name, fn in targets.items():
    path = os.path.join(PRED_OUT_DIR, fn)
    if not os.path.exists(path):
        print(f"  [Skip] Prediction file not found: {path}")
        continue
    res = ev.evaluate(path, feb, nov_coord, COORD_KM)
    s = ev.summarize(res, n_boot=BOOT)
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

# ============================================================
# Step 7: Save JSON results
# ============================================================
out_json = os.path.join(OUT_ROOT, "eval_results", "baseline_compare_report.json")
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n[Done] Machine-readable report: {out_json}")

# ============================================================
# Step 8: Export bilingual DOCX
# ============================================================
try:
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.table import WD_TABLE_ALIGNMENT

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(9)
    p = doc.add_paragraph()
    r = p.add_run(
        "[Insertion point] Insert into Section 6.5 (strong baseline comparison)."
        "Label only, original text unchanged.")
    r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    r.bold = True

    doc.add_heading("Baseline Comparison for Extraction", level=1)
    doc.add_paragraph(
        "Table. Extraction performance of all methods under the same FAA NASR (Feb 2025) "
        "ground truth and identical metrics.")

    cols = [("Method", None), ("#Codes", "n_pred_codes"),
            ("code_F1", "code_F1"), ("wp_P", "waypoint_P"),
            ("OSV", "OSV"), ("order_tau", "order_tau"),
            ("MEA_acc", "MEA_acc"), ("MAA_acc", "MAA_acc"),
            ("trunc%", "trunc_pct")]
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = "Table Grid"
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

    note = doc.add_paragraph()
    rr = note.add_run(
        "Note: All methods consume the same cleaned input and emit the same schema; "
        "content-level metrics use Feb 2025 NASR. "
        "code_F1/waypoint_P/OSV/order_tau/MEA/MAA are reliable extraction capability metrics "
        "(waypoint recall is excluded because FAA selective charting vs. the full database is "
        "not a comparable reference). rule_based uses no LLM; RAG only processes retrieved chunks; "
        "codegen executes an LLM-generated parser.")
    rr.font.size = Pt(8)

    out_docx = os.path.join(OUT_ROOT, "eval_results", "baseline_comparison_bilingual.docx")
    doc.save(out_docx)
    print(f"\n[Done] Bilingual docx: {out_docx}")
except ImportError:
    print("[Skip] python-docx not installed. Run in QGIS Python Console:")
    print("  import subprocess; subprocess.check_call(['pip', 'install', 'python-docx'])")
    print("  Then re-run this script.")

print("\n===== All done =====")