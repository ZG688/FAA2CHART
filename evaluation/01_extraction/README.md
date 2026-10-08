# Extraction Evaluation — Execution Guide

## evaluate_extraction_v2.py (Tier A)

**Purpose:** Dual-gold-source evaluation of airway extraction against FAA NASR.
- Content/structure → Feb 2025 NASR (same PDF cycle)
- Coordinate localization → Nov 2025 NASR (same experiment nav DB)

### Usage

```bash
# Evaluate all 4 models (Claude4.5, deepseek_chat, deepseek_reasoner, gpt5.2)
python evaluate_extraction_v2.py

# Evaluate a single prediction file
python evaluate_extraction_v2.py --pred path/to/prediction.json

# Custom coordinate threshold and bootstrap iterations
python evaluate_extraction_v2.py --coord-km 3.0 --boot 2000

# Custom output path
python evaluate_extraction_v2.py --out ./my_report.json
```

### Parameters

| Parameter    | Default         | Description                        |
|--------------|-----------------|------------------------------------|
| `--pred`     | (all 5 models)  | Single prediction JSON to evaluate |
| `--coord-km` | 2.0             | Coordinate hit threshold (km)      |
| `--boot`     | 1000            | Bootstrap iterations for 95% CI    |
| `--out`      | extraction_eval_report.json | Output JSON report path |

### Output

- Console: per-model metric table + cross-comparison
- JSON: `extraction_eval_report.json` with per-model summaries + 95% CIs

---

## tierB_evaluate_all.py (Tier B)

**Purpose:** Human-annotated gold standard evaluation on 150 sampled routes.
- Gold: dual-annotator adjudicated PDF point names (`data/tierB_labels.xlsx`)
- Metrics: micro/macro P/R/F1, Record-EM, Seq-Exact, order tau, cross-page EM

### Usage

```bash
# Run full evaluation (no arguments needed)
python tierB_evaluate_all.py
```

### Input

| File | Description |
|------|-------------|
| `data/Tier B human-gold benchmark/tierB_labels.xlsx` | 150-route manual annotation table (columns 8-10: annotator1, annotator2, cross-page flag) |
| `data/Tier B human-gold benchmark/pdf_screenshot/` | 150 source PDF screenshots (aligned with label table rows 2–151) |
| Prediction JSON files | From `FAA2CHART_DATA_ROOT/airway_data/text/` (or `FAA2CHART_PRED_DIR` env var) |

### Output

- Console: IAA report + per-model comparison table + 95% CI
- JSON: `data/tierB_eval_all_results.json` with per-route details

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FAA2CHART_DATA_ROOT` | Root directory for experiment data (default: relative to script) |
| `FAA2CHART_PRED_DIR` | Prediction JSON directory (default: `{DATA_ROOT}/airway_data/text`) |