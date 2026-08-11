# 04 Baseline Comparison

## Overview

This module implements systematic comparison between FAA2CHART and 5 strong baselines, covering **paradigm baselines** (different technical approaches) and **framework baselines** (general-purpose multi-agent frameworks).

### Baseline List

| Category | Baseline | File | Description |
|----------|----------|------|-------------|
| **Paradigm** | Rule-based | `baseline_rule_based.py` | Deterministic parser (regex + layout rules, no LLM) |
| | RAG | `baseline_rag.py` | Retrieval-augmented extraction (TF-IDF + LLM single extraction) |
| | Codegen | `baseline_codegen.py` | Direct code generation (LLM writes parser, then execute) |
| **Framework** | ReAct | `baseline_react.py` | ReAct single agent (Thought-Action-Observation loop, faithful to Yao et al. 2023) |
| | AutoGen | `baseline_autogen.py` | AutoGen conversational multi-agent (GroupChat: Extractor + Validator + Aggregator) |

### Fairness Guarantees

- **Unified input**: All baselines consume the same `table_with_title-new1.json` (cleaned airway PDF text)
- **Unified output**: `list[{airway_code, MEA, MAA, airway_point:[{name,region,type,position}]}]`
- **Unified evaluation**: Scored by `evaluate_extraction_v2.py` with identical metrics
- **Unified LLM**: RAG/Codegen/ReAct/AutoGen all use Claude Opus 4.5, temperature=0
- **Unified prompt**: Per-chunk extraction prompt is identical across all LLM-based baselines; only the agent framework differs

---

## File Descriptions

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker |
| `baselines_common.py` | **Shared utilities**: paths, input loading, airway code extraction, point parsing, dedup, normalization, continued-page merging, prediction saving |
| `baseline_rule_based.py` | **Baseline 1: Rule-based** — HTML table / plain text regex parsing, no LLM |
| `baseline_rag.py` | **Baseline 2: RAG** — TF-IDF retriever + Claude single extraction |
| `baseline_codegen.py` | **Baseline 3: Codegen** — LLM generates Python parser → sandboxed exec → batch parse |
| `baseline_react.py` | **Baseline 4: ReAct** — Hand-written Thought-Action-Observation loop |
| `baseline_autogen.py` | **Baseline 5: AutoGen** — GroupChat multi-agent (Extractor+Validator+Aggregator), with both autogen-package and hand-written fallback paths |
| `generated_parser.py` | LLM-generated parser from codegen baseline (archived for human review) |
| `run_baselines_and_compare.py` | **Unified driver**: run baselines → evaluate → export bilingual DOCX comparison table |
| `run_baselines_qgis.py` | QGIS Python Console one-click script (with performance fixes) |
| `generate_gpkg_from_json.py` | Convert prediction JSONs to GPKG (for cartographic evaluation) |
| `evaluate_cartographic_baselines.py` | Cartographic correctness evaluation (topology/geometry/attribute × 3 layers + bootstrap CI) |
| `visual_compare_baselines.py` | Qualitative visual comparison (7 methods overlaid on 3 typical airways) |
| `analyze_gis_codegen.py` | GIS codegen baseline statistical analysis (success rate, attempts, completeness, efficiency) |
| `tierB_baselines_150_evaluate.py` | Tier B 150-route human-annotated gold evaluation (with IAA computation + OCR correction) |

---

## Setup

### Dependencies

```bash
# Core
pip install requests numpy matplotlib pandas openpyxl python-docx

# AutoGen baseline (optional; falls back to hand-written implementation if not installed)
pip install pyautogen==0.2.40
```

### Environment Variables

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your_key_here"
# CMD
set ANTHROPIC_API_KEY=your_key_here
```

### Data Path Configuration

Set the `DATA_ROOT` environment variable, or modify the default path in `baselines_common.py`:

```python
DATA_ROOT = os.environ.get("FAA2CHART_DATA_ROOT", "PATH_TO_YOUR_DATA_ROOT")
```

Required data files:
- Cleaned input: `{DATA_ROOT}/航报/文本/table_with_title-new1.json`
- Prediction output dir: `{DATA_ROOT}/航报/文本/` (same as main experiment for evaluator access)
- FAA NASR data: `{DATA_ROOT}/航报/28DaySubscription_Effective_2025-02-20/CSV_Data/`
- NASR coordinate library: `{DATA_ROOT}/航报/28DaySubscription_Effective_2025-11-27/CSV_Data/`

---

## How to Run

### 1. Run Individual Baselines (Smoke Test)

```bash
# Rule-based (no API key needed, runs immediately)
python baseline_rule_based.py

# RAG (requires API key, try --top-k 10 first)
python baseline_rag.py --top-k 40

# Codegen (requires API key)
python baseline_codegen.py

# ReAct (requires API key, try --limit 50 first)
python baseline_react.py --limit 50

# AutoGen (requires API key, try --limit 50 first)
python baseline_autogen.py --limit 50
```

### 2. Unified Run + Evaluation (Recommended)

```bash
# Run all baselines + evaluate + export DOCX comparison table
python run_baselines_and_compare.py --run rule_based rag codegen react autogen

# Evaluate existing predictions only (skip re-running)
python run_baselines_and_compare.py --eval-only

# Framework baselines with --limit for quick smoke test
python run_baselines_and_compare.py --run react autogen --limit 50
```

### 3. Cartographic Evaluation (requires GPKG)

```bash
# Step 1: Generate GPKG from JSON predictions
python generate_gpkg_from_json.py

# Step 2: Evaluate cartographic correctness
python evaluate_cartographic_baselines.py
```

### 4. Visual Comparison

```bash
python visual_compare_baselines.py
```

### 5. Tier B 150-Route Human-Annotated Gold Evaluation

```bash
python tierB_baselines_150_evaluate.py
```

### 6. GIS Codegen Statistical Analysis

```bash
python analyze_gis_codegen.py
```

---

## Output Artifacts

| File | Description |
|------|-------------|
| `{DATA_ROOT}/航报/文本/parsed_routes_baseline_*.json` | Per-baseline extraction results (5 files) |
| `../评估结果/baseline_compare_report.json` | Machine-readable evaluation report |
| `../评估结果/baseline_comparison_bilingual.docx` | Bilingual comparison table (DOCX) |
| `../评估结果/baselines_gpkg/airway_segments_*.gpkg` | Per-method cartographic GPKG |
| `../评估结果/cartographic_baselines_report.json` | Cartographic evaluation report |
| `../评估结果/visual_compare/compare_*.png` | Visual comparison figures |
| `../评估结果/gis_codegen_stats_report.json` | GIS codegen statistics report |
| `data/tierB_150_baselines_report.json` | Tier B 150 evaluation report |