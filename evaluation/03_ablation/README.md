# Ablation Study — Reproduction Guide

## Experiment Hierarchy

```
03_ablation/
├── architecture/                    # Architecture ablation: single-agent vs multi-agent
│   ├── run_three_tasks.py           # Run: three task difficulties × two architectures
│   ├── analyze_three_tasks.py       # Analyze: statistical tables + DOCX export
│   └── observability/               # Fault injection (part of architecture ablation)
│       ├── run_observability.py     # Run: controlled observation degradation
│       └── analyze_observability.py # Analyze: per-config robustness comparison
│
├── mechanism/                       # Mechanism ablation: four component variants (A/B/C/D)
│   ├── run_ablation_final.py        # Run: 6 variants × N repeats
│   └── analyze_ablation_final.py   # Analyze: Wilcoxon/Mann-Whitney + Cliff's delta + DOCX
│
└── paper/                           # Paper table generation
    └── convert_moderate_from_old.py # Data migration: old format → new format
```

## Environment Dependencies

### QGIS (required for running experiments)
- QGIS 3.40+ with embedded Python 3.12
- The `domap` package must be installed in QGIS site-packages

### Python packages (required for analysis)
```bash
pip install scipy python-docx
```

### Environment variables

| Variable | Required | Used By | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | **Yes** | `run_*` scripts | DeepSeek API key for LLM calls |
| `FAA2CHART_EVAL_ROOT` | No | architecture scripts | Path to evaluation root (default: parent of `03_ablation/`) |
| `FAA2CHART_ABLATION_OUT` | No | architecture scripts | Output directory for run results (default: `eval_results/`) |
| `FAA2CHART_JSON_DIR` | **Yes** | architecture scripts | Directory containing parsed route JSON files |
| `FAA2CHART_SHP_FILE` | **Yes** | architecture scripts | Path to state boundary shapefile |
| `FAA2CHART_MODEL_JSON` | **Yes** (complex task) | architecture scripts | Path to MinerU-parsed model.json |
| `FAA2CHART_DOMAP_PATHS` | No | architecture scripts | Semicolon-separated paths to `domap/` package directory |
| `FAA2CHART_ABLATION_RUNS` | No | mechanism scripts | Path to ablation run results directory |
| `ABLATION_SCRIPT_DIR` | No | mechanism scripts | Script directory (for QGIS exec fallback) |

## Step-by-Step Execution

### 1. Architecture ablation: single-agent vs multi-agent

#### 1a. Run experiments (QGIS Python Console)

```python
import sys
sys.path.insert(0, r"<path_to_evaluation>/03_ablation")
exec(open(r"<path_to_evaluation>/03_ablation/architecture/run_three_tasks.py", encoding="utf-8").read())

# Run all three tasks × both architectures, 10 repeats each
run_three(repeats=10)

# Run specific tasks only
run_three(repeats=10, tasks=["simple", "moderate"])

# Run specific variants only
run_three(repeats=10, variants=["multi"])

# Enable fault injection
run_three(repeats=10, fault=True)
```

**Task variants:**
| Task | Description |
|------|-------------|
| `simple` | Geocode JSON waypoints only (coordinate completion) |
| `moderate` | NY connectivity map from pre-parsed JSON (3 agents, no PDF) |
| `complex` | Full PDF→JSON→geocode→CA map (all 5 agents) |

**Architecture variants:**
| Variant | Description |
|---------|-------------|
| `multi` | Full multi-agent with supervisor + 5 sub-agents |
| `single` | Single-agent with merged tools |

#### 1b. Analyze results (standalone Python, no QGIS needed)

```bash
python architecture/analyze_three_tasks.py
```

Outputs:
- `eval_results/ablation_three_tasks_statistical_report.txt` — Markdown tables
- `eval_results/ablation_three_tasks_statistical_report.docx` — Word document
- Console: per-task comparison with Mann-Whitney U p-values

#### 1c. Observability / fault injection

```bash
# Run fault injection experiment
python architecture/observability/run_observability.py

# Run specific config
python architecture/observability/run_observability.py --config faulted

# Analyze results
python architecture/observability/analyze_observability.py
```

Configurations:
| Config | Description |
|--------|-------------|
| `control` | Full observability (5 sub-agents) |
| `single` | Single agent with full observability |
| `single_serial` | Single agent, serialized tool calls |
| `faulted` | Limited/degraded observability |

### 2. Mechanism ablation: four component variants

#### 2a. Run experiments (QGIS Python Console)

```python
import sys
sys.path.insert(0, r"<path_to_evaluation>/03_ablation")
exec(open(r"<path_to_evaluation>/03_ablation/mechanism/run_ablation_final.py", encoding="utf-8").read())

# Run all 5 main variants, 10 repeats each
run_all(repeats=10)

# Run specific variants
run_all(repeats=10, variants=["control", "A_merge"])

# B_no_store is a necessity demo (expected failure), run separately
run_all(repeats=1, variants=["B_no_store"])
```

**6 variants:**
| Variant | Description | Expected result |
|---------|-------------|-----------------|
| `control` | Full 5 sub-agents (baseline) | Best performance |
| `A_merge` | Merge layer_loader + analysis → 4 sub-agents | Degradation |
| `B_no_store` | Remove ObjectStore, serialize state in conversation | Structural failure |
| `C_natural_lang` | Sub-agent returns natural language prose instead of structured dicts | Degradation |
| `D_no_guard` | Remove inspection/validation tools and prompt pre-guards | Degradation |
| `single` | Single-agent baseline | Degradation |

#### 2b. Analyze results (standalone Python, no QGIS needed)

```bash
python mechanism/analyze_ablation_final.py
```

Outputs:
- `ablation_results/ablation_stats_report.json` — Machine-readable statistics
- `ablation_results/ablation_results_bilingual.docx` — Word document with tables
- Console: per-variant summary with p-values and Cliff's delta

### 3. Data migration (for paper tables)

```bash
# Convert old format (run_ablation_final) to new three-task format
python paper/convert_moderate_from_old.py
```

## Output Interpretation

### Key metrics
| Metric | Description | Low-better |
|--------|-------------|------------|
| `elapsed_sec` | Wall-clock time per run | Yes |
| `total_steps` | Total supervisor stream steps | No |
| `tool_calls` | Total tool invocation count | Yes |
| `subagent_delegations` | Number of sub-agent delegation events | No |
| `tool_errors` | Tool execution errors | Yes |
| `observable_points` | Total AI messages + tool calls | No |
| `success_rate` | Fraction of runs producing an analysis layer or layout | Higher is better |

### Statistical tests
- **Architecture ablation** (architecture/): Mann-Whitney U two-sided test
- **Mechanism ablation** (mechanism/): Wilcoxon signed-rank (equal n) or Mann-Whitney U (unequal n)
- **Effect size**: Cliff's delta (|δ|<0.147 negligible, <0.33 small, <0.474 medium, otherwise large)

## Notes

- All experiments require the `domap` package installed in QGIS site-packages
- The `mechanism/` ablation uses a fixed task (CA connectivity map from pre-parsed JSON)
- The `architecture/` ablation tests three distinct task difficulties
- B_no_store is a necessity demonstration (n=1, expected structural failure), excluded from significance testing
- B_no_store monkey-patches global module state; always run it separately or last in the variant list