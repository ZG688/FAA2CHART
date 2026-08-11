# Cartographic Evaluation — Execution Guide

## evaluate_cartographic.py

**Purpose:** Three-level cartographic correctness evaluation.
- Topology: edge P/R/F1, node Jaccard
- Geometry: endpoint distance, buffer IoU, bearing error, length rel-err
- Attribute: MEA/MAA match rate

Reference: Feb 2025 NASR (topology/attributes) + Nov 2025 NASR (geometry coordinates).

### Usage

```bash
# Default: buffer 5km, bootstrap 1000
python evaluate_cartographic.py

# Custom buffer radius and bootstrap iterations
python evaluate_cartographic.py --buffer-km 10 --boot 2000
```

### Parameters

| Parameter    | Default | Description                              |
|--------------|---------|------------------------------------------|
| `--buffer-km` | 5.0   | Buffer IoU radius (km)                   |
| `--boot`      | 1000  | Bootstrap iterations for 95% CI          |

### Output

- Console: per-level metric table with 95% CI
- JSON: `cartographic_eval_report.json` (bootstrap CIs, per-edge detail)

---

## evaluate_connectivity_consistency.py

**Purpose:** Validates the interstate connectivity thematic map's reproducibility and state-attribution consistency.

Three parts:
- A. Score reproducibility: independently recomputes connectivity scores from source JSON
- B. Region-label geometric validation: point-in-polygon vs. extracted region field
- C. Propagation: ranking consistency (Spearman) and Top-K overlap after geometric correction

### Usage

```bash
# No arguments — runs fully automatically
python evaluate_connectivity_consistency.py
```

### Configuration (edit constants in file)

| Constant       | Default | Description                              |
|----------------|---------|------------------------------------------|
| `TARGET_STATE` | `"CA"`  | Target state for connectivity analysis   |
| `LABEL_TOP_K`  | `5`     | Top-K labels to display in results       |
| `N_BOOT`       | `1000`  | Bootstrap iterations for 95% CI          |

### Output

- JSON: `data/connectivity_consistency_report.json`

---

## overlay_overview_vs_nasr.py

**Purpose:** Overlay visualization of generated airway geometry vs. official NASR reference polylines. Stratified sampling by route length (short/medium/long), side-by-side comparison.

### Usage

```bash
# Default: 15 samples (5 short + 5 medium + 5 long)
python overlay_overview_vs_nasr.py

# Custom sample count
python overlay_overview_vs_nasr.py --n 30

# Specific airway codes (comma-separated, overrides sampling)
python overlay_overview_vs_nasr.py --codes "V11,V327,V16"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n`     | `15`    | Number of samples (stratified by length) |
| `--codes` | `""`    | Comma-separated airway codes (overrides stratified sampling) |

### Output

- `data/overlay_overview_samples.png` — grid overlay figure
- `data/overlay_overview_manifest.json` — sampling manifest + per-route metrics

---

## Input Data Dependencies

All three scripts depend on environment variable `FAA2CHART_DATA_ROOT` pointing to the experiment data directory, with subdirectories:

```
{DATA_ROOT}/
├── airway_data/
│   ├── 28DaySubscription_Effective_2025-02-20/   (Feb NASR — content/sequence gold)
│   ├── 28DaySubscription_Effective_2025-11-27/   (Nov NASR — coordinate geometry gold)
│   └── text/                                      (prediction JSON files)
└── us_vector_data/
    └── ne_10m_admin_1_states_provinces/           (Natural Earth state boundaries)
```