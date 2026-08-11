# FAA2CHART

A collaborative multi-agent framework that combines large language models (LLMs) and vision-language models (VLMs) for automated aeronautical chart generation from regulatory PDF documents.

## Overview

FAA2CHART transforms lengthy FAA regulatory documents (e.g., Part 95: IFR Minimum Enroute Altitudes) into complete, spatially grounded geospatial visualizations. The framework integrates:

1. **Layout-aware document parsing** (MinerU + LLM extraction)
2. **Schema-constrained airway event extraction** (domain profiles)
3. **Waypoint geolocation** (FAA NASR gazetteer lookup)
4. **GIS function calling** (QGIS-based layer loading, spatial analysis, map decoration)
5. **Automated map generation** (connectivity maps with migration-style arc visualization)

## Architecture

The multi-agent system consists of a supervisor agent orchestrating 5 specialized sub-agents:

| Sub-agent | Responsibility |
|-----------|---------------|
| `event_extract` | PDF → structured JSON extraction |
| `geocode` | Waypoint coordinate completion via gazetteer |
| `layer_loader` | QGIS project setup, state boundary loading |
| `analysis` | Connectivity metrics computation, thematic layer creation |
| `map_publish` | Layout creation, title/legend/scale bar, export |

## Package Structure

```
FAA2CHART/
├── __init__.py
├── README.md
├── domap/
│   ├── __init__.py
│   ├── domain_profile.py          # Domain-specific extraction configs (FAA airway, USCG LNM)
│   ├── event_extract.py           # MinerU document parsing + LLM structured extraction
│   ├── name_postion.py            # Waypoint geolocation via FAA NASR CSV / USCG Light List
│   ├── prompts.json               # All agent prompts extracted for reference
│   └── connectivity/
│       ├── __init__.py
│       ├── multi_agents.py        # Multi-agent supervisor + sub-agent definitions
│       ├── single_agent.py        # Single-agent baseline (all tools in one agent)
│       ├── agent_job_tools.py     # Job-aware tool wrappers for sub-agents
│       ├── shared_map_state.py    # MapConfig, layer/style/layout configuration dataclasses
│       ├── virtual_paths.py       # Virtual path ↔ host path mapping
│       ├── runtime_job_store.py   # Thread-safe job_id → state dictionary
│       ├── layer_loader_tools.py  # QGIS layer loading functions
│       ├── spatial_analysis_tools.py   # Connectivity analysis + thematic layer creation
│       ├── map_decoration_tools.py     # Layout creation, decoration, export
│       └── point_event_analysis_tools.py  # Point event analysis (maritime generalization)
├── experiments/
│   ├── __init__.py
│   ├── experiment_1_1_overview.py
│   ├── experiment_1_2_connectivity.py
│   ├── experiment_1_3_generalization.py
│   └── faa2chart_interface.py
└── evaluation/
    ├── 01_extraction/              # Extraction accuracy evaluation
    ├── 02_cartographic/            # Cartographic correctness evaluation
    ├── 03_ablation/                # Ablation studies (architecture & mechanism)
    └── 04_baseline_comparison/     # Strong baseline comparisons (Rule-based, RAG, Codegen, ReAct, AutoGen)
```

## Requirements

- Python 3.10+
- QGIS 3.x (with PyQGIS)
- LangChain / DeepAgents
- DeepSeek API access (or compatible LLM endpoint)
- MinerU API access (for PDF parsing)

## Setup

### 1. Install QGIS

Ensure QGIS is installed and PyQGIS is available in your Python environment.

### 2. Install Python Dependencies

```bash
pip install langchain langchain-deepseek deepagents pandas requests
```

### 3. Set Environment Variables

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export MINERU_TOKEN="your-mineru-token"
export FAA2CHART_HOST_ROOT="/path/to/your/data/root"
```

### 4. Configure Data Paths

- `domap/name_postion.py`: Set `fix_df` and `nav_df` CSV paths (FAA NASR data)
- `domap/connectivity/virtual_paths.py`: Set `HOST_ROOT` to your data root directory
- `domap/event_extract.py`: Set `SAVE_DIR` for MinerU output

## MinerU PDF Parsing Notes

The framework uses the MinerU API for PDF parsing. Be aware of the following limitations:

### 1. MinerU API Does Not Accept Local File Paths

The MinerU API requires a publicly accessible URL for the PDF file. Local file paths (e.g., `/path/to/local/file.pdf`) are **not supported**. You must provide a URL that the MinerU service can reach over the internet.

### 2. Getting a Public URL for Your PDF

If the official FAA PDF is not accessible or you need to use a local copy, you can:

1. Download the PDF to your local machine.
2. Upload it to a temporary file hosting service such as [https://tmpfile.link/](https://tmpfile.link/) to obtain a publicly accessible URL.
3. Pass this public URL to the MinerU API.

### 3. MinerU API Page Limit (200 pages)

The MinerU API **no longer accepts PDFs exceeding 200 pages**. The FAA Part 95 consolidation PDF is typically 400+ pages. To work around this limitation, you must:

1. **Split the PDF** into chunks of ≤200 pages each (e.g., using `PyPDF2` or `pdfplumber`).
2. **Process each chunk separately** through the MinerU API, saving the output JSON for each chunk.
3. **Merge the results** by combining the extracted JSON outputs from all chunks into a single document.

Example splitting script:

```python
import PyPDF2

def split_pdf(input_path, pages_per_chunk=200):
    with open(input_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        total = len(reader.pages)
        for start in range(0, total, pages_per_chunk):
            writer = PyPDF2.PdfWriter()
            end = min(start + pages_per_chunk, total)
            for i in range(start, end):
                writer.add_page(reader.pages[i])
            chunk_path = f"{input_path.replace('.pdf', '')}_p{start+1}-{end}.pdf"
            with open(chunk_path, "wb") as out:
                writer.write(out)
            print(f"Created {chunk_path} ({end-start} pages)")
```

## Usage

Run in QGIS Python Console:

```python
from domap.connectivity.multi_agents import (
    run_unified_airway_factory_multi_agent,
    build_model,
)
from domap.connectivity.shared_map_state import MapConfig, PathConfig

# Configure paths
cfg = MapConfig(paths=PathConfig(
    json_path="/path/to/airway_data.json",
    state_boundary_path="/path/to/states.shp",
    png_path="/path/to/output.png",
    pdf_path="/path/to/output.pdf",
    layout_name="FAA_CA_Connectivity_Map",
))
cfg.data.target_state = "CA"
cfg.data.label_top_k = 5
cfg.run.auto_export = True

# Run multi-agent pipeline
result = run_unified_airway_factory_multi_agent(
    user_request="Draw a target-centered airway connectivity map for California",
    cfg=cfg,
    stream=True,
)
```

For single-agent baseline:

```python
from domap.connectivity.single_agent import run_unified_airway_factory_single_agent

result = run_unified_airway_factory_single_agent(
    user_request="Draw a target-centered airway connectivity map for California",
    cfg=cfg,
    stream=True,
)
```

## Experiments

Three benchmark experiment interfaces are provided under `experiments/`:

| File | Description | Variants |
|------|-------------|----------|
| `experiment_1_1_overview.py` | Human-readable overview map | simple / medium / complex |
| `experiment_1_2_connectivity.py` | Agent-readable connectivity map | simple / medium / complex / single-agent |
| `experiment_1_3_generalization.py` | Domain generalization (maritime) | simple / medium / complex |
| `faa2chart_interface.py` | Unified natural-language interface | all benchmarks + custom task |

### Running Benchmarks

```bash
# Run a specific benchmark
python experiments/faa2chart_interface.py --benchmark 2_simple_connectivity

# Run all benchmarks
python experiments/faa2chart_interface.py --benchmark all

# Run with a custom natural-language request
python experiments/faa2chart_interface.py --task connectivity --request "Draw a CA connectivity map..."
```

### Benchmark IDs

- **Experiment 1.1 (Overview):** `1_simple_overview`, `1_medium_overview`, `1_complex_overview`
- **Experiment 1.2 (Connectivity):** `2_simple_connectivity`, `2_medium_connectivity`, `2_complex_connectivity`, `2_single_agent`
- **Experiment 1.3 (Maritime):** `3_simple_maritime`, `3_medium_maritime`

## Evaluation

The `evaluation/` directory contains all experimental results reported in the paper:

| Directory | Experiment | Description |
|-----------|------------|-------------|
| `01_extraction/` | Extraction accuracy | Tier A / Tier B benchmark results |
| `02_cartographic/` | Cartographic correctness | Topology, geometry, attribute metrics |
| `03_ablation/` | Ablation studies | Architecture (6 variants) & mechanism (sub-agent decomposition) |
| `04_baseline_comparison/` | Baseline comparison | 5 strong baselines (Rule-based, RAG, Codegen, ReAct, AutoGen) |

Each evaluation directory contains its own README with detailed reproduction instructions. See `evaluation/04_baseline_comparison/README.md` for the baseline comparison reproduction guide.

## Domain Profiles

The framework supports domain-specific extraction via `DomainProfile`:

- **`faa_airway`** (default): FAA Part 95 airway route extraction
- **`uscg_lnm`**: USCG Local Notice to Mariners — aid-to-navigation change events

To add a new domain, create a `DomainProfile` instance with a custom `prompt_template` and `is_relevant` function.

## Citation

If you use FAA2CHART in your research, please cite:

> [Paper citation to be added upon publication]

## License

[License to be determined]