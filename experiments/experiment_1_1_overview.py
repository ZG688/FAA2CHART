# -*- coding: utf-8 -*-
"""
Experiment 1.1 — FAA IFR Airway Overview Map (Human-Readable Format)

This experiment reproduces the traditional cartographic output: a multi-panel
overview map of the U.S. FAA IFR airway network, with waypoints and segments
colored by airway type (colored federal, VOR federal, low RNAV, high RNAV, jet).

Benchmarks:
  - CA:  Single-state overview with basic Natural Earth basemap
  - NY:  Multi-state overview with classified airway segments
  - US:  Full continental U.S. 4-panel layout with full legend

The map is styled for human readability — the "human-readable" baseline
against which the agent-readable connectivity map (Experiment 1.2) is compared.

Prerequisites:
  - QGIS Python environment with PyQGIS
  - FAA2CHART package installed (see README.md)
  - Extracted airway JSON (via event_extract + geocode pipeline)
  - Natural Earth state boundary shapefile
"""

import os
import sys
from pathlib import Path

# Ensure QGIS Python paths are available
# Add your QGIS python path here if needed:
# sys.path.append(r"C:\Program Files\QGIS 3.40\apps\Python312\Lib\site-packages")

from domap.connectivity.shared_map_state import (
    MapConfig, PathConfig, LayoutConfig, RunConfig, LayerNameConfig, StyleConfig
)
from domap.connectivity.multi_agents import (
    run_unified_airway_factory_multi_agent, build_model
)
from domap.domain_profile import FAA_AIRWAY_PROFILE


# ============================================================================
# Benchmark Examples — Pre-configured for direct execution
# ============================================================================

# ---------------------------------------------------------------------------
# Benchmark 1: CA — California overview with airway classification
#   Target: CA
#   Scope:  Single-state, Natural Earth basemap
#   Layout: 1-panel classification map
#   Expected output: PNG + PDF in output dir
# ---------------------------------------------------------------------------
def benchmark_ca_overview():
    """Single-state airway classification overview (California)."""
    cfg = MapConfig(
        paths=PathConfig(
            json_path="PATH_TO_YOUR_AIRWAY_JSON",
            state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
            png_path="./output/overview_ca.png",
            pdf_path="./output/overview_ca.pdf",
            layout_name="FAA_CA_Overview",
        ),
        run=RunConfig(
            open_designer=False,
            auto_export=True,
            export_dpi=300,
        ),
    )
    cfg.data.target_state = "CA"
    cfg.data.label_top_k = 10

    return run_unified_airway_factory_multi_agent(
        user_request=(
            "Extract airway routes from the PDF, geocode waypoints, "
            "then create a California-centered overview map with airway "
            "classification labels. Highlight target state, show all "
            "airway segments colored by route type."
        ),
        source_document="https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf",
        cfg=cfg,
        profile=FAA_AIRWAY_PROFILE,
        stream=True,
    )


# ---------------------------------------------------------------------------
# Benchmark 2: NY — Multi-state overview with detailed classification
#   Target: NY
#   Scope:  Multi-state, segmented/colored by airway type
#   Layout: 2-panel layout (overview + classification detail)
#   Expected output: PNG + PDF
# ---------------------------------------------------------------------------
def benchmark_ny_classification():
    """Multi-state airway classification with detailed segment colors (New York)."""
    cfg = MapConfig(
        paths=PathConfig(
            json_path="PATH_TO_YOUR_AIRWAY_JSON",
            state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
            png_path="./output/overview_ny.png",
            pdf_path="./output/overview_ny.pdf",
            layout_name="FAA_NY_Overview",
        ),
        layout=LayoutConfig(
            title_text="FAA Part 95 IFR Airway Network — New York Region",
            subtitle_template="Segments: {segment_count} | Unique nodes: {node_count}",
            footer_note="Colored by airway type: Blue=Colored Federal, Green=VOR, Orange=Low RNAV, Purple=High RNAV, Red=Jet",
        ),
        run=RunConfig(
            open_designer=False,
            auto_export=True,
            export_dpi=300,
        ),
    )
    cfg.data.target_state = "NY"
    cfg.data.label_top_k = 15

    return run_unified_airway_factory_multi_agent(
        user_request=(
            "Extract airway routes from the PDF, geocode waypoints, "
            "then create a New-York-centered overview map with all "
            "airway segments colored by route type. Include a detailed "
            "legend explaining the color scheme."
        ),
        source_document="https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf",
        cfg=cfg,
        profile=FAA_AIRWAY_PROFILE,
        stream=True,
    )


# ---------------------------------------------------------------------------
# Benchmark 3: US — Full continental U.S. 4-panel layout
#   Target: None (full U.S.)
#   Scope:  Continental U.S., 4-panel layout
#   Layout: 4-panel (main, NE, SE, NW sub-panels)
#   Expected output: PNG + PDF (large format)
# ---------------------------------------------------------------------------
def benchmark_us_panels():
    """Full-U.S. 4-panel overview layout (continental scale)."""
    cfg = MapConfig(
        paths=PathConfig(
            json_path="PATH_TO_YOUR_AIRWAY_JSON",
            state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
            png_path="./output/overview_us.png",
            pdf_path="./output/overview_us.pdf",
            layout_name="FAA_US_Overview",
        ),
        layout=LayoutConfig(
            title_text="FAA Part 95 IFR Federal Airway Network — Continental U.S.",
            subtitle_template="Extracted by LLM+VLM | {airway_count} airways | {node_count} unique waypoints",
            footer_note="Data source: FAA Part 95 Consolidation (Feb 2025) | Extraction: DeepSeek-chat + MinerU",
            map_panel_title="Airway Overview Map",
        ),
        run=RunConfig(
            open_designer=False,
            auto_export=True,
            export_dpi=300,
        ),
    )
    cfg.data.target_state = None  # Full U.S.
    cfg.data.label_top_k = 30

    return run_unified_airway_factory_multi_agent(
        user_request=(
            "Extract all airway routes from the PDF, geocode waypoints, "
            "then create a full continental U.S. 4-panel overview map. "
            "Color-code all airway segments by type. Show the top 30 "
            "most connected waypoints with labels."
        ),
        source_document="https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf",
        cfg=cfg,
        profile=FAA_AIRWAY_PROFILE,
        stream=True,
    )


# ============================================================================
# Quick-run entry point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FAA2CHART Experiment 1.1 — Overview Map (Human-Readable)"
    )
    parser.add_argument(
        "--variant", choices=["ca", "ny", "us"], default="ca",
        help="Benchmark variant to run"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Path to pre-extracted airway JSON (skip PDF extraction)"
    )
    parser.add_argument(
        "--states", type=str, default="PATH_TO_NE_10M_STATES_SHP",
        help="Path to Natural Earth state boundary shapefile"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Output directory for PNG/PDF"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    benchmarks = {
        "ca": benchmark_ca_overview,
        "ny": benchmark_ny_classification,
        "us": benchmark_us_panels,
    }

    print(f"Running Experiment 1.1 — Overview Map ({args.variant})")
    result = benchmarks[args.variant]()
    print("Experiment 1.1 complete.")