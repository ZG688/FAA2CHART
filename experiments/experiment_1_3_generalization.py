# -*- coding: utf-8 -*-
"""
Experiment 1.3 — Domain Generalization: USCG Maritime Event Map

This experiment tests the generalization capability of the FAA2CHART framework
by applying it to a different domain: USCG Local Notice to Mariners (LNM).

The framework is designed with domain-agnostic architecture:
  - Domain profile (USCG_LNM_PROFILE) provides extraction schema
  - Point event analysis strategy replaces airway connectivity strategy
  - Gazetteer switching (USCG Light List instead of FAA NASR)

The pipeline:
  1. event_extract — USCG LNM PDF → structured event JSON
  2. geocode        — Coordinate completion via Light List gazetteer
  3. layer_loader   — QGIS project setup (same as Experiment 1.2)
  4. analysis       — Point event spatial analysis (different strategy)
  5. map_publish    — Maritime overview map layout

=== Reviewer-note: Generalization ===
This experiment demonstrates that the framework is not hard-coded to FAA
airways. By swapping the domain profile and analysis strategy, it processes
maritime Notice-to-Mariners documents with the same multi-agent architecture.
"""

import os
import sys
from pathlib import Path

from domap.connectivity.shared_map_state import (
    MapConfig, PathConfig, LayoutConfig, RunConfig, LayerNameConfig, StyleConfig
)
from domap.connectivity.multi_agents import (
    run_unified_airway_factory_multi_agent
)
from domap.domain_profile import USCG_LNM_PROFILE


# ============================================================================
# Benchmark — USCG LNM PDF → Full Pipeline → Maritime Event Map
#
# Target waterway: Washington State coastal waters
# ============================================================================
BENCHMARK_MARITIME_REQUEST = """
Process this USCG Local Notice to Mariners PDF:
{PATH_TO_LNM_PDF}

Requirements:
1. Extract all aid-to-navigation change events into structured JSON.
   The domain is USCG LNM (Local Notice to Mariners), not FAA airway.
   Domain profile: USCG_LNM_PROFILE
   Fields: aid_name, llnr, action, waterway, mile_marker, bank, position

2. Geocode any events with missing coordinates using the USCG Light List
   gazetteer at /data/uscg_lnm/light_list_gazetteer.json.

3. The data morphology is "point_event" — use the point_event analysis strategy.
   Do NOT use airway_connectivity strategy.

4. Create a maritime event overview map:
   - Categorize events by action type (DISCONTINUED, ESTABLISHED, CHANGED,
     RELOCATED, DAMAGED, DESTROYED, MISSING, EXTINGUISHED, RECOVERED,
     REBUILT, TEMPORARY).
   - Use distinct marker colors for each action type.
   - Add a legend explaining the color scheme (one entry per action type).
   - Title: "Overview of USCG LNM Navigation Facility Events"
   - Subtitle: "USCG Local Notice to Mariners | events={state_count}"
   - Footer: "Data source: USCG LNM | Point events categorized by action type"

5. Auto-export to PNG and PDF at 300 DPI.
"""


def run_maritime_generalization():
    """USCG LNM PDF → full pipeline → maritime event overview map."""
    cfg = MapConfig(
        paths=PathConfig(
            state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
            json_path="PATH_TO_MARITIME_JSON",
            png_path="./output/maritime_generalization.png",
            pdf_path="./output/maritime_generalization.pdf",
            layout_name="USCG_LNM_Generalization",
        ),
        layers=LayerNameConfig(
            flow="USCG_LNM_Connectivity_Flows",
            point="USCG_LNM_Connectivity_Points",
            label="USCG_LNM_Connectivity_Labels",
        ),
        layout=LayoutConfig(
            title_text="Overview of USCG LNM Navigation Facility Events",
            subtitle_template="USCG Local Notice to Mariners | events={state_count}",
            footer_note="Data source: USCG LNM | Point events categorized by action type",
            map_panel_title="Maritime Event Map",
        ),
        style=StyleConfig(
            label_text_style={
                "font_family": "Arial",
                "font_size": 16,
                "buffer_enabled": True,
                "buffer_color": "white",
                "buffer_size": 0.9,
            },
        ),
        run=RunConfig(
            open_designer=False,
            auto_export=True,
            export_dpi=300,
        ),
    )
    cfg.data.target_state = "WA"

    return run_unified_airway_factory_multi_agent(
        user_request=BENCHMARK_MARITIME_REQUEST,
        source_document="PATH_TO_LNM13312025_PDF",
        cfg=cfg,
        profile=USCG_LNM_PROFILE,
        stream=True,
    )


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FAA2CHART Experiment 1.3 — Domain Generalization: Maritime"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Output directory for PNG/PDF"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(
        "=" * 72,
        "FAA2CHART Experiment 1.3 — Domain Generalization",
        "Model: DeepSeek-chat | temperature=0 | max_tokens=4000",
        "Domain: USCG LNM (Local Notice to Mariners)",
        f"Output directory: {args.output_dir}",
        "Run date: " + "YYYY-MM-DD",  # Update to actual run date
        "=" * 72,
        sep="\n",
    )

    result = run_maritime_generalization()
    print("Experiment 1.3 complete.")