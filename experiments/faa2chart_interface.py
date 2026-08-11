# -*- coding: utf-8 -*-
"""
FAA2CHART — Unified Natural-Language Interface
===============================================

This is the single entry point for all FAA2CHART experiments. The user
provides a natural-language request, and the system automatically:
  1. Classifies the task type (overview / connectivity / maritime)
  2. Selects the appropriate domain profile and analysis strategy
  3. Configures the multi-agent pipeline
  4. Executes the full extraction → geocoding → mapping pipeline

Three task types are supported:
  ┌─────────────────────┬──────────────────────┬───────────────────────┐
  │ Task                │ Domain Profile       │ Analysis Strategy     │
  ├─────────────────────┼──────────────────────┼───────────────────────┤
  │ overview            │ FAA_AIRWAY_PROFILE   │ airway_connectivity   │
  │ connectivity        │ FAA_AIRWAY_PROFILE   │ airway_connectivity   │
  │ maritime            │ USCG_LNM_PROFILE     │ point_event           │
  └─────────────────────┴──────────────────────┴───────────────────────┘

=== Reviewer-note: Benchmark Reproducibility ===
All 8 benchmark examples below correspond to the experiments described in
the paper. Each benchmark is self-contained with:
  - Pre-configured MapConfig (paths, layout, export settings)
  - Exact natural-language user request text
  - Expected output specification
  - Model: DeepSeek-chat, temperature=0, max_tokens=4000

To reproduce:
  python faa2chart_interface.py --benchmark 1
  python faa2chart_interface.py --benchmark all
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Dict, Any

from domap.connectivity.shared_map_state import (
    MapConfig, PathConfig, LayoutConfig, RunConfig, LayerNameConfig, StyleConfig
)
from domap.connectivity.multi_agents import (
    run_unified_airway_factory_multi_agent
)
from domap.connectivity.single_agent import (
    run_unified_airway_factory_single_agent
)
from domap.domain_profile import FAA_AIRWAY_PROFILE, USCG_LNM_PROFILE


# ============================================================================
# Benchmark Registry
# ============================================================================

BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Experiment 1.1 — Overview Map (Human-Readable)
    # ------------------------------------------------------------------
    "1_ca_overview": {
        "description": "Experiment 1.1 (CA): CA overview, airway classification",
        "task_type": "overview",
        "agent_mode": "multi",
        "cfg": MapConfig(
            paths=PathConfig(
                json_path="PATH_TO_YOUR_AIRWAY_JSON",
                state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
                png_path="./output/overview_ca.png",
                pdf_path="./output/overview_ca.pdf",
                layout_name="FAA_CA_Overview",
            ),
            run=RunConfig(open_designer=False, auto_export=True, export_dpi=300),
        ),
        "profile": FAA_AIRWAY_PROFILE,
        "user_request": (
            "Extract airway routes from the PDF, geocode waypoints, "
            "then create a California-centered overview map with airway "
            "classification labels. Highlight target state, show all "
            "airway segments colored by route type."
        ),
        "source_document": "https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf",
    },
    "1_ny_overview": {
        "description": "Experiment 1.1 (NY): NY overview, detailed classification",
        "task_type": "overview",
        "agent_mode": "multi",
        "cfg": MapConfig(
            paths=PathConfig(
                json_path="PATH_TO_YOUR_AIRWAY_JSON",
                state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
                png_path="./output/overview_ny.png",
                pdf_path="./output/overview_ny.pdf",
                layout_name="FAA_NY_Overview",
            ),
            layout=LayoutConfig(
                title_text="FAA Part 95 IFR Airway Network — New York Region",
                subtitle_template="Segments: {segment_count} | Nodes: {node_count}",
                footer_note="Colored: Blue=Colored Fed, Green=VOR, Orange=Low RNAV, Purple=High RNAV, Red=Jet",
            ),
            run=RunConfig(open_designer=False, auto_export=True, export_dpi=300),
        ),
        "profile": FAA_AIRWAY_PROFILE,
        "user_request": (
            "Extract airway routes from the PDF, geocode waypoints, "
            "then create a New-York-centered overview map with all "
            "airway segments colored by route type. Include a detailed "
            "legend explaining the color scheme."
        ),
        "source_document": "https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf",
    },
    "1_us_overview": {
        "description": "Experiment 1.1 (US): Full U.S. 4-panel layout",
        "task_type": "overview",
        "agent_mode": "multi",
        "cfg": MapConfig(
            paths=PathConfig(
                json_path="PATH_TO_YOUR_AIRWAY_JSON",
                state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
                png_path="./output/overview_us.png",
                pdf_path="./output/overview_us.pdf",
                layout_name="FAA_US_Overview",
            ),
            layout=LayoutConfig(
                title_text="FAA Part 95 IFR Federal Airway Network — Continental U.S.",
                footer_note="Data source: FAA Part 95 (Feb 2025) | Extraction: DeepSeek-chat + MinerU",
            ),
            run=RunConfig(open_designer=False, auto_export=True, export_dpi=300),
        ),
        "profile": FAA_AIRWAY_PROFILE,
        "user_request": (
            "Extract all airway routes from the PDF, geocode waypoints, "
            "then create a full continental U.S. 4-panel overview map. "
            "Color-code all airway segments by type. Show the top 30 "
            "most connected waypoints with labels."
        ),
        "source_document": "https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf",
    },

    # ------------------------------------------------------------------
    # Experiment 1.2 — Connectivity Map (Agent-Readable)
    # ------------------------------------------------------------------
    "2_simple_connectivity": {
        "description": "Experiment 1.2 (Simple): CA connectivity, pre-extracted JSON",
        "task_type": "connectivity",
        "agent_mode": "multi",
        "cfg": MapConfig(
            paths=PathConfig(
                json_path="PATH_TO_PRE_EXTRACTED_AIRWAY_JSON",
                state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
                png_path="./output/connectivity_ca_simple.png",
                pdf_path="./output/connectivity_ca_simple.pdf",
                layout_name="FAA_CA_Connectivity_Simple",
            ),
            run=RunConfig(open_designer=False, auto_export=True, export_dpi=300),
        ),
        "profile": FAA_AIRWAY_PROFILE,
        "user_request": (
            "Based on the FAA airway JSON at /data/parsed_routes_ca_updated.json, "
            "create a California-centered airway connectivity map with migration-style "
            "arcs, label the top 10 most-connected states."
        ),
    },
    "2_medium_connectivity": {
        "description": "Experiment 1.2 (Medium): WA connectivity from PDF",
        "task_type": "connectivity",
        "agent_mode": "multi",
        "cfg": MapConfig(
            paths=PathConfig(
                json_path="PATH_TO_AIRWAY_JSON",
                state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
                png_path="./output/connectivity_wa_medium.png",
                pdf_path="./output/connectivity_wa_medium.pdf",
                layout_name="FAA_WA_Connectivity_Medium",
            ),
            layout=LayoutConfig(
                title_text="Washington-centered Airway Connectivity (Migration-style Arc Map)",
                footer_note="Data source: FAA Part 95 (Feb 2025) | Arc width ∝ connectivity score",
            ),
            run=RunConfig(open_designer=False, auto_export=True, export_dpi=300),
        ),
        "profile": FAA_AIRWAY_PROFILE,
        "user_request": (
            "Process FAA Part 95 PDF from FAA website. Extract all airway routes "
            "into structured JSON, geocode waypoints, then create a Washington-centered "
            "connectivity map with migration-style arcs, label the top 10 states."
        ),
        "source_document": "https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf",
    },
    "2_complex_connectivity": {
        "description": "Experiment 1.2 (Complex): NY full pipeline + error recovery",
        "task_type": "connectivity",
        "agent_mode": "multi",
        "cfg": MapConfig(
            paths=PathConfig(
                json_path="PATH_TO_AIRWAY_JSON",
                state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
                png_path="./output/connectivity_ny_complex.png",
                pdf_path="./output/connectivity_ny_complex.pdf",
                layout_name="FAA_NY_Connectivity_Complex",
            ),
            layers=LayerNameConfig(
                flow="FAA_NY_Connectivity_Flows",
                point="FAA_NY_Connectivity_Points",
                label="FAA_NY_Connectivity_Labels",
            ),
            layout=LayoutConfig(
                title_text="NY-Centered Federal Airway Connectivity Map",
                footer_note="Data source: FAA Part 95 (Feb 2025) | Arc width ∝ connectivity score",
            ),
            style=StyleConfig(
                label_text_style={
                    "font_family": "Arial", "font_size": 9,
                    "buffer_enabled": True, "buffer_color": "white", "buffer_size": 0.9,
                },
            ),
            run=RunConfig(open_designer=False, auto_export=True, export_dpi=300),
        ),
        "profile": FAA_AIRWAY_PROFILE,
        "user_request": (
            "Process FAA Part 95 PDF from FAA website. Full pipeline: extract routes, "
            "geocode waypoints, verify coordinate completeness, create New-York-centered "
            "connectivity map with migration arcs, label top 10 states, full layout with "
            "legend/scale bar/north arrow, export to PNG/PDF at 300 DPI."
        ),
        "source_document": "https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf",
    },
    "2_single_agent": {
        "description": "Experiment 1.2 (Single-Agent Baseline): CA connectivity, single agent",
        "task_type": "connectivity",
        "agent_mode": "single",
        "cfg": MapConfig(
            paths=PathConfig(
                json_path="PATH_TO_AIRWAY_JSON",
                state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
                png_path="./output/connectivity_ca_single_agent.png",
                pdf_path="./output/connectivity_ca_single_agent.pdf",
                layout_name="FAA_CA_Connectivity_SingleAgent",
            ),
            run=RunConfig(open_designer=False, auto_export=True, export_dpi=300),
        ),
        "profile": FAA_AIRWAY_PROFILE,
        "user_request": (
            "Process FAA Part 95 PDF from FAA website. Full pipeline as a single agent: "
            "extract airway routes, geocode waypoints, create California-centered "
            "connectivity map with migration arcs, label top 5 states, export to PNG/PDF."
        ),
        "source_document": "https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf",
    },

    # ------------------------------------------------------------------
    # Experiment 1.3 — Domain Generalization: Maritime
    # ------------------------------------------------------------------
    "3_maritime_generalization": {
        "description": "Experiment 1.3: USCG LNM PDF → full pipeline → maritime event map",
        "task_type": "maritime",
        "agent_mode": "multi",
        "cfg": MapConfig(
            paths=PathConfig(
                json_path="PATH_TO_MARITIME_JSON",
                state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
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
                subtitle_template="USCG LNM | events={state_count}",
                footer_note="Data source: USCG LNM | Point events categorized by action type",
            ),
            run=RunConfig(open_designer=False, auto_export=True, export_dpi=300),
        ),
        "profile": USCG_LNM_PROFILE,
        "user_request": (
            "Process USCG LNM PDF. Extract aid-to-navigation events using USCG_LNM_PROFILE. "
            "Geocode with Light List gazetteer. Use point_event analysis strategy. "
            "Create maritime overview map with 11 action-type categories. "
            "Export at 300 DPI."
        ),
        "source_document": "PATH_TO_LNM13312025_PDF",
    },
}


# ============================================================================
# Natural-Language Interface
# ============================================================================

def run_benchmark(name: str, **overrides):
    """Run a single benchmark by name."""
    if name not in BENCHMARKS:
        available = "\n  ".join(BENCHMARKS.keys())
        raise ValueError(
            f"Unknown benchmark: {name}\nAvailable:\n  {available}"
        )

    bm = BENCHMARKS[name]
    cfg = bm["cfg"]
    profile = bm.get("profile")
    source_document = bm.get("source_document")
    user_request = bm["user_request"]
    agent_mode = bm["agent_mode"]

    # Apply any overrides
    if overrides:
        cfg.data.target_state = overrides.get("target_state", cfg.data.target_state)
        cfg.data.label_top_k = overrides.get("label_top_k", cfg.data.label_top_k)

    print(f"\n{'='*72}")
    print(f"Benchmark: {name}")
    print(f"Description: {bm['description']}")
    print(f"Agent mode: {agent_mode}")
    print(f"{'='*72}\n")

    if agent_mode == "single":
        return run_unified_airway_factory_single_agent(
            user_request=user_request,
            cfg=cfg,
            profile=profile,
            stream=True,
        )
    else:
        return run_unified_airway_factory_multi_agent(
            user_request=user_request,
            cfg=cfg,
            profile=profile,
            source_document=source_document,
            stream=True,
        )


def run_task(
    task_type: str,
    user_request: str,
    cfg: MapConfig = None,
    agent_mode: str = "multi",
    source_document: str = None,
):
    """
    Unified natural-language entry point.

    Args:
        task_type: "overview", "connectivity", or "maritime"
        user_request: Natural-language task description
        cfg: MapConfig (optional, uses defaults if not provided)
        agent_mode: "multi" or "single"
        source_document: PDF URL or path (for extraction)

    Returns:
        Agent execution result
    """
    # Select domain profile
    profile_map = {
        "overview": FAA_AIRWAY_PROFILE,
        "connectivity": FAA_AIRWAY_PROFILE,
        "maritime": USCG_LNM_PROFILE,
    }
    profile = profile_map.get(task_type, FAA_AIRWAY_PROFILE)

    # Default config if none provided
    if cfg is None:
        cfg = MapConfig(
            paths=PathConfig(
                json_path="PATH_TO_YOUR_AIRWAY_JSON",
                state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
                png_path=f"./output/{task_type}_output.png",
                pdf_path=f"./output/{task_type}_output.pdf",
                layout_name=f"FAA2CHART_{task_type}",
            ),
            run=RunConfig(
                open_designer=False,
                auto_export=True,
                export_dpi=300,
            ),
        )

    run_fn = (
        run_unified_airway_factory_single_agent
        if agent_mode == "single"
        else run_unified_airway_factory_multi_agent
    )

    return run_fn(
        user_request=user_request,
        cfg=cfg,
        profile=profile,
        source_document=source_document,
        stream=True,
    )


# ============================================================================
# CLI Entry Point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FAA2CHART — Unified Natural-Language Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark IDs:
  Experiment 1.1 (Overview):
    1_ca_overview, 1_ny_overview, 1_us_overview
  Experiment 1.2 (Connectivity):
    2_simple_connectivity, 2_medium_connectivity, 2_complex_connectivity, 2_single_agent
  Experiment 1.3 (Maritime):
    3_maritime_generalization

Examples:
  python faa2chart_interface.py --benchmark 2_simple_connectivity
  python faa2chart_interface.py --benchmark all
  python faa2chart_interface.py --task connectivity --request "Draw a CA connectivity map..."
        """,
    )

    parser.add_argument(
        "--benchmark", type=str, default=None,
        help="Benchmark ID to run (or 'all' to run all benchmarks)"
    )
    parser.add_argument(
        "--task", type=str, choices=["overview", "connectivity", "maritime"],
        help="Task type for custom natural-language request"
    )
    parser.add_argument(
        "--request", type=str,
        help="Natural-language task description (requires --task)"
    )
    parser.add_argument(
        "--agent", type=str, choices=["multi", "single"], default="multi",
        help="Agent mode: multi (default) or single"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Output directory for PNG/PDF"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.benchmark:
        if args.benchmark == "all":
            for bm_name in BENCHMARKS:
                try:
                    run_benchmark(bm_name)
                except Exception as e:
                    print(f"Benchmark {bm_name} failed: {e}")
        else:
            run_benchmark(args.benchmark)
    elif args.task and args.request:
        run_task(
            task_type=args.task,
            user_request=args.request,
            agent_mode=args.agent,
        )
    else:
        parser.print_help()
        print("\nAvailable benchmarks:")
        for bm_name, bm_info in BENCHMARKS.items():
            print(f"  {bm_name:30s} — {bm_info['description']}")