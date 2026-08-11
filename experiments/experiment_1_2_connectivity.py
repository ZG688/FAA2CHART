# -*- coding: utf-8 -*-
"""
Experiment 1.2 — Target-State-Centered Connectivity Map (Agent-Readable Format)

This experiment produces the "agent-readable" connectivity map: a
target-state-centered migration-style arc visualization, where arc width
encodes the connectivity score between each source state and the target.

The multi-agent system (supervisor + 5 sub-agents) orchestrates:
  1. event_extract — PDF → structured airway JSON
  2. geocode        — Waypoint coordinate completion via FAA NASR gazetteer
  3. layer_loader   — QGIS project environment setup
  4. analysis       — Connectivity metrics + thematic layer creation
  5. map_publish    — Layout creation, legend, scale bar, north arrow, export

Variants (3 difficulty levels):
  - SIMPLE:   Single-step (pre-extracted JSON → map), e.g., CA target
  - MEDIUM:   Two-step (PDF → JSON + geocode → map), e.g., WA target
  - COMPLEX:  Full pipeline (PDF → JSON → geocode → map), e.g., NY target

The "agent-readable" format is compared against the human-readable overview
(Experiment 1.1) for efficiency and correctness evaluation.

=== Reviewer-note: Reproducibility ===
- All agent prompts are documented in prompts.json at the package root.
- Model settings: DeepSeek-chat, temperature=0, max_tokens=4000.
- Run date: Update to your actual run date.
- The pipeline is fully automated; no manual intervention required.
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
from domap.connectivity.single_agent import (
    run_unified_airway_factory_single_agent
)
from domap.domain_profile import FAA_AIRWAY_PROFILE


# ============================================================================
# Benchmark 1: SIMPLE — Pre-extracted JSON → Connectivity Map
#
# Difficulty: Low (1 step)
# Description: Use an already-extracted and geocoded airway JSON. The agent
#   only needs to perform mapping stages (layer_loader → analysis → map_publish).
# Target: California (CA)
#
# User input (natural language):
#   "Based on the FAA airway JSON at /data/parsed_routes_ca_updated.json,
#    create a California-centered connectivity map with migration-style arcs.
#    Label the top 10 most-connected states."
# ============================================================================
BENCHMARK_SIMPLE_USER_REQUEST = """
Based on the FAA airway JSON at /data/parsed_routes_ca_updated.json,
create a California-centered airway connectivity map.

Requirements:
- Use migration-style arc visualization where arc width encodes connectivity score.
- Highlight the target state (California) in gold/amber.
- Label the top 10 most-connected states with their scores.
- Include a legend explaining arc width = connectivity score.
- Export as 300 DPI PNG and PDF.
"""


def benchmark_simple_json_to_map():
    """Simple connectivity map from pre-extracted JSON."""
    cfg = MapConfig(
        paths=PathConfig(
            json_path="PATH_TO_PRE_EXTRACTED_AIRWAY_JSON",
            state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
            png_path="./output/connectivity_ca_simple.png",
            pdf_path="./output/connectivity_ca_simple.pdf",
            layout_name="FAA_CA_Connectivity_Simple",
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
        user_request=BENCHMARK_SIMPLE_USER_REQUEST,
        cfg=cfg,
        stream=True,
    )


# ============================================================================
# Benchmark 2: MEDIUM — PDF → JSON + Geocode → Connectivity Map
#
# Difficulty: Medium (2 steps)
# Description: Extract airway routes from a PDF, then geocode and map.
#   All sub-agents are involved: event_extract + geocode + mapping.
# Target: Washington (WA)
#
# User input (natural language):
#   "Process this FAA airway PDF:
#    {PDF_URL}
#    1. Extract all airway routes into structured JSON.
#    2. Geocode waypoints using the FAA NASR gazetteer.
#    3. Then create a Washington-centered connectivity map.
#    4. Use migration-style arcs, label the top 10 states.
#    5. Export to {output_dir}/connectivity_wa_medium.png/pdf."
# ============================================================================
BENCHMARK_MEDIUM_USER_REQUEST = """
Process this FAA airway PDF:
https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf

Requirements:
1. Extract all airway routes into structured JSON.
2. Geocode waypoints using the FAA NASR gazetteer.
3. Create a Washington-centered airway connectivity map.
4. Use migration-style arcs where arc width encodes connectivity.
5. Label the top 10 most-connected states with scores.
6. Export as 300 DPI PNG and PDF.
"""


def benchmark_medium_pdf_to_map():
    """Medium connectivity map from PDF with extraction + geocoding."""
    cfg = MapConfig(
        paths=PathConfig(
            json_path="PATH_TO_AIRWAY_JSON",  # Fallback if already extracted
            state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
            png_path="./output/connectivity_wa_medium.png",
            pdf_path="./output/connectivity_wa_medium.pdf",
            layout_name="FAA_WA_Connectivity_Medium",
        ),
        layout=LayoutConfig(
            title_text="Washington-centered Airway Connectivity (Migration-style Arc Map)",
            footer_note="Data source: FAA Part 95 Consolidation (Feb 2025) | Arc width ∝ connectivity score",
        ),
        run=RunConfig(
            open_designer=False,
            auto_export=True,
            export_dpi=300,
        ),
    )
    cfg.data.target_state = "WA"
    cfg.data.label_top_k = 10

    return run_unified_airway_factory_multi_agent(
        user_request=BENCHMARK_MEDIUM_USER_REQUEST,
        cfg=cfg,
        profile=FAA_AIRWAY_PROFILE,
        stream=True,
    )


# ============================================================================
# Benchmark 3: COMPLEX — Full Pipeline + Manual Correction Recovery
#
# Difficulty: High (3 steps with failure recovery)
# Description: Full pipeline from PDF URL → extraction → geocode →
#   connectivity map. Includes handling for partial failures (missing
#   coordinates, extraction errors) and retry logic.
# Target: New York (NY)
#
# User input (natural language):
#   "Process FAA Part 95 PDF from the FAA website.
#   Extract all routes, geocode waypoints, verify coordinate completeness.
#   If any coordinates are missing, attempt recovery via alternative gazetteer.
#   Then create a New York-centered connectivity map with:
#   - Migration-style arcs scaled by connectivity score.
#   - Labels for the top 10 states.
#   - Full layout with title, subtitle, legend, scale bar, north arrow.
#   - Export to PNG and PDF at 300 DPI."
# ============================================================================
BENCHMARK_COMPLEX_USER_REQUEST = """
Process this FAA airway PDF from the FAA website:
https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf

Requirements:
1. Extract all airway routes into structured JSON with airway_point arrays.
2. Verify that each airway_point has a region code (2-letter state abbreviation).
3. Geocode all waypoints that are missing coordinates (position=[none, none]).
   Use the FAA NASR gazetteer CSV data for coordinate lookup.
4. Report on coordinate coverage: how many waypoints were resolved, how many remain missing.
5. Create a New-York-centered airway connectivity map with:
   - Migration-style Bezier arcs scaled by connectivity score.
   - Gold highlight for the target state (NY).
   - State abbreviation labels for the top 10 most-connected states.
   - A legend explaining arc width = 2×direct_seg + airway_cnt.
6. Full layout: title, subtitle, legend, scale bar (nautical miles), north arrow.
7. Auto-export to PNG and PDF at 300 DPI.
8. If any stage encounters errors, retry once before failing.
"""


def benchmark_complex_full_pipeline():
    """Complex full-pipeline connectivity map with error recovery."""
    cfg = MapConfig(
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
            subtitle_template="Target: NY | States mapped: {state_count} | Score = 2×direct_seg + airway_cnt",
            footer_note="Data source: FAA Part 95 (Feb 2025) | Extracted by FAA2CHART multi-agent framework",
            map_panel_title="Connectivity Arc Map",
        ),
        style=StyleConfig(
            label_text_style={
                "font_family": "Arial",
                "font_size": 9,
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
    cfg.data.target_state = "NY"
    cfg.data.label_top_k = 10

    return run_unified_airway_factory_multi_agent(
        user_request=BENCHMARK_COMPLEX_USER_REQUEST,
        cfg=cfg,
        profile=FAA_AIRWAY_PROFILE,
        stream=True,
    )


# ============================================================================
# Single-Agent Baseline (for comparison with multi-agent)
# ============================================================================
BENCHMARK_SINGLE_AGENT_REQUEST = """
Process FAA Part 95 PDF from:
https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf

Full pipeline as a single agent:
1. Extract airway routes into structured JSON.
2. Geocode waypoints with missing coordinates.
3. Create a California-centered connectivity map with migration arcs.
4. Label top 5 states, export to PNG/PDF.
"""


def benchmark_single_agent():
    """Single-agent baseline for comparison against multi-agent."""
    cfg = MapConfig(
        paths=PathConfig(
            json_path="PATH_TO_AIRWAY_JSON",
            state_boundary_path="PATH_TO_NE_10M_STATES_SHP",
            png_path="./output/connectivity_ca_single_agent.png",
            pdf_path="./output/connectivity_ca_single_agent.pdf",
            layout_name="FAA_CA_Connectivity_SingleAgent",
        ),
        run=RunConfig(
            open_designer=False,
            auto_export=True,
            export_dpi=300,
        ),
    )
    cfg.data.target_state = "CA"
    cfg.data.label_top_k = 5

    return run_unified_airway_factory_single_agent(
        user_request=BENCHMARK_SINGLE_AGENT_REQUEST,
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
        description="FAA2CHART Experiment 1.2 — Connectivity Map (Agent-Readable)"
    )
    parser.add_argument(
        "--variant",
        choices=["simple", "medium", "complex", "single"],
        default="simple",
        help="Benchmark variant to run"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Output directory for PNG/PDF"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    benchmarks = {
        "simple":  benchmark_simple_json_to_map,
        "medium":  benchmark_medium_pdf_to_map,
        "complex": benchmark_complex_full_pipeline,
        "single":  benchmark_single_agent,
    }

    print(
        "=" * 72,
        f"FAA2CHART Experiment 1.2 — Connectivity Map — {args.variant.upper()}",
        "Model: DeepSeek-chat | temperature=0 | max_tokens=4000",
        f"Output directory: {args.output_dir}",
        "Run date: " + "YYYY-MM-DD",  # Update to actual run date
        "=" * 72,
        sep="\n",
    )

    result = benchmarks[args.variant]()
    print("Experiment 1.2 complete.")