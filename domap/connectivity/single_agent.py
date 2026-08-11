# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool

# ===================================
# Environment variable check
# ===================================
def ensure_env() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set. Please set it before running.")

    # LangSmith is optional
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", "deep-agents-single-agent")
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    print("Environment variable check completed")

# =========================
# Import existing tools
# =========================
from domap.event_extract import parse_doc_tool
from domap.name_postion import update_airway_coordinates_tool, find_file_by_name_tool

from domap.connectivity.shared_map_state import MapConfig, PathConfig
from domap.connectivity.agent_job_tools import (
    get_deepagent_job_tools,
    get_layer_loader_job_tools,
    get_spatial_analysis_job_tools,
    get_map_decoration_job_tools,
)

from domap.connectivity.virtual_paths import (
    to_virtual_path,
    sanitize_user_request_paths,
    to_host_path,
)

# =========================
# Initialize model
# =========================
def build_model():
    ensure_env()
    print("Initializing DeepSeek model...")
    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=4000,
        timeout=60,
        max_retries=3,
    )
    print("Model initialized")
    return model


# =========================
# Tool merge and dedup
# =========================
def _tool_name(t) -> str:
    if hasattr(t, "name") and t.name:
        return t.name
    if hasattr(t, "__name__"):
        return t.__name__
    return str(t)


def merge_tools(*tool_groups) -> List:
    merged = []
    seen = set()

    for group in tool_groups:
        for t in group:
            name = _tool_name(t)
            if name not in seen:
                seen.add(name)
                merged.append(t)
    return merged


# =========================
# Select final JSON path tool
# =========================
@tool("choose_final_json_path_tool", parse_docstring=True)
def choose_final_json_path_tool(
    user_json_path: str = "",
    geocoded_json_path: str = "",
    extracted_json_path: str = "",
    fallback_json_path: str = "",
) -> Dict[str, Any]:
    """
    Select the final JSON path for entering the mapping stage.

    Priority:
    1. User-specified JSON path
    2. geocoded_json_path from the geocode stage
    3. extracted_json_path from the event_extract stage
    4. Default fallback_json_path

    Important constraints:
    - Do not rename files
    - Do not move files
    - Do not fabricate new virtual paths
    - Only return the canonical representation of an actual result path

    Args:
        user_json_path: JSON path explicitly specified by the user.
        geocoded_json_path: JSON path after geocoding.
        extracted_json_path: JSON path from event extraction.
        fallback_json_path: Default JSON path.

    Returns:
        Dict containing the final host_path / virtual_path / source.
    """
    candidates = [
        ("user_json_path", user_json_path),
        ("geocoded_json_path", geocoded_json_path),
        ("extracted_json_path", extracted_json_path),
        ("fallback_json_path", fallback_json_path),
    ]

    chosen_source = None
    chosen_raw = None

    for source, value in candidates:
        if isinstance(value, str) and value.strip():
            chosen_source = source
            chosen_raw = value.strip()
            break

    if not chosen_raw:
        raise ValueError("No available JSON path for entering the mapping stage.")

    if chosen_raw.startswith("/"):
        host_path = to_host_path(chosen_raw)
        virtual_path = chosen_raw
    else:
        host_path = chosen_raw
        virtual_path = to_virtual_path(chosen_raw)

    return {
        "ok": True,
        "source": chosen_source,
        "selected_json_host_path": host_path,
        "selected_json_virtual_path": virtual_path,
    }


# =========================
# Single agent prompt
# =========================
SINGLE_AGENT_PROMPT = """
You are the "Aeronautical Chart Factory Single Agent" and must complete the task serially as a single agent.

[Absolute Constraints]
1. Do not decompose the task into multiple roles, multiple agents, or multiple sub-teams.
2. Do not pretend in your answer to invoke "Event Extraction Sub-Agent / Geocoding Sub-Agent / Layer Loading Sub-Agent" or similar roles.
3. You must complete the following sequentially yourself: task understanding -> path decision -> PDF extraction / JSON geocoding -> mapping.
4. No parallel planning, no self-division of labor, no Supervisor/Worker narrative.
5. You may only use the tools currently provided to you to complete the task directly.

[Task Types]
You handle comprehensive FAA airway-related tasks, which may include one or more of the following stages:
- PDF -> JSON event extraction
- JSON -> coordinate completion / geocoding
- JSON -> connectivity map generation

[Stage Decision Rules]
A. If the user provides a PDF URL or PDF virtual path and needs to extract aeronautical data:
   First call parse_doc_tool.

B. If the user provides an existing JSON path:
   Skip PDF extraction.

C. If the JSON does not yet have spatial coordinates, or the task explicitly requires "geocoding / coordinate completion":
   Call find_file_by_name_tool (only if only a filename is given) and update_airway_coordinates_tool.

D. If the user explicitly requires "mapping / connectivity map / map output / layout / export":
   After extraction/geocoding completes, continue with:
   1) create_connectivity_job_tool
   2) Layer loading related tools
   3) Spatial analysis related tools
   4) Map decoration related tools

E. If the user only requires JSON extraction, or only requires geocoding:
   Stop after completing the corresponding stage; do not force mapping.

[JSON Path Selection Hard Rules]
- If the user has not explicitly specified a new JSON path, when entering the mapping stage you must prioritize the geocoded_json_path actually output by update_airway_coordinates_tool.
- If geocoding was not performed, use the extracted_json_path produced by parse_doc_tool.
- If neither is available, fall back to the json_path in the default config.
- Before entering the mapping stage, you must call choose_final_json_path_tool to select the final JSON path.
- Do not rename actual output paths to new virtual paths on your own.
- Do not assume that a "suggested path" is an actual existing file.
- Do not fabricate non-existent *_geocoded.json / *_updated.json paths.

[Path Rules]
- All paths passed to GIS mapping tools must use virtual paths (starting with /).
- Do not use Windows absolute paths (e.g. D:/... or C:/...) in tool parameters or answers.
- Only after choose_final_json_path_tool returns may you feed the final JSON path into the mapping stage.

[Mapping Stage General Principles]
- First create a job, then execute in order; do not skip steps, do not reverse the order:
  Step 1: create_connectivity_job_tool
  Step 2: Layer loading
  Step 3: Spatial analysis
  Step 4: Map decoration
- All subsequent operations must target the same job_id.
- Before publish_create_layout_tool succeeds, do not call other layout tools.
- If auto_export=False, you must explicitly indicate in the final result how to manually export.

[Result Inspection Requirements]
If mapping was performed, you must at minimum check and summarize the following before finishing:
- job_id
- target_state
- connected_state_count (if available from tool summaries)
- layer_keys
- analysis_keys
- layout_keys
- png_path
- pdf_path
- layout_name
- Whether auto-export was performed

[Final Output Format Requirements]
Output concise results based on the stages actually executed:

1. If only extraction/geocoding was performed:
- extracted_json_path / geocoded_json_path
- Whether completed
- Key summary

2. If mapping was completed:
- job_id
- target_state
- connected_state_count
- layer_keys
- layout_keys
- png_path
- pdf_path
- layout_name
- Whether auto-export was executed
- Key analysis findings (at least 3)

[Style Requirements]
- Output should be concise, verifiable, and execution-result-oriented.
- Do not explain the thought process at length.
"""


# =========================
# Create single agent
# =========================
def build_single_agent(model):
    print("Creating single agent...")

    all_tools = merge_tools(
        [parse_doc_tool, find_file_by_name_tool, update_airway_coordinates_tool, choose_final_json_path_tool],
        get_deepagent_job_tools(),
        get_layer_loader_job_tools(),
        get_spatial_analysis_job_tools(),
        get_map_decoration_job_tools(),
    )

    agent = create_agent(
        model=model,
        tools=all_tools,
        system_prompt=SINGLE_AGENT_PROMPT,
        middleware=[
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=80000,
                        keep=2,
                        clear_tool_inputs=True,
                    )
                ]
            )
        ],
    )

    print(f"Single agent created, loaded {len(all_tools)} tools")
    return agent


# =========================
# Assemble user task
# =========================
def build_user_task_from_natural_language(
    user_request: str,
    cfg: Optional[MapConfig] = None,
    extra_requirement: str = "",
) -> str:
    user_request_sanitized = sanitize_user_request_paths(user_request, cfg)

    if cfg is None:
        cfg_block = """
[Default Mapping Config]
- No default config object provided.
- If the task enters the mapping stage, you need to extract from the user's natural language:
  state_boundary_path / json_path / png_path / pdf_path / layout_name / target_state etc.
- All paths must use virtual paths (starting with /).
        """.strip()
    else:
        layers = cfg.layers
        cfg_block = f"""
[Default Mapping Config (usable when entering the mapping stage)]
- state_boundary_path: {to_virtual_path(cfg.paths.state_boundary_path)}
- json_path: {to_virtual_path(cfg.paths.json_path)}
- png_path: {to_virtual_path(cfg.paths.png_path)}
- pdf_path: {to_virtual_path(cfg.paths.pdf_path)}
- layout_name: {cfg.paths.layout_name}
- target_state: {cfg.data.target_state}
- top_n_states: {cfg.data.top_n_states}
- label_top_k: {cfg.data.label_top_k}
- open_designer: {cfg.run.open_designer}
- auto_export: {cfg.run.auto_export}
- export_dpi: {cfg.run.export_dpi}
[Layer Name Config (use when calling create_connectivity_job_tool)]
- layer_point_name: {layers.point}
- layer_flow_name: {layers.flow}
- layer_label_name: {layers.label}
- layer_state_source_name: {layers.state_source}
- layer_state_display_name: {layers.state_display}
- layer_target_display_name: {layers.target_display}
        """.strip()

    return f"""
You will receive a comprehensive task. This task may include only part of the pipeline, or the full pipeline:

- PDF -> JSON event extraction
- JSON -> coordinate completion / geocoding
- JSON -> connectivity map generation

You must first understand what the user actually needs done, then decide which tools to invoke.

[User Requirements]
{user_request_sanitized}

{cfg_block}

[Execution Requirements]
1. First determine whether the input is a PDF, JSON, or just a filename;
2. First determine whether the user only wants extraction/geocoding, or also wants mapping;
3. If the input is a PDF and needs extraction, first call parse_doc_tool;
4. If the JSON needs coordinate completion, call find_file_by_name_tool and/or update_airway_coordinates_tool;
5. If the user needs mapping, then proceed to create_connectivity_job_tool + layer loading + spatial analysis + map decoration;
6. When entering the mapping stage, json_path must use the "final mappable JSON path" obtained from previous stages;
7. All GIS tool paths must use virtual paths (starting with /);
8. Do not use Windows absolute paths (e.g. D:/... or C:/...) in any tool parameters or answers;
9. If a previous stage produced a new JSON file path, and the user has not explicitly requested a new filename, the subsequent mapping must use that actual output path directly;
10. If no new JSON was produced by previous stages, only then may the mapping stage fall back to the json_path in the default config;
11. Finally, output concise results based on the stages actually completed.

[Additional Requirements]
{extra_requirement if extra_requirement else "None"}
    """.strip()


# =========================
# Stream printing
# =========================
def pretty_print_stream_chunk(chunk: Dict[str, Any]) -> None:
    try:
        if not isinstance(chunk, dict) or not chunk:
            print(f"   Non-standard chunk: {chunk}")
            return

        node_name = list(chunk.keys())[0]
        node_data = chunk[node_name]

        print(f"\n[{node_name}] executing...")

        if isinstance(node_data, dict):
            messages = node_data.get("messages")
            if messages:
                if hasattr(messages, "value"):
                    latest_msg = messages.value[-1] if messages.value else None
                else:
                    latest_msg = messages[-1] if messages else None

                if latest_msg is not None:
                    content = getattr(latest_msg, "content", None)
                    if content:
                        text = str(content).strip()
                        if text:
                            print(f"   {text[:300]}")

                    tool_calls = getattr(latest_msg, "tool_calls", None)
                    if tool_calls:
                        print("   Tool call details:")
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                tool_name = tc.get("name", "unknown")
                                tool_args = tc.get("args", {})
                            else:
                                tool_name = getattr(tc, "name", "unknown")
                                tool_args = getattr(tc, "args", {})
                            print(f"      - Tool name: {tool_name}")
                            print(f"        Args: {tool_args}")
            else:
                print(f"   Output data: {node_data}")
        else:
            print(f"   Output data: {node_data}")

        print("   " + "-" * 60)

    except Exception as e:
        print(f"   Chunk parse failed: {e}")
        print(f"   Raw chunk: {chunk}")


# =========================
# Unified entry point
# =========================
def run_unified_airway_factory_single_agent(
    user_request: str,
    cfg: Optional[MapConfig] = None,
    extra_requirement: str = "",
    stream: bool = True,
    show_progress: bool = True,
):
    if cfg is not None:
        print("[DEBUG] fixed cfg json_path =", cfg.paths.json_path)
    else:
        print("[DEBUG] cfg is None")

    model = build_model()
    agent = build_single_agent(model)

    user_task = build_user_task_from_natural_language(
        user_request=user_request,
        cfg=cfg,
        extra_requirement=extra_requirement,
    )

    print("\n[DEBUG] user_task sent to single agent:")
    print(user_task)
    print("-" * 80)

    input_data = {
        "messages": [
            {
                "role": "user",
                "content": user_task
            }
        ]
    }

    print("\nStarting unified single agent task...")
    print("=" * 80)

    if not stream:
        try:
            result = agent.invoke(input_data)
            print("\nSingle agent task completed!")
            return result
        except Exception as e:
            print(f"\nExecution error: {str(e)}")
            print(f"Detailed error:\n{traceback.format_exc()}")
            raise

    collected_chunks = []

    try:
        for chunk in agent.stream(input_data, stream_mode="updates"):
            collected_chunks.append(chunk)
            if show_progress:
                pretty_print_stream_chunk(chunk)

        print("\nSingle agent task completed!")
        if os.getenv("LANGCHAIN_API_KEY"):
            print("View full trace: https://smith.langchain.com")
        return collected_chunks

    except Exception as e:
        print(f"\nExecution error: {str(e)}")
        print(f"Detailed error:\n{traceback.format_exc()}")
        raise


# =========================
# Optional: default cfg
# =========================
def build_default_cfg():
    cfg = MapConfig(
        paths=PathConfig(
            json_path=r"PATH_TO_AIRWAY_JSON",
            state_boundary_path=r"PATH_TO_STATE_BOUNDARY_SHP",
            png_path=r"PATH_TO_OUTPUT_PNG",
            pdf_path=r"PATH_TO_OUTPUT_PDF",
            layout_name="FAA_CA_Connectivity_Map",
        )
    )
    cfg.data.target_state = "CA"
    cfg.data.top_n_states = None
    cfg.data.label_top_k = 10
    cfg.run.open_designer = False
    cfg.run.auto_export = False
    cfg.run.export_dpi = 300
    return cfg


if __name__ == "__main__":
    cfg = build_default_cfg()

    user_request = """
Please process this PDF aeronautical chart:
https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf/Part_95_Consolidation_February_2025.pdf

Requirements:
1. First convert the PDF to structured JSON;
2. Geocode the airway waypoints;
3. Then draw a target-state-centered airway connectivity map for California;
4. Preserve migration-map-style arc representation and label the top 5 most connected states;
"""

    result = run_unified_airway_factory_single_agent(
        user_request=user_request,
        cfg=cfg,
        extra_requirement="If not auto-exported, please clearly indicate how to manually export in the final result.",
        stream=True,
        show_progress=True,
    )

    print("\n================ SINGLE AGENT RESULT ================\n")
    print(result)