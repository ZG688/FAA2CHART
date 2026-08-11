# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional

from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
from langgraph.store.memory import InMemoryStore
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.backends import FilesystemBackend

# ===================================
# Environment variable configuration
# ===================================
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "YOUR_DEEPSEEK_API_KEY"

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
os.environ["LANGCHAIN_PROJECT"] = "deep-agents-debug"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

print("Environment variables configured")

# =========================
# Import existing tools
# =========================
# ---- Event extraction / geocoding ----
from domap.event_extract import parse_doc_tool
from domap.name_postion import update_airway_coordinates_tool, find_file_by_name_tool, make_update_coordinates_tool
from domap.domain_profile import DomainProfile, FAA_AIRWAY_PROFILE

# ---- Mapping main workflow job-aware tools ----
from domap.connectivity.shared_map_state import MapConfig, PathConfig
from domap.connectivity.agent_job_tools import (
    get_deepagent_job_tools,
    get_layer_loader_job_tools,
    get_spatial_analysis_job_tools,
    get_map_decoration_job_tools,
)

# ---- Virtual paths / backend ----
from domap.connectivity.virtual_paths import (
    HOST_ROOT,
    to_virtual_path,
    maybe_to_virtual_path,
    sanitize_user_request_paths,
)

# =========================
# Initialize model
# =========================
def build_model():
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
# Generic sub-agent creator
# =========================
def create_specialized_agent(model, tools: List, system_prompt: str, name: str = ""):
    """
    Create a specialized sub-agent.
    """
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
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
        ]
    )
    print(f"{name} sub-agent created")
    return agent

from langchain_core.tools import tool
from domap.connectivity.virtual_paths import to_virtual_path, to_host_path

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
    2. geocoded_json_path from the geocode sub-agent
    3. extracted_json_path from the event_extract sub-agent
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
# Sub-agent prompts
# =========================

EVENT_EXTRACT_PROMPT = """
You are the "Event Extraction Sub-Agent", responsible for parsing PDF aeronautical documents into structured JSON for mapping.

Your responsibilities:
1. Extract the source document path specified in the [Step 1] of the supervisor task;
2. Call parse_doc_tool, with file_url set to exactly that path, character for character;
3. Output the structured extraction results;
4. Clearly tell the supervisor: what is the generated JSON path.

Strict requirements:
- You may only use the parse_doc_tool.
- The path given in the supervisor task's [Step 1] is the file_url for parse_doc_tool and must be passed unchanged.
- Do not fabricate, guess, or use any placeholder paths (e.g. "TBD", "input/xxx.pdf", etc.).
- If the task does not clearly specify a path, immediately reply "Please provide the source document path" - do not guess.
- Do not perform geocoding.
- Do not perform map generation.
- If the tool result already contains a JSON output path, you must explicitly report it in the final reply using the key:
  - extracted_json_path
- If the tool result does not have a path but includes save info, try to organize it as extracted_json_path.
- Keep output concise; prefer structured results.
"""

GEOCODE_PROMPT = """
You are the "Geocoding Sub-Agent", responsible for assigning spatial coordinates to waypoints in airway JSON.

Your responsibilities:
1. If only a filename is given, first use find_file_by_name_tool to search for the file;
2. Call update_airway_coordinates_tool to add coordinates to place names/waypoints in the JSON;
3. Return the geocoded JSON path.

Strict requirements:
- You may only use find_file_by_name_tool and update_airway_coordinates_tool.
- Do not perform PDF parsing.
- Do not perform map generation.
- If the input is already a full path, you may process it directly without searching.
- You must explicitly report in the final reply:
  - geocoded_json_path
- geocoded_json_path must be the actual output path recorded in the summary returned by update_airway_coordinates_tool.
  Do not guess, do not append suffixes like _updated, do not infer.
- If the current task involves maritime/aids-to-navigation data, update_airway_coordinates_tool will require you to pass a gazetteer_path parameter.
  The value of this parameter must be extracted from the user's task description (the user will explicitly provide the gazetteer file path).
  Do not fabricate, guess, or use any default path. If no gazetteer path is provided in the task, ask the user first.
"""


LAYER_LOADER_PROMPT = """
You are the "Layer Loader Sub-Agent", responsible for preparing the basemap and project environment for the airway connectivity map.

Your responsibilities (must strictly follow this order, one at a time, no skipping, no reordering, no merging):
1. loader_setup_project_crs_tool                 - Set project CRS (must be EPSG:4326, prerequisite for all subsequent steps);
2. loader_reset_project_environment_tool         - Clean old layers/old layouts;
3. loader_load_state_boundary_source_tool        - Load state boundary source layer;
4. loader_build_display_state_layers_tool        - Build display state layers and target state highlight layer (produce state_display_layer / target_display_layer);
5. loader_register_basemap_layers_tool           - Register basemap layers to the project.

Strict requirements:
- You may only use the layer loader tools.
- The above 5 steps must be executed in order. Step 5 loader_register_basemap_layers_tool depends on the state_display_layer / target_display_layer produced by step 4; it is absolutely forbidden to call step 5 before step 4 has succeeded.
- If any step's returned summary contains an error or lacks expected layer_keys, you must fix the prerequisite step before proceeding; do not force through to the next step.
- Do not perform spatial analysis.
- Do not create a layout; do not add legends, titles, or scale bars.
- You must return the final job_id to the supervisor.
- After all steps are complete, you must use inspect_connectivity_job_tool to check the current summary and confirm that at least the following appear in layer_keys:
  state_source_layer / state_display_layer / target_display_layer.
- Use only virtual paths starting with /.
- Do not use Windows absolute paths (e.g. D:/... or C:/...) anywhere.
"""

SPATIAL_ANALYSIS_PROMPT = """
You are the "Spatial Analysis Sub-Agent", responsible for analyzing FAA airway JSON and generating thematic mapping layers.

Your responsibilities:
1. Read FAA airway JSON;
2. Build state centroids;
3. Compute target state connectivity metrics;
4. Print analysis summary;
5. Create flow/point/label layers;
6. Write analysis results into layers;
7. Apply thematic styles;
8. Register thematic layers;
9. Build visible_layers / refresh_layers.

Strict requirements:
- You may only use the spatial analysis tools.
- Do not create a new job.
- Do not create a layout.
- You receive an existing job_id; all operations target this job_id.
- After completion, you must use inspect_connectivity_job_tool to verify:
  analysis_keys contains rows / summary / visible_layers / refresh_layers;
  layer_keys contains flow_layer / point_layer / label_layer.
- Use only virtual paths starting with /.
- Do not use Windows absolute paths (e.g. D:/... or C:/...) anywhere.
"""

MAP_DECORATION_PROMPT = """
You are the "Map Decoration Sub-Agent", responsible for completing map layout and export.

Your responsibilities:
1. Create a layout;
2. Add title and subtitle;
3. Add the main map panel;
4. Add a scale bar;
5. Add a legend;
6. Add a north arrow;
7. Add a footer note;
8. Refresh the layout;
9. If configured, attempt to open the designer;
10. If configured, export the map.

Strict requirements:
- You may only use the map decoration tools.
- Do not perform spatial analysis.
- Do not reload the basemap.
- You receive an existing job_id; all operations target this job_id.
- After completion, you must use inspect_connectivity_job_tool to verify:
  layout_keys contains layout / map_item;
  And in the final answer, explicitly provide png_path / pdf_path / layout_name / whether auto-export was performed.
- Before publish_create_layout_tool succeeds, it is forbidden to call any other layout tools.
- Paths presented in the answer must be virtual paths, not Windows absolute paths.
"""

SUPERVISOR_PROMPT = """
You are the "Aeronautical Chart Factory Supervisor", responsible for coordinating the following 5 specialized sub-agents:

1. event_extract   - Event extraction (PDF -> JSON)
2. geocode         - Geocoding (add coordinates to waypoints in JSON)
3. layer_loader    - Layer loading
4. analysis        - Spatial analysis
5. map_publish     - Map decoration

Your overall task is not to always run the full pipeline, but to select the appropriate stages based on the user's needs.

[Decision Rules]
A. If the user provides a PDF URL or PDF virtual path and needs to extract aeronautical data:
   Call event_extract first.

B. If the user provides an existing JSON path:
   Skip event_extract.

C. If the JSON does not yet have spatial coordinates, or the task explicitly requires "geocoding / coordinate completion":
   Call geocode.

D. If the user explicitly requires "mapping / connectivity map / map output / layout / export":
   After event_extract / geocode completes, continue with:
   - create_connectivity_job_tool
   - layer_loader
   - analysis
   - map_publish

E. If the user only requires JSON extraction, or only requires geocoding:
   You may stop after the corresponding stage; do not force mapping.

[JSON Path Selection Hard Rules]
- If the user has not explicitly specified a new JSON path, when entering the mapping stage you must prioritize the geocoded_json_path returned by the geocode sub-agent.
- If geocode was not executed, use the extracted_json_path returned by event_extract.
- If neither is available, fall back to the json_path in the default config.
- Do not rename actual output paths to new virtual paths, e.g. do not rewrite /mineru_output/.../a_updated.json as /text/xxx_geocoded.json.
- Do not assume that a "suggested virtual path" is an actual existing file.
- Before calling create_connectivity_job_tool, you must call choose_final_json_path_tool to select the final JSON path.

[Analysis Strategy Parameter]
- create_connectivity_job_tool has an analysis_strategy parameter, with values "airway_connectivity" or "point_event".
- The [Data Type] section in the user task specifies which one to pass; you must pass it exactly as specified without modification.

[Mapping Stage General Principles]
- Create a job and initialize the task;
- Validate input paths and parameters;
- Use the task tool to dispatch tasks to the most appropriate sub-agent;
- The general execution order for the mapping stage is:
  Step 1: layer_loader
  Step 2: analysis
  Step 3: map_publish
- Do not skip steps, do not reverse the order.
- After each stage, read the job_id returned by the sub-agent and pass the same job_id to the next sub-agent.
- GIS operations must be performed by sub-agents using their tools.
- File paths must only use virtual paths (starting with /); do not use Windows absolute paths.

[You Must Ensure]
- If event_extract completes, try to obtain extracted_json_path;
- If geocode completes, try to obtain geocoded_json_path;
- When entering the mapping stage, the json_path used by create_connectivity_job_tool must be the "final mappable JSON path";
- If the user does not require mapping, do not perform additional mapping stages;
- If the user requires final map export but auto_export=False, include a manual export hint in the final result.

[Final Output Requirements]
Output results based on the stages actually executed:
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
   - Whether auto_export was executed
   - Key analysis findings (at least 3)
"""


# =========================
# Build sub-agents
# =========================
def build_subagents(model, profile: Optional[DomainProfile] = None):
    """
    Create specialized sub-agents.

    profile=None  -> default FAA airways
    profile=USCG_LNM_PROFILE -> maritime aids to navigation
    """
    print("Creating specialized sub-agents...")

    _profile = profile or FAA_AIRWAY_PROFILE
    _geocode_tool = make_update_coordinates_tool(_profile)

    event_extract_ai = create_specialized_agent(
        model=model,
        tools=[parse_doc_tool],
        system_prompt=EVENT_EXTRACT_PROMPT,
        name="Event Extraction"
    )

    geocode_ai = create_specialized_agent(
        model=model,
        tools=[find_file_by_name_tool, _geocode_tool],
        system_prompt=GEOCODE_PROMPT,
        name="Geocoding"
    )

    loader_ai = create_specialized_agent(
        model=model,
        tools=get_layer_loader_job_tools(),
        system_prompt=LAYER_LOADER_PROMPT,
        name="Layer Loading"
    )

    analysis_ai = create_specialized_agent(
        model=model,
        tools=get_spatial_analysis_job_tools(),
        system_prompt=SPATIAL_ANALYSIS_PROMPT,
        name="Spatial Analysis"
    )

    publish_ai = create_specialized_agent(
        model=model,
        tools=get_map_decoration_job_tools(),
        system_prompt=MAP_DECORATION_PROMPT,
        name="Map Decoration"
    )

    subagents = [
        CompiledSubAgent(
            name="event_extract",
            description="Event extraction: parse PDF / aeronautical documents, output mappable JSON",
            runnable=event_extract_ai
        ),
        CompiledSubAgent(
            name="geocode",
            description="Geocoding: add coordinates to waypoints/place names in JSON",
            runnable=geocode_ai
        ),
        CompiledSubAgent(
            name="layer_loader",
            description="Layer loading: set project CRS, load state boundaries, create basemap display layers",
            runnable=loader_ai
        ),
        CompiledSubAgent(
            name="analysis",
            description="Spatial analysis: compute connectivity, create and populate thematic layers, apply thematic styles",
            runnable=analysis_ai
        ),
        CompiledSubAgent(
            name="map_publish",
            description="Map decoration: create layout, add title/legend/scale bar/north arrow, refresh and export layout",
            runnable=publish_ai
        ),
    ]

    print("Sub-agents created")
    return subagents


# =========================
# Build supervisor agent
# =========================
def get_supervisor_tools():
    return get_deepagent_job_tools() + [choose_final_json_path_tool]

def build_supervisor_agent(model, subagents):
    print("Creating supervisor agent...")

    agent = create_deep_agent(
        model=model,
        tools=get_supervisor_tools(),
        backend=FilesystemBackend(
            root_dir=str(HOST_ROOT),
            virtual_mode=True,
        ),
        store=InMemoryStore(),
        system_prompt=SUPERVISOR_PROMPT,
        subagents=subagents,
    )

    print("Supervisor agent created")
    return agent


# =========================
# Assemble user task for supervisor
# =========================
def build_user_task_from_natural_language(
    user_request: str,
    cfg: Optional[MapConfig] = None,
    extra_requirement: str = "",
    profile=None,
    source_document: Optional[str] = None,
) -> str:
    """
    Wrap a user's natural language task into a unified supervisor task.
    """
    user_request_sanitized = sanitize_user_request_paths(user_request, cfg)

    # Source document to parse: embed directly in user_request header to ensure the supervisor does not miss it when passing the task
    if source_document:
        try:
            _src_vp = maybe_to_virtual_path(source_document)
        except Exception:
            _src_vp = source_document
        source_block = f"""
[Step 1: Call the event_extract sub-agent; parse_doc_tool file_url must be: {_src_vp}]
- This is a hard requirement; must be passed unchanged, no modification, no fabrication, no placeholders.
- If the source document is already model.json, parse_doc_tool will automatically skip PDF parsing and extract directly.
        """.strip()
    else:
        source_block = ""

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
- state_boundary_path: {maybe_to_virtual_path(cfg.paths.state_boundary_path)}
- json_path: {maybe_to_virtual_path(cfg.paths.json_path)}
- png_path: {maybe_to_virtual_path(cfg.paths.png_path)}
- pdf_path: {maybe_to_virtual_path(cfg.paths.pdf_path)}
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

    _strategy = getattr(profile, "analysis_strategy", "airway_connectivity") if profile else "airway_connectivity"
    if _strategy == "point_event":
        strategy_block = """
[Data Type]
- This data consists of independent geographic point events (each record carries its own coordinate point), not linear airways composed of multiple waypoints.
- When calling create_connectivity_job_tool, you must pass analysis_strategy="point_event".
- The analysis stage does not compute interstate connectivity; it only performs point mapping.
        """.strip()
    else:
        strategy_block = """
[Data Type]
- This data consists of airways (each record contains multiple ordered waypoints); interstate connectivity must be computed.
- When calling create_connectivity_job_tool, pass analysis_strategy="airway_connectivity" (default, can be omitted).
        """.strip()

    return f"""
You will receive a comprehensive task. This task may include only part of the pipeline, or the full pipeline:

- PDF -> JSON event extraction
- JSON -> coordinate completion / geocoding
- JSON -> connectivity map generation

You must first understand what the user actually needs done, then decide which sub-agents to invoke.

[User Requirements]
{user_request_sanitized}

{source_block}

{cfg_block}

{strategy_block}

[Execution Requirements]
1. First determine whether the input is a PDF, JSON, or just a filename;
2. First determine whether the user only wants extraction/geocoding, or also wants mapping;
3. If the input is a PDF and needs extraction, first call event_extract;
4. If the JSON needs coordinate completion, call geocode;
5. If the user needs mapping, then proceed to create_connectivity_job_tool + layer_loader + analysis + map_publish;
6. When entering the mapping stage, json_path must use the "final mappable JSON path" obtained from previous stages;
7. All paths must use virtual paths (starting with /);
8. Do not use Windows absolute paths (e.g. D:/... or C:/...) in any tool parameters or answers;
9. If a previous sub-agent produced a new JSON file path, and the user has not explicitly requested a new filename, the subsequent mapping must use that actual output path directly; otherwise the mapping stage path parameters must use the default config, and must not be rewritten to a different virtual path name.
10. Finally, output concise results based on the stages actually completed.

[Additional Requirements]
{extra_requirement if extra_requirement else "None"}
    """.strip()


# =========================
# Stream printing
# =========================
def pretty_print_stream_chunk(chunk: Dict[str, Any]) -> None:
    try:
        node_name = list(chunk.keys())[0]
        print(f"\n[{node_name}] executing...")

        node_data = chunk[node_name]

        if isinstance(node_data, dict) and "messages" in node_data:
            messages = node_data["messages"]

            if hasattr(messages, "value"):
                latest_msg = messages.value[-1] if messages.value else None
            else:
                latest_msg = messages[-1] if messages else None

            if latest_msg:
                if hasattr(latest_msg, "content") and latest_msg.content:
                    content = str(latest_msg.content).strip()
                    short_content = content[:300]
                    if "task" in short_content.lower():
                        print(f"   Calling sub-agent: {short_content}")
                    else:
                        print(f"   {short_content}")

                if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                    print("   Tool call details:")
                    for tc in latest_msg.tool_calls:
                        try:
                            tool_name = tc.get("name")
                            tool_args = tc.get("args", {})
                        except Exception:
                            tool_name = getattr(tc, "name", "unknown")
                            tool_args = getattr(tc, "args", {})
                        print(f"      - Tool name: {tool_name}")
                        print(f"        Args: {tool_args}")
        else:
            if node_data is None:
                print(f"   {node_name} has no output data")
            else:
                print(f"   {node_name} output data: {node_data}")

        print("   " + "-" * 60)

    except Exception as e:
        print(f"   Chunk parse failed: {e}")
        print(f"   Raw chunk: {chunk}")


# =========================
# Unified entry point
# =========================
def run_unified_airway_factory_multi_agent(
    user_request: str,
    cfg: Optional[MapConfig] = None,
    extra_requirement: str = "",
    stream: bool = True,
    show_progress: bool = True,
    profile: Optional[DomainProfile] = None,
    source_document: Optional[str] = None,
):
    """
    Unified multi-agent entry point:
    Handles comprehensive tasks including event extraction, geocoding, and connectivity map generation.

    profile=None  -> default FAA airways
    profile=USCG_LNM_PROFILE -> maritime aids to navigation
    """
    if cfg is not None:
        print("[DEBUG] fixed cfg json_path =", cfg.paths.json_path)
    else:
        print("[DEBUG] cfg is None")
    if profile is not None:
        print(f"[DEBUG] domain profile = {profile.name}")

    model = build_model()
    subagents = build_subagents(model, profile=profile)
    supervisor = build_supervisor_agent(model, subagents)

    user_task = build_user_task_from_natural_language(
        user_request=user_request,
        cfg=cfg,
        extra_requirement=extra_requirement,
        profile=profile,
        source_document=source_document,
    )

    print("\n[DEBUG] user_task sent to supervisor:")
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

    print("\nStarting unified multi-agent collaborative task...")
    print("=" * 80)

    if not stream:
        try:
            result = supervisor.invoke(input_data)
            print("\nMulti-agent task completed!")
            return result
        except Exception as e:
            print(f"\nExecution error: {str(e)}")
            print(f"Detailed error:\n{traceback.format_exc()}")
            raise

    collected_chunks = []

    try:
        for chunk in supervisor.stream(input_data, stream_mode="updates"):
            collected_chunks.append(chunk)
            if show_progress:
                pretty_print_stream_chunk(chunk)

        print("\nMulti-agent task completed!")
        print("View full trace: https://smith.langchain.com")
        return collected_chunks

    except Exception as e:
        print(f"\nExecution error: {str(e)}")
        print(f"Detailed error:\n{traceback.format_exc()}")
        raise