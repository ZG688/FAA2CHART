# Wrapping bottom-level stateful functions into job_id-aware tools.
#
# Sub-agents should not directly call state -> state wrappers;
# instead, add a layer of job-aware tools.
#
# This file turns the original three categories of bottom-level functions
# into job_id -> summary tools.

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from langchain_core.tools import tool

from domap.connectivity.virtual_paths import (
    to_host_path,
    virtualize_obj,
)
from domap.connectivity.shared_map_state import MapConfig, PathConfig, LayerNameConfig
from domap.connectivity.runtime_job_store import (
    create_job,
    get_job_state,
    save_job_state,
    get_job_summary,
)
# ========================
# Generic tool wrapping helpers
# ========================
from domap.connectivity.layer_loader_tools import (
    validate_inputs,
    setup_project_crs,
    reset_project_environment,
    load_state_boundary_source,
    build_display_state_layers,
    register_basemap_layers,
)

from domap.connectivity.spatial_analysis_tools import (
    load_routes_json,
    build_state_centroids,
    compute_connectivity_metrics,
    print_analysis_summary,
    create_connectivity_layers,
    populate_connectivity_layers,
    apply_thematic_styles,
    register_analysis_layers,
    build_visible_layer_order,
)

from domap.connectivity.map_decoration_tools import (
    create_layout,
    add_layout_title,
    add_map_panel,
    add_scale_bar,
    add_legend,
    add_simple_north_arrow,
    add_footer_note,
    force_refresh_layout,
    open_layout_designer_safe,
    export_layout_if_needed,
    export_layout_manual,
)


def _resolve_analysis_step(state: Dict[str, Any], step_name: str, default_fn: Callable):
    """
    Select the analysis step implementation based on analysis_strategy in state.

    When undeclared or declared as airway_connectivity, return the FAA default implementation,
    keeping existing behavior completely unchanged.
    """
    strategy = state.get("analysis_strategy") or "airway_connectivity"
    if strategy == "airway_connectivity":
        return default_fn

    if strategy == "point_event":
        from domap.connectivity import point_event_analysis_tools as _pe
        fn = getattr(_pe, step_name, None)
        if fn is None:
            raise RuntimeError(
                f"point_event strategy does not implement step {step_name}"
            )
        return fn

    raise RuntimeError(f"Unknown analysis_strategy: {strategy}")


def _run_analysis_step(job_id: str, default_fn: Callable, step_name: str) -> Dict[str, Any]:
    """Analysis step entry: resolve implementation by strategy, then execute."""
    state = get_job_state(job_id)
    fn = _resolve_analysis_step(state, step_name, default_fn)
    new_state = fn(state)
    save_job_state(job_id, new_state)
    summary = get_job_summary(job_id)

    return {
        "job_id": job_id,
        "step": step_name,
        "ok": True,
        "summary": virtualize_obj(summary),
    }


def _run_stateful_step(
    job_id: str,
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    step_name: str
) -> Dict[str, Any]:
    """
    Retrieve state by job_id, execute bottom-level function, then write state back.
    """
    state = get_job_state(job_id)
    new_state = fn(state)
    save_job_state(job_id, new_state)
    summary = get_job_summary(job_id)

    return {
        "job_id": job_id,
        "step": step_name,
        "ok": True,
        "summary": virtualize_obj(summary),
    }


# ========================
# Job creation and status inspection tools
# ========================
@tool("create_connectivity_job_tool", parse_docstring=True)
def create_connectivity_job_tool(
    state_boundary_path: str,
    json_path: str,
    png_path: str,
    pdf_path: str,
    layout_name: str = "FAA_CA_Connectivity_Map",
    target_state: str = "CA",
    top_n_states: Optional[int] = None,
    label_top_k: int = 10,
    open_designer: bool = True,
    auto_export: bool = False,
    export_dpi: int = 300,
    analysis_strategy: str = "airway_connectivity",
    layer_point_name: str = "FAA_Connectivity_Points",
    layer_flow_name: str = "FAA_Connectivity_Flows",
    layer_label_name: str = "FAA_Connectivity_Labels",
    layer_state_source_name: str = "US_States_Source",
    layer_state_display_name: str = "State_Display",
    layer_target_display_name: str = "Target_State_Display",
) -> Dict[str, Any]:
    """
    Create a new airway connectivity mapping job and return the job_id.

    Args:
        state_boundary_path: Virtual path to state boundary vector, e.g. /US_vector_data/.../xxx.shp
        json_path: Virtual path to FAA JSON, e.g. /text/xxx.json
        png_path: Virtual path for PNG output, e.g. /map/output.png
        pdf_path: Virtual path for PDF output, e.g. /map/output.pdf
        layout_name: Layout name.
        target_state: Target state abbreviation, e.g. CA.
        top_n_states: Only keep top N states; None means all.
        label_top_k: Label top K states.
        open_designer: Whether to open QGIS Designer.
        auto_export: Whether to auto-export.
        export_dpi: Export DPI.
        analysis_strategy: Spatial analysis strategy. "airway_connectivity" (default)
            or "point_event". Determined by domain configuration; for point event
            data (e.g., aids-to-navigation, disaster points) pass "point_event".
        layer_point_name: Point layer name.
        layer_flow_name: Flow layer name.
        layer_label_name: Label layer name.
        layer_state_source_name: State boundary source layer name.
        layer_state_display_name: State display layer name.
        layer_target_display_name: Target state layer name.

    Returns:
        Dict containing job_id and task summary.
    """
    print("\n[DEBUG] create_connectivity_job_tool called")
    print("  virtual state_boundary_path =", state_boundary_path)
    print("  virtual json_path           =", json_path)
    print("  virtual png_path            =", png_path)
    print("  virtual pdf_path            =", pdf_path)

    # Critical: virtual path -> host real path
    state_boundary_host = to_host_path(state_boundary_path)
    json_host = to_host_path(json_path)
    png_host = to_host_path(png_path)
    pdf_host = to_host_path(pdf_path)

    print("  host state_boundary_path    =", state_boundary_host)
    print("  host json_path              =", json_host)
    print("  host png_path               =", png_host)
    print("  host pdf_path               =", pdf_host)

    cfg = MapConfig(
        paths=PathConfig(
            state_boundary_path=state_boundary_host,
            json_path=json_host,
            png_path=png_host,
            pdf_path=pdf_host,
            layout_name=layout_name,
        ),
        layers=LayerNameConfig(
            point=layer_point_name,
            flow=layer_flow_name,
            label=layer_label_name,
            state_source=layer_state_source_name,
            state_display=layer_state_display_name,
            target_display=layer_target_display_name,
        ),
    )
    cfg.data.target_state = target_state
    cfg.data.top_n_states = top_n_states
    cfg.data.label_top_k = label_top_k
    cfg.run.open_designer = open_designer
    cfg.run.auto_export = auto_export
    cfg.run.export_dpi = export_dpi

    job_id = create_job(cfg)

    # Record analysis strategy for _run_analysis_step dispatch
    _state = get_job_state(job_id)
    _state["analysis_strategy"] = analysis_strategy
    save_job_state(job_id, _state)

    summary = get_job_summary(job_id)

    # Re-virtualize when returning to LLM to avoid D:/... leaking into context
    return {
        "job_id": job_id,
        "ok": True,
        "summary": virtualize_obj(summary),
    }



@tool("inspect_connectivity_job_tool", parse_docstring=True)
def inspect_connectivity_job_tool(job_id: str) -> Dict[str, Any]:
    """
    View a lightweight summary of the current mapping task.

    Args:
        job_id: Mapping task ID.

    Returns:
        Current task status summary.
    """
    summary = get_job_summary(job_id)

    # Critical: real path -> virtual path
    return {
        "job_id": job_id,
        "ok": True,
        "summary": virtualize_obj(summary),
    }


# ========================
# Layer loader sub-agent tools
# ========================
@tool("loader_validate_inputs_tool", parse_docstring=True)
def loader_validate_inputs_tool(job_id: str) -> Dict[str, Any]:
    """
    Validate input parameters and critical file paths for the current task.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, validate_inputs, "validate_inputs")


@tool("loader_setup_project_crs_tool", parse_docstring=True)
def loader_setup_project_crs_tool(job_id: str) -> Dict[str, Any]:
    """
    Initialize the QGIS project CRS.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, setup_project_crs, "setup_project_crs")


@tool("loader_reset_project_environment_tool", parse_docstring=True)
def loader_reset_project_environment_tool(job_id: str) -> Dict[str, Any]:
    """
    Clean old layers and old layouts.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, reset_project_environment, "reset_project_environment")


@tool("loader_load_state_boundary_source_tool", parse_docstring=True)
def loader_load_state_boundary_source_tool(job_id: str) -> Dict[str, Any]:
    """
    Load the state boundary source layer and auto-detect fields.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, load_state_boundary_source, "load_state_boundary_source")


@tool("loader_build_display_state_layers_tool", parse_docstring=True)
def loader_build_display_state_layers_tool(job_id: str) -> Dict[str, Any]:
    """
    Build display state boundary layers and target state highlight layer.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, build_display_state_layers, "build_display_state_layers")


@tool("loader_register_basemap_layers_tool", parse_docstring=True)
def loader_register_basemap_layers_tool(job_id: str) -> Dict[str, Any]:
    """
    Register basemap layers to the QGIS project.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, register_basemap_layers, "register_basemap_layers")

# ========================
# Spatial analysis sub-agent tools
# ========================
@tool("analysis_load_routes_json_tool", parse_docstring=True)
def analysis_load_routes_json_tool(job_id: str) -> Dict[str, Any]:
    """
    Read FAA airway JSON.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_analysis_step(job_id, load_routes_json, "load_routes_json")


@tool("analysis_build_state_centroids_tool", parse_docstring=True)
def analysis_build_state_centroids_tool(job_id: str) -> Dict[str, Any]:
    """
    Build state centroids based on state boundary display layer.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_analysis_step(job_id, build_state_centroids, "build_state_centroids")


@tool("analysis_compute_connectivity_metrics_tool", parse_docstring=True)
def analysis_compute_connectivity_metrics_tool(job_id: str) -> Dict[str, Any]:
    """
    Compute airway connectivity metrics for the target state.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_analysis_step(job_id, compute_connectivity_metrics, "compute_connectivity_metrics")


@tool("analysis_print_analysis_summary_tool", parse_docstring=True)
def analysis_print_analysis_summary_tool(job_id: str) -> Dict[str, Any]:
    """
    Print analysis summary for logging and debugging.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_analysis_step(job_id, print_analysis_summary, "print_analysis_summary")


@tool("analysis_create_connectivity_layers_tool", parse_docstring=True)
def analysis_create_connectivity_layers_tool(job_id: str) -> Dict[str, Any]:
    """
    Create flow/point/label thematic layers.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_analysis_step(job_id, create_connectivity_layers, "create_connectivity_layers")


@tool("analysis_populate_connectivity_layers_tool", parse_docstring=True)
def analysis_populate_connectivity_layers_tool(job_id: str) -> Dict[str, Any]:
    """
    Write analysis results into flow/point/label layers.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_analysis_step(job_id, populate_connectivity_layers, "populate_connectivity_layers")


@tool("analysis_apply_thematic_styles_tool", parse_docstring=True)
def analysis_apply_thematic_styles_tool(job_id: str) -> Dict[str, Any]:
    """
    Apply styles to basemap and thematic layers.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_analysis_step(job_id, apply_thematic_styles, "apply_thematic_styles")


@tool("analysis_register_analysis_layers_tool", parse_docstring=True)
def analysis_register_analysis_layers_tool(job_id: str) -> Dict[str, Any]:
    """
    Add thematic layers to the QGIS project.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_analysis_step(job_id, register_analysis_layers, "register_analysis_layers")


@tool("analysis_build_visible_layer_order_tool", parse_docstring=True)
def analysis_build_visible_layer_order_tool(job_id: str) -> Dict[str, Any]:
    """
    Build visible layer order and refresh layer order.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_analysis_step(job_id, build_visible_layer_order, "build_visible_layer_order")

# ========================
# Map decoration sub-agent tools
# ========================

@tool("publish_create_layout_tool", parse_docstring=True)
def publish_create_layout_tool(job_id: str) -> Dict[str, Any]:
    """
    Create a layout object.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, create_layout, "create_layout")


@tool("publish_add_layout_title_tool", parse_docstring=True)
def publish_add_layout_title_tool(job_id: str) -> Dict[str, Any]:
    """
    Add title and subtitle.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, add_layout_title, "add_layout_title")


@tool("publish_add_map_panel_tool", parse_docstring=True)
def publish_add_map_panel_tool(job_id: str) -> Dict[str, Any]:
    """
    Add the main map panel.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, add_map_panel, "add_map_panel")


@tool("publish_add_scale_bar_tool", parse_docstring=True)
def publish_add_scale_bar_tool(job_id: str) -> Dict[str, Any]:
    """
    Add a scale bar.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, add_scale_bar, "add_scale_bar")


@tool("publish_add_legend_tool", parse_docstring=True)
def publish_add_legend_tool(job_id: str) -> Dict[str, Any]:
    """
    Add a legend.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, add_legend, "add_legend")


@tool("publish_add_simple_north_arrow_tool", parse_docstring=True)
def publish_add_simple_north_arrow_tool(job_id: str) -> Dict[str, Any]:
    """
    Add a simple north arrow.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, add_simple_north_arrow, "add_simple_north_arrow")


@tool("publish_add_footer_note_tool", parse_docstring=True)
def publish_add_footer_note_tool(job_id: str) -> Dict[str, Any]:
    """
    Add footer note text.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, add_footer_note, "add_footer_note")


@tool("publish_force_refresh_layout_tool", parse_docstring=True)
def publish_force_refresh_layout_tool(job_id: str) -> Dict[str, Any]:
    """
    Refresh layout and map item display.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, force_refresh_layout, "force_refresh_layout")


@tool("publish_open_layout_designer_safe_tool", parse_docstring=True)
def publish_open_layout_designer_safe_tool(job_id: str) -> Dict[str, Any]:
    """
    Attempt to open QGIS Layout Designer based on configuration.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, open_layout_designer_safe, "open_layout_designer_safe")


@tool("publish_export_layout_if_needed_tool", parse_docstring=True)
def publish_export_layout_if_needed_tool(job_id: str) -> Dict[str, Any]:
    """
    Decide whether to auto-export PNG/PDF based on configuration.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, export_layout_if_needed, "export_layout_if_needed")


@tool("publish_export_layout_manual_tool", parse_docstring=True)
def publish_export_layout_manual_tool(job_id: str) -> Dict[str, Any]:
    """
    Manually export the current layout to PNG/PDF.

    Args:
        job_id: Mapping task ID.

    Returns:
        Task summary after execution.
    """
    return _run_stateful_step(job_id, export_layout_manual, "export_layout_manual")

# ========================
# Tool list distribution functions
# ========================
def get_deepagent_job_tools() -> List:
    """Return the list of job-aware tools available to the main coordinator agent."""
    return [
        create_connectivity_job_tool,
        inspect_connectivity_job_tool,
    ]

        
def get_layer_loader_job_tools() -> List:
    """Return the list of tools available to the layer loader sub-agent."""
    return [
        inspect_connectivity_job_tool,
        loader_validate_inputs_tool,
        loader_setup_project_crs_tool,
        loader_reset_project_environment_tool,
        loader_load_state_boundary_source_tool,
        loader_build_display_state_layers_tool,
        loader_register_basemap_layers_tool,
    ]


def get_spatial_analysis_job_tools() -> List:
    """Return the list of tools available to the spatial analysis sub-agent."""
    return [
        inspect_connectivity_job_tool,
        analysis_load_routes_json_tool,
        analysis_build_state_centroids_tool,
        analysis_compute_connectivity_metrics_tool,
        analysis_print_analysis_summary_tool,
        analysis_create_connectivity_layers_tool,
        analysis_populate_connectivity_layers_tool,
        analysis_apply_thematic_styles_tool,
        analysis_register_analysis_layers_tool,
        analysis_build_visible_layer_order_tool,
    ]


def get_map_decoration_job_tools() -> List:
    """Return the list of tools available to the map decoration sub-agent."""
    return [
        inspect_connectivity_job_tool,
        publish_create_layout_tool,
        publish_add_layout_title_tool,
        publish_add_map_panel_tool,
        publish_add_scale_bar_tool,
        publish_add_legend_tool,
        publish_add_simple_north_arrow_tool,
        publish_add_footer_note_tool,
        publish_force_refresh_layout_tool,
        publish_open_layout_designer_safe_tool,
        publish_export_layout_if_needed_tool,
        publish_export_layout_manual_tool,
    ]