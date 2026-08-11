# -*- coding: utf-8 -*-
"""
Three-task single/multi-agent ablation experiment (Simple / Moderate / Complex)
==================================================================
Addresses: baseline comparison, repeated runs, variance, and statistical tests.
Requirement: Each task (simple / moderate / complex) runs single-agent vs multi-agent,
10 repeats each, recording 6 metrics + observability evidence.

Three task type definitions (fully corresponding to the original user_request / user_request1 / user_request2):
    Simple    : Complete existing JSON waypoint coordinates (only goes through geocoder stage, 1 Agent)
    Moderate  : Draw a route connectivity map for New York (layer + analysis + cartography, 3 Agents)
    Complex   : From PDF -> JSON -> geocoding -> California cartography (all 5 Agents)

Usage (QGIS Python console):
    import sys
    sys.path.insert(0, r"<path_to_evaluation>/03_ablation")
    exec(open(r"<path_to_evaluation>/03_ablation/architecture/run_three_tasks.py", encoding="utf-8").read())

    # First run lightweight tasks to avoid long waits
    run_three(repeats=3, tasks=["simple"])
    run_three(repeats=3, tasks=["moderate", "simple"])

    # Full formal run
    run_three(repeats=10, tasks=["simple", "moderate", "complex"])
    run_three(repeats=10, tasks=["complex"])   # Run complex separately (~4000s each, can be overnight)

Design principles:
    - Directly call domap official unified entry points run_unified_airway_factory_single_agent /
      run_unified_airway_factory_multi_agent, ensuring consistency with the original Section 6.4 three-level task definitions.
    - Retain the complete statistical metric collection framework (steps / tool_calls / delegations / errors, etc.).
"""
from __future__ import annotations

import os
import sys
import io
import json
import time
import shutil
import traceback
import contextlib
from datetime import datetime
from typing import Any, Dict, List

# ==================================================================
# Directories and paths (set via environment variables)
# ==================================================================
_ROOT = os.environ.get("FAA2CHART_DATA_ROOT", "data")
_ABL_CODE = os.environ.get("FAA2CHART_ABL_CODE", "")
_OBS_CODE = os.environ.get("FAA2CHART_OBS_CODE", "")
_NEW_CODE_DIR = os.environ.get("FAA2CHART_NEW_CODE_DIR", "")
_NEW_OUT_DIR = os.environ.get("FAA2CHART_ABLATION_OUT", os.path.join(_ROOT, "eval_results"))

# domap original experiment paths (used for cfg and user_request paths)
_EXP_ROOT = os.environ.get("FAA2CHART_EXP_ROOT", "")
_JSON_DIR = os.environ.get("FAA2CHART_JSON_DIR", os.path.join(_EXP_ROOT, "text"))
_SHP_DIR = os.environ.get("FAA2CHART_SHP_DIR", os.path.join(_EXP_ROOT, "us_vector_data"))
_SHP_FILE = os.environ.get("FAA2CHART_SHP_FILE", os.path.join(_SHP_DIR, "ne_10m_admin_1_states_provinces.shp"))
_PDF_URL = ("https://nfdc.faa.gov/webContent/Part95/"
            "Part_95_Consolidation_February_2025.pdf/"
            "Part_95_Consolidation_February_2025.pdf")
# Reuse the existing MinerU model.json (417-page complete parsing result), skip MinerU API call
_MINERU_OUTPUT_DIR = os.environ.get("FAA2CHART_MINERU_OUTPUT", os.path.join(_EXP_ROOT, "mineru_output"))
_MODEL_JSON = os.environ.get(
    "FAA2CHART_MODEL_JSON",
    os.path.join(_MINERU_OUTPUT_DIR, "model.json"),
)
_GEOCODE_INPUT = os.environ.get("FAA2CHART_GEOCODE_INPUT", os.path.join(_JSON_DIR, "parsed_routes.json"))
_GEOCODE_OUTPUT_DIR = os.path.join(_NEW_OUT_DIR, "simple", "_geocode_output")
os.makedirs(_GEOCODE_OUTPUT_DIR, exist_ok=True)

for p in [_ABL_CODE, _OBS_CODE]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

# domap official unified entry points
from domap.connectivity.multi_agents import (  # noqa: E402
    run_unified_airway_factory_multi_agent,
    build_model, build_subagents, build_supervisor_agent,
    build_user_task_from_natural_language,
)
from domap.connectivity.single_agent import run_unified_airway_factory_single_agent  # noqa: E402
from domap.connectivity.shared_map_state import MapConfig, PathConfig  # noqa: E402
from domap.connectivity import multi_agents as MA  # noqa: E402
from domap.connectivity.virtual_paths import to_virtual_path, HOST_ROOT  # noqa: E402

# Reuse run_observability's _collect_trace for instrumentation
try:
    from run_observability import _collect_trace, _install_fault_injection
except ImportError:
    _collect_trace = None
    _install_fault_injection = None

print(f"[THREE] Main output directory -> {_NEW_OUT_DIR}")


# ==================================================================
# User request and cfg construction for the three task types
# ==================================================================
def _make_cfg(task: str) -> MapConfig:
    """Construct domap MapConfig, adjust target_state by task type."""
    cfg = MapConfig(paths=PathConfig(
        json_path=os.path.join(_JSON_DIR, "parsed_routes_newnew_4_updated.json"),
        state_boundary_path=_SHP_FILE,
        png_path=os.path.join(_EXP_ROOT, "map", "FAA_CA_Connectivity_Map.png"),
        pdf_path=os.path.join(_EXP_ROOT, "map", "faa_ca_connectivity_map.pdf"),
        layout_name="FAA_CA_Connectivity_Map",
    ))

    if task == "moderate":
        cfg.data.target_state = "NY"
        cfg.data.top_n_states = None
        cfg.data.label_top_k = 5
    elif task == "complex":
        cfg.data.target_state = "CA"
        cfg.data.top_n_states = None
        cfg.data.label_top_k = 5
    else:  # trivial
        cfg.data.target_state = "CA"
        cfg.data.top_n_states = None
        cfg.data.label_top_k = 10

    cfg.run.open_designer = False
    cfg.run.auto_export = False
    cfg.run.export_dpi = 300
    return cfg


def _make_user_request(task: str) -> str:
    """Fully corresponds to the original three requests.

    Paths use the original format, internally converted to virtual paths
    by build_user_task_from_natural_language via sanitize_user_request_paths.
    """
    if task == "trivial":
        # sanitize_user_request_paths only replaces cfg paths, not _GEOCODE_INPUT.
        # Must manually convert to virtual path, otherwise supervisor cannot recognize it.
        geocode_input_vp = to_virtual_path(_GEOCODE_INPUT)
        return (
            f"Please check and complete the spatial coordinates of waypoints in the file \"{geocode_input_vp}\".\n"
            f"Coordinates can only be found by matching from the CSV table. Waypoints not in the CSV "
            f"(mostly non-US waypoints) should be skipped directly. Do not attempt other methods.\n"
        )
    if task == "moderate":
        return (
            "Please use the processed JSON file path:\n"
            f"{os.path.join(_JSON_DIR, 'parsed_routes_newnew_4_updated.json')}\n\n"
            "1. Draw a route connectivity map centered on the target state for New York;\n"
            "2. Annotate the top 5 most connected states;\n"
            "3. Use the default cartography paths.\n"
        )
    # complex (full PDF -> map, model.json skips MinerU)
    model_json_vp = to_virtual_path(_MODEL_JSON)
    return f"""
Please process this FAA aeronautical report's model.json file (MinerU PDF parsing has been completed, containing 417 pages of full content):
`{model_json_vp}`

Requirements:
1. First convert model.json to structured JSON (event_extract sub-agent, parse_doc_tool will automatically skip MinerU and extract directly);
2. Perform spatial geocoding for the waypoints inside (geocode sub-agent);
3. Then draw a route connectivity map centered on the target state for California;
4. Preserve the migration-map-style arc representation, annotate the top 5 most connected states.
"""


# ==================================================================
# Metric collection
# ==================================================================
_ERROR_MARKERS = ("Error", "error", "Traceback", "Exception",
                  "not found", "failed", "Failed")


def _is_error_content(status: Any, content: str) -> bool:
    if status == "error":
        return True
    return any(m in content for m in _ERROR_MARKERS)


def _extract_metrics_from_chunks(chunks: List[Dict]) -> Dict[str, Any]:
    """Extract metrics from stream chunks (fallback logic, consistent with old version)."""
    tool_calls = 0
    deleg = 0
    tool_errs = 0
    ai_n = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for _node, data in chunk.items():
            if not isinstance(data, dict):
                continue
            msgs = (getattr(data.get("messages", None), "value", None)
                    or data.get("messages") or [])
            for m in msgs:
                mtype = getattr(m, "type", "")
                if getattr(m, "tool_calls", None):
                    for tc in getattr(m, "tool_calls"):
                        tn = (getattr(tc, "name", "")
                              if not isinstance(tc, dict) else tc.get("name", ""))
                        tool_calls += 1
                        if tn == "task":
                            deleg += 1
                if mtype == "ai":
                    ai_n += 1
                if mtype == "tool":
                    content = str(getattr(m, "content", ""))
                    status = getattr(m, "status", None)
                    if _is_error_content(status, content):
                        tool_errs += 1
    return {
        "total_steps": len(chunks),
        "tool_calls": tool_calls,
        "subagent_delegations": deleg,
        "tool_errors": tool_errs,
        "ai_messages": ai_n,
        "produced_layout": None,
        "produced_analysis": None,
        "observable_points": tool_calls + ai_n,
        "n_errors_detected": tool_errs,
        "recovery_rate": None,
        "call_result_pairing_rate": None,
    }


# ==================================================================
# Single run
# ==================================================================

class _SubagentCollector:
    """Wrap sub-agent runnable, collect internal observable_points and other metrics.

    stream_mode="updates" only captures supervisor node output; sub-agent internal
    tool calls and AI messages are invisible. This collector intercepts invoke() calls,
    counts messages generated inside sub-agents, solving the problem of underestimated
    multi-agent observable_points.
    """

    def __init__(self, runnable, name: str = ""):
        self._runnable = runnable
        self._name = name
        self.tool_calls = 0
        self.ai_messages = 0
        self.tool_errors = 0
        self.invoke_count = 0

    def invoke(self, input_data, config=None, **kwargs):
        self.invoke_count += 1
        result = self._runnable.invoke(input_data, config=config, **kwargs)
        messages = result.get("messages", []) if isinstance(result, dict) else []
        for msg in messages:
            mtype = getattr(msg, "type", "")
            if mtype == "ai":
                self.ai_messages += 1
                tcs = getattr(msg, "tool_calls", None) or []
                self.tool_calls += len(tcs)
            elif mtype == "tool":
                content = str(getattr(msg, "content", ""))
                if "Error" in content or "error" in content.lower():
                    self.tool_errors += 1
        return result

    @property
    def observable_points(self):
        return self.tool_calls + self.ai_messages

    def __getattr__(self, name):
        # Forward undefined attributes to the original runnable
        return getattr(self._runnable, name)


def _run_multi_with_subagent_trace(
    user_request: str, cfg, extra_requirement: str = ""
) -> tuple:
    """Build supervisor + wrap sub-agents, run and return (chunks, collectors)."""
    model = build_model()
    subagents = build_subagents(model)

    # Wrap each sub-agent's runnable
    collectors: Dict[str, _SubagentCollector] = {}
    for sa in subagents:
        name = sa["name"]
        col = _SubagentCollector(sa["runnable"], name=name)
        sa["runnable"] = col
        collectors[name] = col

    supervisor = build_supervisor_agent(model, subagents)

    user_task = build_user_task_from_natural_language(
        user_request=user_request,
        cfg=cfg,
        extra_requirement=extra_requirement,
    )

    input_data = {"messages": [{"role": "user", "content": user_task}]}
    collected_chunks = []
    for chunk in supervisor.stream(input_data, stream_mode="updates"):
        collected_chunks.append(chunk)

    return collected_chunks, collectors


def run_one(task: str, agent: str, run_idx: int, fault: bool = False
            ) -> Dict[str, Any]:
    """
    Single run.
        task:  trivial / moderate / complex
        agent: multi / single
        run_idx: 1-based
        fault:  whether to enable controlled fault injection
    """
    from qgis.core import QgsProject
    QgsProject.instance().layoutManager().clear()
    QgsProject.instance().removeAllMapLayers()

    # Install fault injection (if enabled)
    if fault and _install_fault_injection is not None:
        restore = _install_fault_injection()
    else:
        restore = lambda: None  # noqa: E731

    # ---- trivial task: backup original file (update_airway_coordinates_tool will overwrite) ----
    _trivial_backup = None
    if task == "trivial":
        _trivial_backup = _GEOCODE_INPUT + ".bak"
        shutil.copy2(_GEOCODE_INPUT, _trivial_backup)

    try:
        user_req = _make_user_request(task)
        cfg = _make_cfg(task)

        # ---- Call domap unified entry point, suppress internal print ----
        t0 = time.time()
        chunks: List[Dict] = []
        stamps: List[float] = []
        last_err: str = ""
        success_flag = False

        try:
            devnull = io.StringIO()
            subagent_collectors = {}   # Initialize early to avoid reference to undefined in except path
            with contextlib.redirect_stdout(devnull):
                if agent == "multi":
                    # Simple task: explicitly tell supervisor to delegate to geocode, avoid
                    # supervisor using FilesystemBackend to operate files itself (30 LLM rounds, 306s)
                    if task == "trivial":
                        extra = "This task is purely coordinate completion. Please directly delegate to the geocode sub-agent. Do not operate files yourself."
                    else:
                        extra = ""
                    chunks, subagent_collectors = \
                        _run_multi_with_subagent_trace(
                            user_request=user_req, cfg=cfg,
                            extra_requirement=extra,
                        )
                else:
                    chunks = run_unified_airway_factory_single_agent(
                        user_request=user_req,
                        cfg=cfg,
                        stream=True,
                        show_progress=False,
                    )
                    subagent_collectors = {}
            stamps = [t0 + (i + 1) * 0.1 for i in range(len(chunks))]
            success_flag = True
            # Check for errors in chunks
            for chunk in chunks:
                if isinstance(chunk, dict):
                    for _node, data in chunk.items():
                        if isinstance(data, dict):
                            msgs = (getattr(data.get("messages", None), "value", None)
                                    or data.get("messages") or [])
                            for m in msgs:
                                content = str(getattr(m, "content", ""))
                                if _is_error_content("", content):
                                    if not last_err:
                                        last_err = content[:200]
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            if ("KeyError" in tb or "StateNotFound" in tb
                    or "NoneType" in tb or "Recursion" in tb):
                pass
            else:
                last_err += "\n" + tb[:1000]
            success_flag = False
            stamps = []

        elapsed = round(time.time() - t0, 2)

        # ---- Metric calculation ----
        if _collect_trace is not None and chunks:
            try:
                result = _collect_trace(chunks, stamps, t0)
                metrics = result["metrics"]
                events = result["events"]
                obs_ok = True
            except Exception:
                obs_ok = False
                metrics = _extract_metrics_from_chunks(chunks)
                events = []
        else:
            metrics = _extract_metrics_from_chunks(chunks)
            events = []
            obs_ok = False

        # ---- Merge sub-agent internal metrics (multi-agent mode) ----
        subagent_tool_calls = 0
        subagent_ai_msgs = 0
        subagent_tool_errs = 0
        subagent_obs = 0
        for name, col in subagent_collectors.items():
            subagent_tool_calls += col.tool_calls
            subagent_ai_msgs += col.ai_messages
            subagent_tool_errs += col.tool_errors
            subagent_obs += col.observable_points

        total_tool_calls = int(metrics["tool_calls"]) + subagent_tool_calls
        total_ai_msgs = int(metrics.get("ai_messages", 0)) + subagent_ai_msgs
        total_obs = int(metrics.get("observable_points", 0)) + subagent_obs
        total_tool_errs = float(metrics["tool_errors"]) + subagent_tool_errs

        record = {
            "task": task,
            "agent": agent,
            "run_idx": run_idx,
            "start_time": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            "elapsed_sec": elapsed,
            "success": success_flag,
            "error": last_err,
            "fault_injection_enabled": fault,
            "metrics": {
                "total_steps": int(metrics["total_steps"]),
                "tool_calls": total_tool_calls,
                "subagent_delegations": int(metrics["subagent_delegations"]),
                "tool_errors": total_tool_errs,
                "ai_messages": total_ai_msgs,
                "observable_points": total_obs,
                # Retain supervisor-level raw values for comparison analysis
                "supervisor_tool_calls": int(metrics["tool_calls"]),
                "supervisor_ai_messages": int(metrics.get("ai_messages", 0)),
                "supervisor_observable_points": int(metrics.get("observable_points", 0)),
                "subagent_tool_calls": subagent_tool_calls,
                "subagent_ai_messages": subagent_ai_msgs,
                "subagent_observable_points": subagent_obs,
                # Sub-agent details
                "subagent_details": {
                    name: {
                        "tool_calls": col.tool_calls,
                        "ai_messages": col.ai_messages,
                        "observable_points": col.observable_points,
                        "invoke_count": col.invoke_count,
                    }
                    for name, col in subagent_collectors.items()
                },
                "n_errors_detected": int(metrics.get("n_errors_detected", 0)),
                "first_error_step": metrics.get("first_error_step"),
                "error_localization_rate": metrics.get("error_localization_rate"),
                "recovery_rate": metrics.get("recovery_rate"),
                "mean_recovery_steps": metrics.get("mean_recovery_steps"),
                "mean_recovery_sec": metrics.get("mean_recovery_sec"),
                "call_result_pairing_rate": metrics.get("call_result_pairing_rate"),
            },
            "obs_trace_collected": obs_ok,
            "task_description": {
                "trivial": "Geocode JSON waypoints only",
                "moderate": "NY connectivity map (3 agents, no PDF)",
                "complex": "Full PDF->JSON->geocode->CA map (all 5 agents)",
            }[task],
        }

        # events stored separately
        events_file = None
        if events:
            events_dir = os.path.join(_NEW_OUT_DIR, task, agent, "_events")
            os.makedirs(events_dir, exist_ok=True)
            events_file = os.path.join(events_dir, f"events_run_{run_idx}.json")
            with open(events_file, "w", encoding="utf-8") as f:
                json.dump(events, f, ensure_ascii=False, indent=2)
        record["events_file"] = events_file
        return record

    finally:
        restore()
        # ---- trivial task: restore original file ----
        if _trivial_backup is not None:
            try:
                if os.path.exists(_trivial_backup):
                    shutil.move(_trivial_backup, _GEOCODE_INPUT)
            except Exception as e:
                print(f"WARNING [trivial] File restore exception: {e}")


# ==================================================================
# Main entry
# ==================================================================
def _existing_count(task: str, agent: str) -> int:
    folder = os.path.join(_NEW_OUT_DIR, task, agent)
    os.makedirs(folder, exist_ok=True)
    return len([f for f in os.listdir(folder)
                if f.startswith("run_") and f.endswith(".json")])


def _append_index(record: Dict):
    idx_path = os.path.join(_NEW_OUT_DIR, "index.jsonl")
    with open(idx_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_three(repeats: int = 10,
              tasks: List[str] | None = None,
              variants: List[str] | None = None,
              fault: bool = False,
              ):
    """
    repeats : Number of repetitions per (task x variant) combination
    tasks   : ["trivial","moderate","complex"], None=all
    variants: ["multi","single"], None=all
    fault   : Whether to enable controlled fault injection
    """
    tasks = tasks or ["trivial", "moderate", "complex"]
    variants = variants or ["multi", "single"]

    for t in tasks:
        for a in variants:
            existing = _existing_count(t, a)
            start = existing + 1
            end = existing + repeats
            print(f"\n===== {t} / {a}: Running from #{start} to #{end} (existing #{existing}) =====")
            for idx in range(start, end + 1):
                print(f"\n  >> {t} {a} #{idx}/{end}")
                try:
                    rec = run_one(t, a, idx, fault=fault)
                except Exception as e:
                    rec = {
                        "task": t, "agent": a, "run_idx": idx,
                        "success": False, "error": f"OuterException: {e}",
                        "elapsed_sec": -1, "metrics": {},
                    }
                folder = os.path.join(_NEW_OUT_DIR, t, a)
                out_path = os.path.join(folder, f"run_{idx}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                _append_index(rec)
                status = "OK" if rec.get("success") else "FAIL"
                err_info = ""
                if not rec.get("success"):
                    err_info = f" | err={str(rec.get('error',''))[:80]}"
                m = rec.get("metrics", {})
                print(f"   -> {status} | steps={m.get('total_steps')} "
                      f"tools={m.get('tool_calls')} deleg={m.get('subagent_delegations')} "
                      f"errors={m.get('tool_errors')} "
                      f"obs={m.get('observable_points')}"
                      f"(sup={m.get('supervisor_observable_points','')}"
                      f" sub={m.get('subagent_observable_points','')}) "
                      f"elapsed={rec.get('elapsed_sec')}s"
                      f"{err_info}")

    print(f"\n[Done] Results directory {_NEW_OUT_DIR}")
    print(f"[Done] Summary index {os.path.join(_NEW_OUT_DIR, 'index.jsonl')}")
    print("[Next step] Run analyze_three_tasks.py to generate statistics tables.")


if __name__ == "__main__":
    print("Please use run_three() in the QGIS Python console")