# -*- coding: utf-8 -*-
from __future__ import annotations

import uuid
import threading
from typing import Any, Dict

from domap.connectivity.shared_map_state import MapConfig, make_initial_state


_JOB_STORE: Dict[str, Dict[str, Any]] = {}
_JOB_LOCK = threading.RLock()


def create_job(cfg: MapConfig) -> str:
    """
    Create a new mapping job and return the job_id.
    """
    job_id = str(uuid.uuid4())
    state = make_initial_state(cfg)
    with _JOB_LOCK:
        _JOB_STORE[job_id] = state
    return job_id


def get_job_state(job_id: str) -> Dict[str, Any]:
    """
    Get the shared state by job_id.
    """
    with _JOB_LOCK:
        if job_id not in _JOB_STORE:
            raise KeyError(f"job_id not found: {job_id}")
        return _JOB_STORE[job_id]


def save_job_state(job_id: str, state: Dict[str, Any]) -> None:
    """
    Save the shared state.
    """
    with _JOB_LOCK:
        if job_id not in _JOB_STORE:
            raise KeyError(f"job_id not found: {job_id}")
        _JOB_STORE[job_id] = state


def delete_job(job_id: str) -> None:
    """
    Delete the job state.
    """
    with _JOB_LOCK:
        _JOB_STORE.pop(job_id, None)


def summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a lightweight summary suitable for agents, avoiding putting QGIS objects into LLM context.
    """
    cfg = state.get("cfg")
    analysis = state.get("analysis", {})
    layers = state.get("layers", {})
    layout = state.get("layout", {})

    rows = analysis.get("rows", [])
    summary = analysis.get("summary", {})

    return {
        "target_state": getattr(cfg.data, "target_state", None) if cfg else None,
        "json_path": getattr(cfg.paths, "json_path", None) if cfg else None,
        "png_path": getattr(cfg.paths, "png_path", None) if cfg else None,
        "pdf_path": getattr(cfg.paths, "pdf_path", None) if cfg else None,
        "layout_name": getattr(cfg.paths, "layout_name", None) if cfg else None,
        "layer_keys": sorted(list(layers.keys())),
        "analysis_keys": sorted(list(analysis.keys())),
        "layout_keys": sorted(list(layout.keys())),
        "connected_state_count": len(rows) if rows else 0,
        "analysis_summary": summary if summary else {},
    }


def get_job_summary(job_id: str) -> Dict[str, Any]:
    state = get_job_state(job_id)
    return summarize_state(state)
