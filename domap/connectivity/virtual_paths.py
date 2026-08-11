# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

# Root directory for the "airway report" project. Replace with actual path.
HOST_ROOT = Path(
    "PATH_TO_HOST_ROOT"  # <-- Replace with actual root directory
).resolve()


def _normalize_slashes(s: str) -> str:
    return str(s).replace("\\", "/").strip()


def is_windows_abs_path(path_str: str) -> bool:
    s = _normalize_slashes(path_str)
    return bool(re.match(r"^[A-Za-z]:/", s))


def is_virtual_path(path_str: str) -> bool:
    s = _normalize_slashes(path_str)
    return s.startswith("/")


def _ensure_under_host_root(real_path: Path, raw_path: str) -> Path:
    """
    Ensure the real path is inside HOST_ROOT to prevent traversal.
    """
    real_path = real_path.resolve()

    if real_path == HOST_ROOT:
        return real_path

    if HOST_ROOT not in real_path.parents:
        raise ValueError(
            f"Path traversal: access outside HOST_ROOT is not allowed: {raw_path}\n"
            f"HOST_ROOT={HOST_ROOT}\n"
            f"resolved={real_path}"
        )
    return real_path


def to_host_path(path_str: str) -> str:
    """
    Convert a virtual path /xxx/yyy or a Windows absolute path inside HOST_ROOT
    to a real host path.

    Notes:
    - LLM / Agent should preferably pass only virtual paths;
    - For compatibility with manual debugging, Windows absolute paths inside HOST_ROOT are also accepted.
    """
    s = _normalize_slashes(path_str)

    if not s:
        raise ValueError("Empty path is invalid")

    # Compatible with manual debugging where Windows absolute paths are directly passed
    if is_windows_abs_path(s):
        real = _ensure_under_host_root(Path(s), s)
        return str(real)

    # Agent standard input: virtual path
    if is_virtual_path(s):
        real = _ensure_under_host_root(HOST_ROOT / s.lstrip("/"), s)
        return str(real)

    raise ValueError(
        f"Unsupported path format: {path_str}\n"
        f"Please use a virtual absolute path (e.g., /text/a.json)"
    )


def to_virtual_path(host_path: str) -> str:
    """
    Convert a host real path to a virtual path /xxx/yyy.
    """
    s = _normalize_slashes(host_path)

    if not s:
        raise ValueError("Empty path is invalid")

    # If it's already a virtual path, return directly
    if is_virtual_path(s) and not is_windows_abs_path(s):
        return s

    real = _ensure_under_host_root(Path(s), s)
    rel = real.relative_to(HOST_ROOT)
    return "/" + rel.as_posix()


def maybe_to_virtual_path(value: str) -> str:
    """
    Attempt to convert a string to a virtual path; if it's not a path or conversion fails, return as-is.
    """
    if not isinstance(value, str):
        return value

    s = _normalize_slashes(value)

    # Only attempt conversion for Windows absolute paths to avoid false positives
    if is_windows_abs_path(s):
        try:
            return to_virtual_path(s)
        except Exception:
            return value

    return value


def virtualize_obj(obj: Any) -> Any:
    """
    Recursively convert Windows absolute paths in an object to virtual paths.
    """
    if isinstance(obj, dict):
        return {k: virtualize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [virtualize_obj(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(virtualize_obj(x) for x in obj)
    if isinstance(obj, str):
        return maybe_to_virtual_path(obj)
    return obj


def sanitize_user_request_paths(user_request: str, cfg: Optional[Any] = None) -> str:
    """
    Replace known host paths in user_request with virtual paths,
    to avoid exposing D:/... to the LLM.

    Currently only does reliable replacement for known paths in cfg.
    """
    if not user_request:
        return user_request

    text = user_request

    if cfg is None:
        return text

    candidates = [
        getattr(cfg.paths, "state_boundary_path", None),
        getattr(cfg.paths, "json_path", None),
        getattr(cfg.paths, "png_path", None),
        getattr(cfg.paths, "pdf_path", None),
    ]

    for p in candidates:
        if not p:
            continue
        try:
            vp = to_virtual_path(str(p))
            raw1 = str(p)
            raw2 = _normalize_slashes(str(p))
            text = text.replace(raw1, vp)
            text = text.replace(raw2, vp)
        except Exception:
            # If the path is not inside HOST_ROOT, don't replace
            pass

    return text
