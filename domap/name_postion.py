# -*- coding: utf-8 -*-
import pandas as pd
import json
import os
from functools import partial
from langchain.tools import tool
from domap.connectivity.virtual_paths import to_host_path, to_virtual_path
from typing import Any, Optional

# ---- FAA gazetteer data ----
# Paths to FAA NASR CSV files. Replace with actual paths.
fix_df = pd.read_csv(
    "PATH_TO_FIX_BASE_CSV",  # <-- Replace with actual path
    low_memory=False
)
nav_df = pd.read_csv(
    "PATH_TO_NAV_BASE_CSV",  # <-- Replace with actual path
    low_memory=False
)

SEARCH_ROOT = "PATH_TO_SEARCH_ROOT"  # <-- Replace with actual root directory

NAV_Types = ["VOT", "VORTAC", "VOR/DME", "VOR", "TACAN", "NDB/DME", "NDB", "MARINE NDB", "FAN MARKER", "DME"]
FIX_Types = ["FIX", "WP"]

# ---- USCG Light List gazetteer (lazy loading) ----
_llnr_gazetteer: Optional[dict] = None
_llnr_gazetteer_path: Optional[str] = None


def _load_llnr_gazetteer(json_path: str) -> dict:
    global _llnr_gazetteer, _llnr_gazetteer_path
    if _llnr_gazetteer is not None and _llnr_gazetteer_path == json_path:
        return _llnr_gazetteer
    with open(json_path, "r", encoding="utf-8") as f:
        _llnr_gazetteer = json.load(f)
    _llnr_gazetteer_path = json_path
    print(f"[name_position] Loaded Light List gazetteer: {len(_llnr_gazetteer)} LLNR entries")
    return _llnr_gazetteer


def _resolve_llnr(llnr: str, gazetteer: dict) -> Optional[dict]:
    """Resolve coordinates by LLNR number."""
    for c in [llnr, llnr.strip(), llnr.replace("-", ""), llnr.replace(" ", "")]:
        if c in gazetteer:
            return gazetteer[c]
    return None

# Find file by name
def find_file_by_name(file_name: str = "parsed_routes_newnew_5.json", search_root: str = SEARCH_ROOT) -> str:
    for root, dirs, files in os.walk(search_root):
        if file_name in files:
            return os.path.join(root, file_name)
    raise FileNotFoundError(f"File not found: {file_name}")


def parse_route_point(text: str):
    """
    Input: 'BORINQUEN,PR,VORTAC'
    Output: {
        'name': 'BORINQUEN',
        'region': 'PR',
        'type': 'VORTAC'
    }
    """
    text = text.strip().upper()
    name, region, wp_type = [p.strip() for p in text.split(",")]

    return {
        "name": name,
        "region": region,
        "type": wp_type
    }


def get_latlon_from_csv(route_point):
    #print(route_point)
    name = route_point["name"]
    wp_type = route_point["type"]
    region = route_point["region"]

    # Special waypoints
    if str(wp_type).upper() == "NONE" and str(region).upper() == "NONE":
        print(f"Skipping special waypoint: {name} (region: {region}, type: {wp_type})")
        return {
            "name": name,
            "type": wp_type,
            "lat": None,
            "lon": None
        }

    if wp_type in NAV_Types:
        candidates = nav_df[
            (nav_df["NAME"] == name) &
            ((nav_df["STATE_CODE"] == region) | pd.isna(nav_df["STATE_CODE"]))
        ]
        # print("Filtered results:")
        # print(candidates)

    elif wp_type in FIX_Types:
        candidates = fix_df[fix_df["FIX_ID"] == name]

    else:
        print(f"Unsupported waypoint type: {wp_type} for {name}")
        return {
            "name": name,
            "type": wp_type,
            "lat": None,
            "lon": None
        }

    if candidates.empty:
        print(f"Waypoint not found: {name} (type: {wp_type}, region: {region})")
        return None

    row = candidates.iloc[0]
    return {
        "name": name,
        "type": wp_type,
        "lat": float(row["LAT_DECIMAL"]),
        "lon": float(row["LONG_DECIMAL"])
    }


def update_airway_coordinates(json_file: str, return_data: bool = False) -> dict:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_points = 0
    updated_points = 0
    error_points = 0

    for airway in data:
        if "airway_point" in airway and airway["airway_point"]:
            for point in airway["airway_point"]:
                total_points += 1
                try:
                    point_info = {
                        "name": point["name"],
                        "region": point["region"],
                        "type": point["type"],
                    }
                    coordinates = get_latlon_from_csv(point_info)

                    if (
                        coordinates is not None
                        and coordinates.get("lat") is not None
                        and coordinates.get("lon") is not None
                    ):
                        point["position"] = [coordinates["lat"], coordinates["lon"]]
                        updated_points += 1
                except Exception:
                    error_points += 1
                    continue

    output_file = json_file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    result = {
        "output_file": output_file,
        "total_points": total_points,
        "updated_points": updated_points,
        "error_points": error_points,
    }

    if return_data:
        result["data"] = data

    return result


def _update_maritime_coordinates(json_file: str, gazetteer_path: str, return_data: bool = False) -> dict:
    """Maritime event JSON: LLNR -> coordinate completion."""
    with open(json_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    gazetteer = _load_llnr_gazetteer(gazetteer_path)
    total = len(events)
    updated = 0
    skipped = 0

    for e in events:
        pos = e.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            if all(isinstance(v, (int, float)) for v in pos):
                skipped += 1
                continue
        if isinstance(pos, dict) and isinstance(pos.get("lat"), (int, float)):
            skipped += 1
            continue

        llnr = e.get("llnr", "")
        if not llnr:
            continue

        found = _resolve_llnr(llnr, gazetteer)
        if found:
            e["position"] = [found["lat"], found["lon"]]
            updated += 1

    output_file = json_file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    result = {
        "output_file": output_file,
        "total_events": total,
        "updated_events": updated,
        "skipped_events": skipped,
        "unresolved_events": total - updated - skipped,
    }
    print(f"[name_position maritime] total: {total}, already have: {skipped}, completed: {updated}, unresolved: {result['unresolved_events']}")

    if return_data:
        result["data"] = events
    return result

def _normalize_input_path(path_str: str) -> str:
    """
    Supports two input types:
    1. Virtual path: /text/xxx.json
    2. Host absolute path: D:/.../xxx.json
    """
    if isinstance(path_str, str) and path_str.startswith("/"):
        return to_host_path(path_str)
    return path_str

@tool(parse_docstring=True)
def find_file_by_name_tool(file_name: str = 'parsed_routes_newnew_5.json', search_root: str = SEARCH_ROOT) -> dict:
    """
    Recursively search for the target file in the specified directory and its subdirectories,
    and return the full path.

    Args:
        file_name: Target filename, typically parsed_routes_newnew_5.json.
        search_root: Starting directory for the search.

    Returns:
        Dict containing the target filename, full file path, and search root directory.
    """
    #file_path = find_file_by_name(file_name, search_root)

    host_search_dir = _normalize_input_path(search_root)

    found_path = find_file_by_name(file_name, host_search_dir)  # adjusted per original function signature
    
    if not found_path:
        return {"ok": False, "found": False, "file_name": file_name}
    return {
        "ok": True,
        "found": True,
        "file_name": file_name,
        "search_root": search_root,
        "found_host_path": found_path,
        "found_virtual_path": to_virtual_path(found_path),
    }

def make_update_coordinates_tool(profile=None):
    """
    Create a domain-aware coordinate completion tool.

    profile=None  → FAA waypoint coordinate completion (NASR CSV)
    profile=USCG_LNM_PROFILE → Maritime LLNR coordinate completion (Light List JSON)

    The tool signature is identical to update_airway_coordinates_tool,
    so the multi-agent framework does not need to be aware of underlying gazetteer differences.
    """
    from domap.domain_profile import FAA_AIRWAY_PROFILE, USCG_LNM_PROFILE

    _profile = profile or FAA_AIRWAY_PROFILE
    _gazetteer_config = getattr(_profile, "gazetteer_config", None) or {}

    is_maritime = (
        _profile.name == USCG_LNM_PROFILE.name
        or _gazetteer_config.get("type") == "light_list_json"
    )

    if is_maritime:
        @tool(parse_docstring=True, response_format="content_and_artifact")
        def _maritime_update_tool(
            json_file: str,
            gazetteer_path: str,
        ) -> tuple[str, dict[str, Any]]:
            """Perform LLNR -> coordinate completion on maritime event JSON files using USCG Light List as gazetteer.

            Args:
                json_file: Input JSON path. Supports virtual paths (starting with /) or host absolute paths.
                gazetteer_path: USCG Light List aids-to-navigation dictionary path (JSON format).
                    Must be extracted from the user's task description; do not fabricate. If the user
                    did not specify a dictionary path, ask the user.

            Returns:
                Tuple of (content text summary, payload structured result)
            """
            host_json_file = _normalize_input_path(json_file)
            result = _update_maritime_coordinates(host_json_file, gazetteer_path, return_data=True)

            root, ext = os.path.splitext(host_json_file)
            guessed_updated = root + "_updated" + ext

            output_host_path = result.get("output_file") or guessed_updated
            if not os.path.exists(output_host_path) and os.path.exists(guessed_updated):
                output_host_path = guessed_updated
            if not os.path.exists(output_host_path):
                output_host_path = host_json_file

            payload: dict[str, Any] = dict(result)
            payload["ok"] = True
            payload["geocoded_json_host_path"] = output_host_path
            try:
                payload["geocoded_json_path"] = to_virtual_path(output_host_path)
            except ValueError:
                payload["geocoded_json_path"] = output_host_path

            content = (
                f"Maritime event coordinate completion finished.\n"
                f"- Output file: {payload['geocoded_json_path']}\n"
                f"- Total events: {result.get('total_events', 'unknown')}\n"
                f"- Successfully completed: {result.get('updated_events', 'unknown')}\n"
                f"- Already had coordinates: {result.get('skipped_events', 0)}\n"
                f"- Unresolved: {result.get('unresolved_events', 0)}\n"
                f"- Gazetteer data source: USCG Light List"
            )
            return content, payload

        return _maritime_update_tool

    # === FAA default logic (identical to original update_airway_coordinates_tool) ===
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def _faa_update_tool(json_file: str) -> tuple[str, dict[str, Any]]:
        """Perform place name geocoding and coordinate completion on airway JSON files using FAA NASR CSV as gazetteer.

        Args:
            json_file: Input JSON path. Supports virtual paths (starting with /) or host absolute paths.

        Returns:
            Tuple of (content text summary, payload structured result)
        """
        host_json_file = _normalize_input_path(json_file)
        result = update_airway_coordinates(host_json_file, return_data=True)

        root, ext = os.path.splitext(host_json_file)
        guessed_updated = root + "_updated" + ext

        output_host_path = None
        if isinstance(result, dict):
            output_host_path = (
                result.get("updated_json_path")
                or result.get("output_json_path")
                or result.get("output_file")
                or result.get("saved_json_path")
                or result.get("saved_path")
                or result.get("json_file")
                or result.get("path")
            )
            if (not output_host_path) and os.path.exists(guessed_updated):
                output_host_path = guessed_updated
            if (not output_host_path) and os.path.exists(host_json_file):
                output_host_path = host_json_file

            payload: dict[str, Any] = dict(result)
            payload["ok"] = True
            payload["geocoded_json_host_path"] = output_host_path
            payload["geocoded_json_path"] = to_virtual_path(output_host_path)

            total_points = (
                payload.get("total_points")
                or payload.get("all_points")
                or payload.get("point_count")
                or "unknown"
            )
            updated_points = (
                payload.get("updated_points")
                or payload.get("success_points")
                or payload.get("updated_count")
                or "unknown"
            )
            error_points = (
                payload.get("error_points")
                or payload.get("failed_points")
                or payload.get("error_count")
                or 0
            )

            content = (
                f"Waypoint coordinate completion finished.\n"
                f"- Output file: {payload['geocoded_json_path']}\n"
                f"- Total points: {total_points}\n"
                f"- Successfully completed: {updated_points}\n"
                f"- Failed: {error_points}\n"
                f"- Note: prioritized the actually generated updated JSON file."
            )
            return content, payload

        # Fallback
        if os.path.exists(guessed_updated):
            output_host_path = guessed_updated
        else:
            output_host_path = host_json_file

        payload = {
            "ok": True,
            "geocoded_json_host_path": output_host_path,
            "geocoded_json_path": to_virtual_path(output_host_path),
            "raw_result": result,
        }
        content = (
            f"Waypoint coordinate completion finished.\n"
            f"- Output file: {payload['geocoded_json_path']}\n"
            f"- Note: the underlying tool did not return standard fields; auto-checked for _updated.json in the same directory."
        )
        return content, payload

    return _faa_update_tool


# =========================
# Backward-compatibility aliases
# =========================
# Default (FAA) tool
update_airway_coordinates_tool = make_update_coordinates_tool()




if __name__ == "__main__":
    result = update_airway_coordinates(
        find_file_by_name("parsed_routes_newnew_5.json"
    ))
    #print(result)