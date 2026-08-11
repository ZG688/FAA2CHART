# -*- coding: utf-8 -*-
"""
FAA2CHART/domap/connectivity/point_event_analysis_tools.py

Point event spatial analysis strategy.

Peer to spatial_analysis_tools.py (FAA airway connectivity strategy):
both implement the same set of step function signatures, with DomainProfile.analysis_strategy
determining which one the framework loads.

Applicable data: each record is an independent geographic point event (with a position field),
not a linear airway composed of multiple waypoints.

══════════════════════════════════════════════════════════════
⚠️ Generalization experiment / bottom-level runtime logic
══════════════════════════════════════════════════════════════
This file is the bottom-level runtime logic of the domain code. All defensive checks
(.get() / None checks / early return) MUST be written inside each function body in this file.
It is ABSOLUTELY FORBIDDEN to modify upper-layer framework code
(multi_agents.py, map_decoration_tools.py, domain_profile.py, etc.).
──────────────────────────────────────────────────────────────
In generalization experiments, the framework layer only knows about "point event" data
structures through natural language instructions. It may incorrectly pass incomplete state
to certain steps, so every step function in this file MUST gracefully degrade (SKIP)
under incomplete state rather than crash.
══════════════════════════════════════════════════════════════
"""
import json

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    QgsGeometry,
    QgsPointXY,
    QgsFeature,
    QgsField,
    QgsProject,
    QgsVectorLayer,
    QgsFillSymbol,
    QgsMarkerSymbol,
    QgsSingleSymbolRenderer,
    QgsCategorizedSymbolRenderer,
    QgsRendererCategory,
    QgsPalLayerSettings,
    QgsTextFormat,
    QgsTextBufferSettings,
    QgsVectorLayerSimpleLabeling,
    QgsCoordinateReferenceSystem,
)


def get_live_project():
    """Get the current QGIS active project instance."""
    project = QgsProject.instance()
    if project is None:
        raise RuntimeError("QgsProject.instance() returned None, no QGIS project object available.")
    return project


def _extract_lat_lon(event: dict):
    """Extract (lat, lon) from an event record; return (None, None) if not available."""
    pos = event.get("position")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        lat, lon = pos[0], pos[1]
    elif isinstance(pos, dict):
        lat, lon = pos.get("lat"), pos.get("lon")
    else:
        return None, None

    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return float(lat), float(lon)
    return None, None


def _pick_label_field(events: list) -> str:
    """Pick a label field from common names."""
    for key in ("aid_name", "name", "title", "id"):
        if any(key in e for e in events):
            return key
    return ""


# =====================================================================
# Step functions (same signatures as spatial_analysis_tools)
# =====================================================================

def load_routes_json(state: dict) -> dict:
    cfg = state["cfg"]
    with open(cfg.paths.json_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    state["data"]["routes"] = events
    return state


def build_state_centroids(state: dict) -> dict:
    """Point events don't need state centroids; placeholder to keep step sequence consistent."""
    state["analysis"]["state_centroids"] = {}
    state["analysis"]["state_name_map"] = {}
    return state


def compute_connectivity_metrics(state: dict) -> dict:
    """Count coordinate availability of point events; no connectivity computation."""
    events = state["data"]["routes"]

    rows = []
    for e in events:
        lat, lon = _extract_lat_lon(e)
        if lat is None:
            continue
        rows.append({
            "label": e.get("aid_name") or e.get("name") or "",
            "lat": lat,
            "lon": lon,
        })

    state["analysis"]["rows"] = rows
    state["analysis"]["summary"] = {
        "events_loaded": len(events),
        "events_with_coords": len(rows),
        "events_without_coords": len(events) - len(rows),
    }
    return state


def print_analysis_summary(state: dict) -> dict:
    if "summary" not in state["analysis"]:
        state = compute_connectivity_metrics(state)
    summary = state["analysis"]["summary"]
    print("====================================")
    print("Point event data summary")
    print("Events loaded:", summary["events_loaded"])
    print("With coordinates:", summary["events_with_coords"])
    print("Without coordinates:", summary["events_without_coords"])
    print("====================================")
    return state


def create_connectivity_layers(state: dict) -> dict:
    """Only create point layer; flow/label set to None (point events have no flow lines, labels attach directly to points)."""
    cfg = state["cfg"]
    events = state["data"]["routes"]

    # Collect all string fields that appear in events as attribute table structure
    field_names = []
    for e in events:
        for k, v in e.items():
            if k == "position":
                continue
            if k not in field_names and isinstance(v, (str, int, float, type(None))):
                field_names.append(k)

    point_layer = QgsVectorLayer("Point?crs=EPSG:4326", cfg.layers.point, "memory")
    point_layer.dataProvider().addAttributes(
        [QgsField(name, QVariant.String) for name in field_names]
    )
    point_layer.updateFields()

    state["layers"]["flow_layer"] = None
    state["layers"]["point_layer"] = point_layer
    state["layers"]["label_layer"] = None
    state["analysis"]["point_fields"] = field_names
    return state


def populate_connectivity_layers(state: dict) -> dict:
    point_layer = state["layers"].get("point_layer")
    if point_layer is None:
        print("[populate_connectivity_layers] SKIP: point_layer not yet created, please call create_connectivity_layers first")
        return state
    events = state["data"]["routes"]
    field_names = state["analysis"].get("point_fields", [])

    feats = []
    for e in events:
        lat, lon = _extract_lat_lon(e)
        if lat is None:
            continue
        f = QgsFeature(point_layer.fields())
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(lon, lat)))
        for name in field_names:
            f[name] = "" if e.get(name) is None else str(e.get(name))
        feats.append(f)

    point_layer.dataProvider().addFeatures(feats)
    point_layer.updateExtents()
    print(f"[point_event_analysis] populated {len(feats)} event points")
    return state


def apply_thematic_styles(state: dict) -> dict:
    cfg = state["cfg"]
    events = state["data"]["routes"]

    state_display_layer = state["layers"].get("state_display_layer")
    target_display_layer = state["layers"].get("target_display_layer")
    point_layer = state["layers"].get("point_layer")

    # Bottom-level defense: gracefully degrade when any layer is missing
    if state_display_layer is None and target_display_layer is None and point_layer is None:
        print("[apply_thematic_styles] SKIP: all layers are None, framework may have skipped create_connectivity_layers")
        return state

    if state_display_layer is not None:
        state_display_layer.setRenderer(
            QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(cfg.style.state_style))
        )
    if target_display_layer is not None:
        target_display_layer.setRenderer(
            QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(cfg.style.target_style))
        )

    if point_layer is None:
        print("[apply_thematic_styles] SKIP point styling: point_layer is None")
        return state

    # Categorized renderer: prefer notice_type, then action, fall back to single symbol
    cat_field = "notice_type" if any(e.get("notice_type") for e in events) else None
    if cat_field is None:
        cat_field = "action" if any(e.get("action") for e in events) else None

    if cat_field:
        cat_values = list(dict.fromkeys(
            e.get(cat_field, "") for e in events if e.get(cat_field)
        ))
        palette = [
            QColor(220, 30, 30), QColor(30, 144, 255), QColor(255, 165, 0),
            QColor(50, 205, 50), QColor(138, 43, 226), QColor(255, 215, 0),
            QColor(0, 206, 209), QColor(255, 105, 180), QColor(160, 82, 45),
            QColor(128, 128, 128),
        ]
        renderer = QgsCategorizedSymbolRenderer(cat_field, [])
        for i, val in enumerate(cat_values):
            color = palette[i % len(palette)]
            sym = QgsMarkerSymbol.createSimple({
                "color": f"{color.red()},{color.green()},{color.blue()}",
                "size": "7.0", "size_unit": "MM",
                "outline_color": "255,255,255", "outline_width": "0.8",
            })
            cat = QgsRendererCategory(val, sym, val)
            renderer.addCategory(cat)
        point_layer.setRenderer(renderer)
        print(f"[apply_thematic_styles] categorized renderer: field={cat_field}, categories={cat_values}")
    else:
        point_layer.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple({
            "color": "220,30,30",
            "size": "8.0",
            "size_unit": "MM",
            "outline_color": "255,255,255",
            "outline_width": "0.8",
        })))

    label_field = _pick_label_field(events)
    if label_field:
        _font_size = cfg.style.label_text_style.get("font_size", 12)
        _buf_size = cfg.style.label_text_style.get("buffer_size", 1.5)
        txt = QgsTextFormat()
        txt.setFont(QFont(cfg.style.label_text_style["font_family"], _font_size))
        txt.setSize(_font_size)
        txt.setColor(QColor(30, 30, 30))

        buf = QgsTextBufferSettings()
        buf.setEnabled(True)
        buf.setColor(QColor("white"))
        buf.setSize(_buf_size)
        txt.setBuffer(buf)

        pal = QgsPalLayerSettings()
        pal.fieldName = label_field
        pal.placement = QgsPalLayerSettings.Placement.AroundPoint
        pal.setFormat(txt)

        point_layer.setLabeling(QgsVectorLayerSimpleLabeling(pal))
        point_layer.setLabelsEnabled(True)

    return state


def register_analysis_layers(state: dict) -> dict:
    project = get_live_project()
    point_layer = state["layers"].get("point_layer")
    if point_layer is None:
        print("[register_analysis_layers] SKIP: point_layer is None, framework may have skipped the creation step")
        return state
    if project.mapLayer(point_layer.id()) is None:
        project.addMapLayer(point_layer)
        print(f"[register_analysis_layers] added layer to project: {point_layer.name()} (id={point_layer.id()})")
    else:
        print(f"[register_analysis_layers] layer already in project: {point_layer.name()}")

    # List all layers in the project to confirm visibility
    _all_layers = [l.name() for l in project.mapLayers().values()]
    print(f"[register_analysis_layers] current project layers: {_all_layers}")

    # Trigger main window refresh
    try:
        from qgis.utils import iface
        from PyQt5.QtWidgets import QApplication
        if iface is not None:
            iface.mapCanvas().refresh()
            QApplication.processEvents()
            print("[register_analysis_layers] main canvas refreshed")
    except Exception as _e:
        print(f"[register_analysis_layers] main canvas refresh exception: {_e}")
    return state


def build_visible_layer_order(state: dict) -> dict:
    cfg = state["cfg"]

    point_layer = state["layers"].get("point_layer")
    target_display_layer = state["layers"].get("target_display_layer")
    state_display_layer = state["layers"].get("state_display_layer")

    if target_display_layer is None:
        print("[build_visible_layer_order] SKIP: target_display_layer is None, cannot determine map extent")
        state["analysis"]["visible_layers"] = [
            l for l in [point_layer, state_display_layer] if l is not None
        ]
        state["analysis"]["refresh_layers"] = [
            l for l in [state_display_layer, point_layer] if l is not None
        ]
        return state

    visible_layers = [
        point_layer,
        target_display_layer,
        state_display_layer,
    ]
    refresh_layers = [
        state_display_layer,
        target_display_layer,
        point_layer,
    ]
    state["analysis"]["visible_layers"] = [l for l in visible_layers if l is not None]
    state["analysis"]["refresh_layers"] = [l for l in refresh_layers if l is not None]
    state["analysis"]["map_extent"] = target_display_layer.extent()

    # Bottom-level diagnostics: confirm visible_layers content
    _vl_names = []
    for _l in state["analysis"]["visible_layers"]:
        try:
            _vl_names.append(f"{_l.name()}(feat={_l.featureCount()})")
        except Exception:
            _vl_names.append(type(_l).__name__)
    print(f"[build_visible_layer_order] visible_layers ({len(_vl_names)}): {_vl_names}")

    # Main window canvas zoom to analysis extent
    try:
        from qgis.utils import iface
        from PyQt5.QtWidgets import QApplication
        if iface is not None:
            canvas = iface.mapCanvas()
            canvas.setDestinationCrs(QgsCoordinateReferenceSystem(cfg.data.project_crs))
            canvas.setExtent(state["analysis"]["map_extent"])
            # Sync refresh + process event queue to ensure background thread also takes effect
            canvas.refresh()
            QApplication.processEvents()
            print(f"[build_visible_layer_order] canvas extent set to: {state['analysis']['map_extent'].toString()}")
    except Exception as _e:
        print(f"[build_visible_layer_order] canvas zoom failed: {_e}")

    print(f"[build_visible_layer_order] map_extent = {state['analysis']['map_extent'].toString()}")
    return state