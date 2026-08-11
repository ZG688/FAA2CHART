# -*- coding: utf-8 -*-
import json

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    QgsGeometry,
    QgsPointXY,
    QgsFeature,
    QgsField,
    QgsVectorLayer,
    QgsFillSymbol,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsSingleSymbolRenderer,
    QgsCategorizedSymbolRenderer,
    QgsRendererCategory,
    QgsPalLayerSettings,
    QgsTextFormat,
    QgsTextBufferSettings,
    QgsVectorLayerSimpleLabeling,
    QgsProperty,
    QgsSymbolLayer,
)

from domap.connectivity.shared_map_state import (
    US_STATE_ABBR,
    normalize_state_abbr,
    clean_coord,
    safe_point_on_surface,
    curved_line_points,
    scale_width,
)

from qgis.core import QgsProject

def get_live_project():
    project = QgsProject.instance()
    if project is None:
        raise RuntimeError("QgsProject.instance() returned None.")
    return project

def load_routes_json(state: dict) -> dict:
    cfg = state["cfg"]
    with open(cfg.paths.json_path, "r", encoding="utf-8") as f:
        routes = json.load(f)
    state["data"]["routes"] = routes
    return state


def build_state_centroids(state: dict) -> dict:
    display_layer = state["layers"]["state_display_layer"]

    centroids = {}
    names = {}

    for f in display_layer.getFeatures():
        abbr = f["abbr"]
        pt = safe_point_on_surface(f.geometry())
        if pt is not None:
            centroids[abbr] = QgsPointXY(pt.x(), pt.y())
            names[abbr] = f["name"]

    target_state = state["cfg"].data.target_state
    if target_state not in centroids:
        raise Exception("TARGET_STATE {} not found in boundary file.".format(target_state))

    state["analysis"]["state_centroids"] = centroids
    state["analysis"]["state_name_map"] = names
    return state


def ensure_metric(metrics, st):
    if st not in metrics:
        metrics[st] = {
            "direct_segment_count": 0,
            "airway_codes": set(),
            "airway_count": 0,
            "score": 0
        }


def compute_connectivity_metrics(state: dict) -> dict:
    cfg = state["cfg"]
    routes = state["data"]["routes"]
    state_centroids = state["analysis"]["state_centroids"]
    target_state = cfg.data.target_state

    metrics = {}
    total_adjacent_pairs = 0
    valid_adjacent_pairs = 0
    skipped_missing = 0
    target_direct_segments = 0

    for airway in routes:
        airway_code = airway.get("airway_code", "UNKNOWN")
        pts = airway.get("airway_point", []) or []

        states_in_airway = set()
        for p in pts:
            reg = normalize_state_abbr(p.get("region"))
            pos = p.get("position") or [None, None]
            lat = clean_coord(pos[0])
            lon = clean_coord(pos[1])
            if reg in US_STATE_ABBR and lat is not None and lon is not None:
                states_in_airway.add(reg)

        if target_state in states_in_airway:
            for st in states_in_airway:
                if st == target_state:
                    continue
                ensure_metric(metrics, st)
                metrics[st]["airway_codes"].add(airway_code)

        for i in range(len(pts) - 1):
            total_adjacent_pairs += 1

            p1 = pts[i]
            p2 = pts[i + 1]

            reg1 = normalize_state_abbr(p1.get("region"))
            reg2 = normalize_state_abbr(p2.get("region"))

            pos1 = p1.get("position") or [None, None]
            pos2 = p2.get("position") or [None, None]

            lat1 = clean_coord(pos1[0])
            lon1 = clean_coord(pos1[1])
            lat2 = clean_coord(pos2[0])
            lon2 = clean_coord(pos2[1])

            if None in (lat1, lon1, lat2, lon2):
                skipped_missing += 1
                continue

            if reg1 not in US_STATE_ABBR or reg2 not in US_STATE_ABBR:
                continue

            valid_adjacent_pairs += 1

            if reg1 != reg2 and target_state in {reg1, reg2}:
                other = reg2 if reg1 == target_state else reg1
                ensure_metric(metrics, other)
                metrics[other]["direct_segment_count"] += 1
                target_direct_segments += 1

    rows = []
    for st, m in metrics.items():
        m["airway_count"] = len(m["airway_codes"])
        m["score"] = (
            cfg.data.score_weight_direct_segment * m["direct_segment_count"] +
            cfg.data.score_weight_airway_count * m["airway_count"]
        )

        if st not in state_centroids:
            continue
        if m["score"] <= 0:
            continue

        rows.append({
            "state": st,
            "direct_segment_count": m["direct_segment_count"],
            "airway_count": m["airway_count"],
            "score": m["score"]
        })

    rows.sort(key=lambda x: x["score"], reverse=True)

    if cfg.data.top_n_states is not None:
        rows = rows[:cfg.data.top_n_states]

    summary = {
        "routes_loaded": len(routes),
        "total_adjacent_pairs": total_adjacent_pairs,
        "valid_adjacent_pairs": valid_adjacent_pairs,
        "skipped_missing": skipped_missing,
        "target_direct_segments": target_direct_segments,
        "connected_state_count": len(rows),
    }

    state["analysis"]["rows"] = rows
    state["analysis"]["summary"] = summary
    return state


def print_analysis_summary(state: dict) -> dict:
    cfg = state["cfg"]
    if "summary" not in state["analysis"]:
        state = compute_connectivity_metrics(state)
    summary = state["analysis"]["summary"]

    print("====================================")
    print("FAA JSON summary")
    print("Target state:", cfg.data.target_state)
    print("Routes loaded:", summary["routes_loaded"])
    print("Total adjacent point pairs:", summary["total_adjacent_pairs"])
    print("Valid adjacent point pairs:", summary["valid_adjacent_pairs"])
    print("Skipped due to missing coordinates:", summary["skipped_missing"])
    print("Target direct interstate adjacent segments:", summary["target_direct_segments"])
    print("States connected to {}: {}".format(cfg.data.target_state, summary["connected_state_count"]))
    print("====================================")
    return state


def create_connectivity_layers(state: dict) -> dict:
    cfg = state["cfg"]

    flow_layer = QgsVectorLayer("LineString?crs=EPSG:4326", cfg.layers.flow, "memory")
    flow_pr = flow_layer.dataProvider()
    flow_pr.addAttributes([
        QgsField("state", QVariant.String),
        QgsField("score", QVariant.Double),
        QgsField("direct_seg", QVariant.Int),
        QgsField("airway_cnt", QVariant.Int),
        QgsField("width_mm", QVariant.Double),
        QgsField("label_txt", QVariant.String),
    ])
    flow_layer.updateFields()

    point_layer = QgsVectorLayer("Point?crs=EPSG:4326", cfg.layers.point, "memory")
    point_pr = point_layer.dataProvider()
    point_pr.addAttributes([
        QgsField("abbr", QVariant.String),
        QgsField("name", QVariant.String),
        QgsField("role", QVariant.String),
        QgsField("score", QVariant.Double),
    ])
    point_layer.updateFields()

    label_layer = QgsVectorLayer("Point?crs=EPSG:4326", cfg.layers.label, "memory")
    label_pr = label_layer.dataProvider()
    label_pr.addAttributes([
        QgsField("label", QVariant.String),
        QgsField("score", QVariant.Double),
    ])
    label_layer.updateFields()

    state["layers"]["flow_layer"] = flow_layer
    state["layers"]["point_layer"] = point_layer
    state["layers"]["label_layer"] = label_layer
    return state


def populate_connectivity_layers(state: dict) -> dict:
    cfg = state["cfg"]
    rows = state["analysis"]["rows"]
    state_centroids = state["analysis"]["state_centroids"]
    state_name_map = state["analysis"]["state_name_map"]

    if not rows:
        raise Exception("No target-state connectivity found for TARGET_STATE={}".format(cfg.data.target_state))

    flow_layer = state["layers"]["flow_layer"]
    point_layer = state["layers"]["point_layer"]
    label_layer = state["layers"]["label_layer"]

    flow_pr = flow_layer.dataProvider()
    point_pr = point_layer.dataProvider()
    label_pr = label_layer.dataProvider()

    scores = [r["score"] for r in rows]
    smin = min(scores)
    smax = max(scores)

    flow_feats = []
    point_feats = []
    label_feats = []

    target_pt = state_centroids[cfg.data.target_state]

    pf = QgsFeature(point_layer.fields())
    pf.setGeometry(QgsGeometry.fromPointXY(target_pt))
    pf["abbr"] = cfg.data.target_state
    pf["name"] = state_name_map.get(cfg.data.target_state, cfg.data.target_state)
    pf["role"] = "target"
    pf["score"] = 0.0
    point_feats.append(pf)

    for r in rows:
        st = r["state"]
        src_pt = state_centroids[st]

        curve_pts = curved_line_points(
            src_pt,
            target_pt,
            bend_factor=cfg.data.flow_bend_factor,
            steps=cfg.data.flow_steps
        )

        ff = QgsFeature(flow_layer.fields())
        ff.setGeometry(QgsGeometry.fromPolylineXY(curve_pts))
        ff["state"] = st
        ff["score"] = float(r["score"])
        ff["direct_seg"] = int(r["direct_segment_count"])
        ff["airway_cnt"] = int(r["airway_count"])
        ff["width_mm"] = float(scale_width(
            r["score"], smin, smax,
            cfg.data.flow_width_min_mm,
            cfg.data.flow_width_max_mm
        ))
        ff["label_txt"] = "{} -> {} | score={} | direct_seg={} | airway_cnt={}".format(
            st, cfg.data.target_state, r["score"], r["direct_segment_count"], r["airway_count"]
        )
        flow_feats.append(ff)

        pf = QgsFeature(point_layer.fields())
        pf.setGeometry(QgsGeometry.fromPointXY(src_pt))
        pf["abbr"] = st
        pf["name"] = state_name_map.get(st, st)
        pf["role"] = "other"
        pf["score"] = float(r["score"])
        point_feats.append(pf)

    for r in rows[:cfg.data.label_top_k]:
        st = r["state"]
        src_pt = state_centroids[st]

        lf = QgsFeature(label_layer.fields())
        lf.setGeometry(QgsGeometry.fromPointXY(src_pt))
        lf["label"] = "{} ({})".format(st, r["score"])
        lf["score"] = float(r["score"])
        label_feats.append(lf)

    flow_pr.addFeatures(flow_feats)
    point_pr.addFeatures(point_feats)
    label_pr.addFeatures(label_feats)

    flow_layer.updateExtents()
    point_layer.updateExtents()
    label_layer.updateExtents()

    return state


def apply_thematic_styles(state: dict) -> dict:
    cfg = state["cfg"]

    state_display_layer = state["layers"]["state_display_layer"]
    target_display_layer = state["layers"]["target_display_layer"]
    flow_layer = state["layers"]["flow_layer"]
    point_layer = state["layers"]["point_layer"]
    label_layer = state["layers"]["label_layer"]

    state_symbol = QgsFillSymbol.createSimple(cfg.style.state_style)
    target_symbol = QgsFillSymbol.createSimple(cfg.style.target_style)
    state_display_layer.setRenderer(QgsSingleSymbolRenderer(state_symbol))
    target_display_layer.setRenderer(QgsSingleSymbolRenderer(target_symbol))

    flow_symbol = QgsLineSymbol.createSimple({
        "line_color": cfg.style.flow_style["line_color"],
        "line_width": cfg.style.flow_style["line_width"],
        "capstyle": cfg.style.flow_style["capstyle"],
        "joinstyle": cfg.style.flow_style["joinstyle"]
    })
    flow_symbol.setOpacity(cfg.style.flow_style["opacity"])

    try:
        flow_symbol.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeWidth,
            QgsProperty.fromExpression('coalesce("width_mm", 0.8)')
        )
    except Exception:
        pass
    flow_layer.setRenderer(QgsSingleSymbolRenderer(flow_symbol))

    point_categories = []
    sym_target = QgsMarkerSymbol.createSimple(cfg.style.point_target_style)
    point_categories.append(QgsRendererCategory("target", sym_target, "target"))

    sym_other = QgsMarkerSymbol.createSimple(cfg.style.point_other_style)
    point_categories.append(QgsRendererCategory("other", sym_other, "other"))

    point_layer.setRenderer(QgsCategorizedSymbolRenderer("role", point_categories))

    label_symbol = QgsMarkerSymbol.createSimple(cfg.style.label_point_style)
    label_layer.setRenderer(QgsSingleSymbolRenderer(label_symbol))

    txt = QgsTextFormat()
    txt.setFont(QFont(
        cfg.style.label_text_style["font_family"],
        cfg.style.label_text_style["font_size"]
    ))
    txt.setSize(cfg.style.label_text_style["font_size"])

    buf = QgsTextBufferSettings()
    buf.setEnabled(cfg.style.label_text_style["buffer_enabled"])
    buf.setColor(QColor(cfg.style.label_text_style["buffer_color"]))
    buf.setSize(cfg.style.label_text_style["buffer_size"])
    txt.setBuffer(buf)

    pal = QgsPalLayerSettings()
    pal.fieldName = "label"
    pal.placement = QgsPalLayerSettings.Placement.OverPoint
    pal.setFormat(txt)

    label_layer.setLabelsEnabled(True)
    label_layer.setLabeling(QgsVectorLayerSimpleLabeling(pal))

    return state


def register_analysis_layers(state: dict) -> dict:
    """
    Add thematic layers to the QGIS project.
    """
    project = get_live_project()
    # project.addMapLayer(state["layers"]["flow_layer"])
    # project.addMapLayer(state["layers"]["point_layer"])
    # project.addMapLayer(state["layers"]["label_layer"])
    layers_state = state.get("layers", {})

    flow_layer = layers_state.get("flow_layer")
    point_layer = layers_state.get("point_layer")
    label_layer = layers_state.get("label_layer")

    for lyr in [flow_layer, point_layer, label_layer]:
        if lyr is None:
            continue

        # Avoid duplicate adding
        if project.mapLayer(lyr.id()) is None:
            project.addMapLayer(lyr)

    return state


def build_visible_layer_order(state: dict) -> dict:
    visible_layers = [
        state["layers"]["label_layer"],
        state["layers"]["point_layer"],
        state["layers"]["flow_layer"],
        state["layers"]["target_display_layer"],
        state["layers"]["state_display_layer"],
    ]
    refresh_layers = [
        state["layers"]["state_display_layer"],
        state["layers"]["target_display_layer"],
        state["layers"]["flow_layer"],
        state["layers"]["point_layer"],
        state["layers"]["label_layer"],
    ]
    state["analysis"]["visible_layers"] = visible_layers
    state["analysis"]["refresh_layers"] = refresh_layers
    state["analysis"]["map_extent"] = state["layers"]["state_display_layer"].extent()
    return state