# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass, field

from qgis.core import (
    QgsRectangle,
    QgsPointXY,
    QgsGeometry,
    QgsWkbTypes,
)

US_STATE_ABBR = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
}

STATE_NAME_TO_ABBR = {
    "ALABAMA":"AL","ALASKA":"AK","ARIZONA":"AZ","ARKANSAS":"AR","CALIFORNIA":"CA",
    "COLORADO":"CO","CONNECTICUT":"CT","DELAWARE":"DE","FLORIDA":"FL","GEORGIA":"GA",
    "HAWAII":"HI","IDAHO":"ID","ILLINOIS":"IL","INDIANA":"IN","IOWA":"IA",
    "KANSAS":"KS","KENTUCKY":"KY","LOUISIANA":"LA","MAINE":"ME","MARYLAND":"MD",
    "MASSACHUSETTS":"MA","MICHIGAN":"MI","MINNESOTA":"MN","MISSISSIPPI":"MS","MISSOURI":"MO",
    "MONTANA":"MT","NEBRASKA":"NE","NEVADA":"NV","NEW HAMPSHIRE":"NH","NEW JERSEY":"NJ",
    "NEW MEXICO":"NM","NEW YORK":"NY","NORTH CAROLINA":"NC","NORTH DAKOTA":"ND","OHIO":"OH",
    "OKLAHOMA":"OK","OREGON":"OR","PENNSYLVANIA":"PA","RHODE ISLAND":"RI","SOUTH CAROLINA":"SC",
    "SOUTH DAKOTA":"SD","TENNESSEE":"TN","TEXAS":"TX","UTAH":"UT","VERMONT":"VT",
    "VIRGINIA":"VA","WASHINGTON":"WA","WEST VIRGINIA":"WV","WISCONSIN":"WI","WYOMING":"WY",
    "DISTRICT OF COLUMBIA":"DC","WASHINGTON, D.C.":"DC","WASHINGTON DC":"DC"
}


@dataclass
class PathConfig:
    state_boundary_path: str
    json_path: str
    png_path: str
    pdf_path: str
    layout_name: str = "FAA_CA_Connectivity_Map"


@dataclass
class LayerNameConfig:
    state_source: str = "US_States_Source"
    state_display: str = "State_Display"
    target_display: str = "Target_State_Display"
    flow: str = "FAA_Connectivity_Flows"
    point: str = "FAA_Connectivity_Points"
    label: str = "FAA_Connectivity_Labels"

    def all_names(self):
        return [
            self.state_source,
            self.state_display,
            self.target_display,
            self.flow,
            self.point,
            self.label,
        ]


@dataclass
class DataConfig:
    target_state: str = "CA"
    top_n_states: int = None
    label_top_k: int = 10
    project_crs: str = "EPSG:4326"

    score_weight_direct_segment: float = 2.0
    score_weight_airway_count: float = 1.0

    flow_bend_factor: float = 0.22
    flow_steps: int = 40
    flow_width_min_mm: float = 0.45
    flow_width_max_mm: float = 2.60


@dataclass
class StyleConfig:
    state_style: dict = field(default_factory=lambda: {
        "color": "234,234,234,255",
        "outline_color": "185,185,185,255",
        "outline_width": "0.18"
    })
    target_style: dict = field(default_factory=lambda: {
        "color": "255,209,102,255",
        "outline_color": "120,90,20,255",
        "outline_width": "0.35"
    })
    flow_style: dict = field(default_factory=lambda: {
        "line_color": "#2c7fb8",
        "line_width": "0.8",
        "capstyle": "round",
        "joinstyle": "round",
        "opacity": 0.74
    })
    point_target_style: dict = field(default_factory=lambda: {
        "name": "circle",
        "color": "#b30000",
        "outline_color": "#ffffff",
        "outline_width": "0.35",
        "size": "3.8"
    })
    point_other_style: dict = field(default_factory=lambda: {
        "name": "circle",
        "color": "#3a3a3a",
        "outline_color": "#ffffff",
        "outline_width": "0.2",
        "size": "2.1"
    })
    label_point_style: dict = field(default_factory=lambda: {
        "name": "circle",
        "color": "#111111",
        "outline_color": "#ffffff",
        "outline_width": "0.2",
        "size": "1.4"
    })
    label_text_style: dict = field(default_factory=lambda: {
        "font_family": "Arial",
        "font_size": 8,
        "buffer_enabled": True,
        "buffer_color": "white",
        "buffer_size": 0.9
    })


@dataclass
class LayoutConfig:
    page_size: str = "A3"
    title_text: str = "Target-centered Airway Connectivity (Migration-style Arc Map)"
    subtitle_template: str = (
        "Built from FAA airway JSON | target={target} | states mapped={state_count} | "
        "score = {w1}xdirect interstate adjacent segments + {w2}xairway co-occurrence"
    )
    footer_note: str = (
        "Arc width = connectivity score | Yellow fill = target state | Labels = top connected states"
    )

    map_panel_title: str = "Arc Map"

    map_x: float = 10
    map_y: float = 24
    map_w: float = 400
    map_h: float = 224
    panel_title_h: float = 8

    foot_y: float = 255

    extent_pad_left: float = 2.0
    extent_pad_right: float = 2.0
    extent_pad_bottom: float = 0.35
    extent_pad_top: float = 1.10

    title_x: float = 10
    title_y: float = 6
    subtitle_x: float = 10
    subtitle_y: float = 14

    scale_bar_x: float = 145
    scale_bar_y: float = 266
    scale_bar_w: float = 80
    scale_bar_h: float = 8
    scale_bar_units_per_segment: float = 500
    scale_bar_segments: int = 4
    scale_bar_height: float = 3
    scale_bar_unit_label: str = "km"

    legend_x: float = 292
    legend_y: float = 255
    legend_w: float = 108
    legend_h: float = 30
    legend_title: str = "Legend"

    north_arrow_x: float = 262
    north_arrow_y: float = 256

    footer_x: float = 10
    footer_y: float = 286


@dataclass
class RunConfig:
    open_designer: bool = True
    auto_export: bool = False
    export_dpi: int = 300


@dataclass
class MapConfig:
    paths: PathConfig
    layers: LayerNameConfig = field(default_factory=LayerNameConfig)
    data: DataConfig = field(default_factory=DataConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    run: RunConfig = field(default_factory=RunConfig)


def make_initial_state(cfg: MapConfig) -> dict:
    return {
        "cfg": cfg,
        "project": None,
        "data": {},
        "layers": {},
        "analysis": {},
        "layout": {},
        "runtime": {},
    }


def normalize_state_abbr(s):
    if s is None:
        return None
    s = str(s).strip().upper()
    if s.startswith("US-"):
        s = s[3:]
    if s in US_STATE_ABBR:
        return s
    if s in STATE_NAME_TO_ABBR:
        return STATE_NAME_TO_ABBR[s]
    return None


def clean_coord(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"none", "", "nan", "null"}:
            return None
    try:
        return float(v)
    except Exception:
        return None


def pick_field(layer, candidates):
    existing = [f.name() for f in layer.fields()]
    existing_upper = {f.name().upper(): f.name() for f in layer.fields()}
    for c in candidates:
        if c in existing:
            return c
        if c.upper() in existing_upper:
            return existing_upper[c.upper()]
    return None


def scale_width(v, vmin, vmax, wmin, wmax):
    if vmax <= vmin:
        return (wmin + wmax) / 2.0
    t = (v - vmin) / float(vmax - vmin)
    return wmin + t * (wmax - wmin)


def pad_rect_asym(rect, left=2.0, right=2.0, bottom=0.35, top=1.10):
    return QgsRectangle(
        rect.xMinimum() - left,
        rect.yMinimum() - bottom,
        rect.xMaximum() + right,
        rect.yMaximum() + top
    )


def curved_line_points(p0, p1, bend_factor=0.22, steps=40):
    x0, y0 = p0.x(), p0.y()
    x1, y1 = p1.x(), p1.y()

    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist == 0:
        return [p0, p1]

    mx = (x0 + x1) / 2.0
    my = (y0 + y1) / 2.0
    px = -dy / dist
    py = dx / dist

    sign = 1.0 if mx < -95 else -1.0
    offset = bend_factor * dist

    cx = mx + sign * px * offset
    cy = my + sign * py * offset

    pts = []
    for i in range(steps + 1):
        t = i / float(steps)
        x = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * cx + t * t * x1
        y = (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * cy + t * t * y1
        pts.append(QgsPointXY(x, y))
    return pts


def detect_us_filter_expression(layer):
    field_names_upper = {f.name().upper(): f.name() for f in layer.fields()}
    candidates = [
        ("ADM0_A3", "USA"),
        ("SOV_A3", "USA"),
        ("GU_A3", "USA"),
        ("ISO_A2", "US"),
        ("ISO_3166_2", "US"),
        ("ADMIN", "United States of America"),
        ("ADM0_NAME", "United States of America"),
        ("GEONUNIT", "United States of America"),
        ("BRK_NAME", "United States of America"),
        ("COUNTRY", "United States of America"),
        ("NAME_0", "United States of America"),
    ]
    for fld_upper, val in candidates:
        if fld_upper in field_names_upper:
            fld = field_names_upper[fld_upper]
            return '"{}" = \'{}\''.format(fld, val)
    return None


def safe_point_on_surface(geom):
    if geom is None or geom.isEmpty():
        return None
    try:
        g = geom.pointOnSurface()
        if g and not g.isEmpty():
            return g.asPoint()
    except Exception:
        pass
    try:
        g = geom.centroid()
        if g and not g.isEmpty():
            return g.asPoint()
    except Exception:
        pass
    return None


def shift_positive_lon_parts_west(geom):
    if geom is None or geom.isEmpty():
        return geom

    if QgsWkbTypes.geometryType(geom.wkbType()) != QgsWkbTypes.PolygonGeometry:
        return geom

    if QgsWkbTypes.isMultiType(geom.wkbType()):
        mp = geom.asMultiPolygon()
        new_mp = []
        for poly in mp:
            g = QgsGeometry.fromPolygonXY(poly)
            try:
                cx = g.centroid().asPoint().x()
            except Exception:
                cx = None
            if cx is not None and cx > 0:
                g.translate(-360.0, 0.0)
            new_mp.append(g.asPolygon())
        return QgsGeometry.fromMultiPolygonXY(new_mp)

    poly = geom.asPolygon()
    g = QgsGeometry.fromPolygonXY(poly)
    try:
        cx = g.centroid().asPoint().x()
    except Exception:
        cx = None
    if cx is not None and cx > 0:
        g.translate(-360.0, 0.0)
    return g


def make_display_geom(abbr, geom):
    if abbr == "AK":
        return shift_positive_lon_parts_west(geom)
    return geom
