# -*- coding: utf-8 -*-
import os

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsFeature,
    QgsGeometry,
    QgsField,
)
from qgis.core import QgsProject
from domap.connectivity.shared_map_state import (
    US_STATE_ABBR,
    normalize_state_abbr,
    pick_field,
    detect_us_filter_expression,
    make_display_geom,
)
from domap.connectivity.virtual_paths import to_host_path, to_virtual_path


def validate_inputs(state: dict) -> dict:
    cfg = state["cfg"]
    cfg.data.target_state = normalize_state_abbr(cfg.data.target_state)

    if cfg.data.target_state not in US_STATE_ABBR:
        raise Exception("Invalid target_state: {}".format(cfg.data.target_state))

    if not os.path.exists(cfg.paths.state_boundary_path):
        raise Exception("STATE_BOUNDARY_PATH not found: {}".format(cfg.paths.state_boundary_path))

    if not os.path.exists(cfg.paths.json_path):
        raise Exception("JSON_PATH not found: {}".format(cfg.paths.json_path))

    return state


def setup_project_crs(state: dict) -> dict:
    cfg = state["cfg"]
    project = QgsProject.instance()
    project.setCrs(QgsCoordinateReferenceSystem(cfg.data.project_crs))
    state["project"] = project
    return state


def remove_layers_by_name(project, names):
    print("[DEBUG] remove_layers_by_name project =", project)
    print("[DEBUG] remove_layers_by_name layer_names =", names)

    if project is None:
        raise RuntimeError("remove_layers_by_name received project as None.")
    for lyr in list(project.mapLayers().values()):
        if lyr.name() in names:
            project.removeMapLayer(lyr.id())


def remove_layout_by_name(project, layout_name):
    manager = project.layoutManager()
    for ly in manager.printLayouts():
        if ly.name() == layout_name:
            manager.removeLayout(ly)



def get_live_project():
    """
    Get the current QGIS active project instance.
    Do not read from state to avoid losing the object across steps.
    """
    project = QgsProject.instance()
    print("[DEBUG] QgsProject.instance() ->", project)
    if project is None:
        raise RuntimeError("QgsProject.instance() returned None, no QGIS project object available.")
    return project


def reset_project_environment(state: dict) -> dict:
    """
    Clean old layers and old layouts, and ensure the project CRS is set.
    """
    cfg = state["cfg"]

    # Critical: do not take project from state
    project = get_live_project()

    # Ensure project CRS is set (fallback if LLM skipped setup_project_crs)
    if project.crs().authid() != cfg.data.project_crs:
        project.setCrs(QgsCoordinateReferenceSystem(cfg.data.project_crs))

    print("[DEBUG] reset_project_environment project =", project)
    print("[DEBUG] reset_project_environment layout_name =", cfg.paths.layout_name)
    print("[DEBUG] reset_project_environment layer_names =", cfg.layers.all_names())

    remove_layers_by_name(project, cfg.layers.all_names())

    layout_manager = project.layoutManager()
    print("[DEBUG] layout_manager =", layout_manager)

    if layout_manager is not None:
        old_layout = layout_manager.layoutByName(cfg.paths.layout_name)
        print("[DEBUG] old_layout =", old_layout)
        if old_layout is not None:
            layout_manager.removeLayout(old_layout)

    # Prevent old logic from keeping live objects in state
    state.pop("project", None)
    state.pop("map_item", None)
    
    state.setdefault("layout", {})
    state["layout"].clear()
    state["layout"]["layout_name"] = state["cfg"].paths.layout_name
    state["layout"]["created"] = False


    runtime = state.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("project", None)
        runtime.pop("layout", None)
        runtime.pop("map_item", None)

    return state



def load_vector_layer(path: str, name: str, provider: str = "ogr", add_to_project: bool = False):
    # Resolve path
    path = to_host_path(path) if path.startswith("/") else path
    layer = QgsVectorLayer(path, name, provider)
    if not layer.isValid():
        raise Exception("Failed to load vector layer: {}".format(path))
    if add_to_project:
        QgsProject.instance().addMapLayer(layer)
    return layer


def load_raster_layer(path: str, name: str, provider: str = "gdal", add_to_project: bool = False):
    # Resolve path
    path = to_host_path(path) if path.startswith("/") else path
    layer = QgsRasterLayer(path, name, provider)
    if not layer.isValid():
        raise Exception("Failed to load raster layer: {}".format(path))
    if add_to_project:
        QgsProject.instance().addMapLayer(layer)
    return layer


def load_state_boundary_source(state: dict) -> dict:
    cfg = state["cfg"]

    state_layer = load_vector_layer(
        cfg.paths.state_boundary_path,
        cfg.layers.state_source,
        provider="ogr",
        add_to_project=False
    )

    us_expr = detect_us_filter_expression(state_layer)
    if us_expr is not None:
        state_layer.setSubsetString(us_expr)

    name_field = pick_field(state_layer, ["NAME", "name", "name_en", "admin1Name"])
    abbr_field = pick_field(state_layer, ["STUSPS", "postal", "postal_abbr", "abbr", "state_abbr", "usps"])

    if name_field is None and abbr_field is None:
        raise Exception("Could not detect state name/abbr field in state boundary layer.")

    state["layers"]["state_source_layer"] = state_layer
    state["analysis"]["state_name_field"] = name_field
    state["analysis"]["state_abbr_field"] = abbr_field
    return state


def build_display_state_layers(state: dict) -> dict:
    cfg = state["cfg"]
    project = get_live_project()

    # Guard: single agent may call this before load_state_boundary_source,
    # auto-fallback if state_source_layer is missing
    if "state_source_layer" not in state.get("layers", {}):
        state = load_state_boundary_source(state)

    state_layer = state["layers"]["state_source_layer"]
    abbr_field = state["analysis"]["state_abbr_field"]
    name_field = state["analysis"]["state_name_field"]

    state_display = QgsVectorLayer("Polygon?crs=EPSG:4326", cfg.layers.state_display, "memory")
    state_dp = state_display.dataProvider()
    state_dp.addAttributes([
        QgsField("abbr", QVariant.String),
        QgsField("name", QVariant.String),
    ])
    state_display.updateFields()

    target_display = QgsVectorLayer("Polygon?crs=EPSG:4326", cfg.layers.target_display, "memory")
    target_dp = target_display.dataProvider()
    target_dp.addAttributes([
        QgsField("abbr", QVariant.String),
        QgsField("name", QVariant.String),
    ])
    target_display.updateFields()

    src_crs = state_layer.crs()
    dst_crs = QgsCoordinateReferenceSystem("EPSG:4326")
    transform = None
    if src_crs != dst_crs:
        transform = QgsCoordinateTransform(src_crs, dst_crs, project)

    merged_geoms = {}
    merged_names = {}

    for f in state_layer.getFeatures():
        abbr = None

        if abbr_field is not None:
            abbr = normalize_state_abbr(f[abbr_field])

        if abbr is None and name_field is not None:
            abbr = normalize_state_abbr(f[name_field])

        if abbr not in US_STATE_ABBR:
            continue

        geom = QgsGeometry(f.geometry())
        if geom is None or geom.isEmpty():
            continue

        if transform is not None:
            try:
                geom.transform(transform)
            except Exception:
                continue

        disp_geom = make_display_geom(abbr, geom)

        if abbr in merged_geoms:
            try:
                merged_geoms[abbr] = merged_geoms[abbr].combine(disp_geom)
            except Exception:
                pass
        else:
            merged_geoms[abbr] = QgsGeometry(disp_geom)
            merged_names[abbr] = str(f[name_field]) if name_field is not None else abbr

    state_feats = []
    target_feats = []

    for abbr, geom in merged_geoms.items():
        sf = QgsFeature(state_display.fields())
        sf.setGeometry(geom)
        sf["abbr"] = abbr
        sf["name"] = merged_names.get(abbr, abbr)
        state_feats.append(sf)

        if abbr == cfg.data.target_state:
            tf = QgsFeature(target_display.fields())
            tf.setGeometry(geom)
            tf["abbr"] = abbr
            tf["name"] = merged_names.get(abbr, abbr)
            target_feats.append(tf)

    state_dp.addFeatures(state_feats)
    target_dp.addFeatures(target_feats)
    state_display.updateExtents()
    target_display.updateExtents()

    state["layers"]["state_display_layer"] = state_display
    state["layers"]["target_display_layer"] = target_display
    return state


def register_basemap_layers(state: dict) -> dict:
    # Self-healing: if upstream steps were skipped by LLM, re-run prerequisite steps
    # in dependency order to avoid KeyError aborting the entire pipeline.
    if "state_source_layer" not in state["layers"]:
        state = setup_project_crs(state)
        state = load_state_boundary_source(state)
    if ("state_display_layer" not in state["layers"]
            or "target_display_layer" not in state["layers"]):
        state = build_display_state_layers(state)

    project = get_live_project()
    project.addMapLayer(state["layers"]["state_display_layer"])
    project.addMapLayer(state["layers"]["target_display_layer"])
    return state