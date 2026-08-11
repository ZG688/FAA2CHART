# -*- coding: utf-8 -*-
from __future__ import annotations

from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtWidgets import QApplication

from qgis.core import (
    QgsProject,
    QgsPrintLayout,
    QgsLayoutItem,
    QgsLayoutItemPage,
    QgsLayoutItemLabel,
    QgsLayoutItemMap,
    QgsLayoutItemScaleBar,
    QgsLayoutItemLegend,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsUnitTypes,
    QgsLayoutExporter,
    QgsLayoutRenderContext
)

from domap.connectivity.shared_map_state import pad_rect_asym

# Module-level tracking: prevent the same layout from being add_map_panel'd multiple times
_add_map_panel_done_layouts = set()
import threading
_add_map_panel_lock = threading.Lock()

# =========================================================
# Runtime helpers
# =========================================================
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


def _ensure_layout_state(state: dict) -> dict:
    state.setdefault("layout", {})
    return state["layout"]


def get_live_layout(state: dict):
    """
    Dynamically get the current layout object by layout_name.
    Prefer the cached layout object from the current run; fall back to layoutManager lookup by name.
    """
    layout_state = _ensure_layout_state(state)

    # Try cached object first
    cached_layout = layout_state.get("_layout_obj")
    if cached_layout is not None:
        try:
            _ = cached_layout.name()
            return cached_layout
        except Exception:
            # Cached object is stale, continue with manager lookup
            pass

    project = get_live_project()
    cfg = state["cfg"]

    layout_name = layout_state.get("layout_name") or cfg.paths.layout_name
    layout_manager = project.layoutManager()
    layout = layout_manager.layoutByName(layout_name)

    if layout is None:
        try:
            all_layout_names = [lyt.name() for lyt in layout_manager.printLayouts()]
        except Exception:
            all_layout_names = []

        raise RuntimeError(
            f"Layout not found: {layout_name}."
            f" Current layouts in layoutManager: {all_layout_names}"
        )

    # Cache upon finding
    layout_state["_layout_obj"] = layout
    return layout

def ensure_layout_exists(state: dict):
    """
    Ensure the current task layout exists.
    If not, auto-create it.
    """
    try:
        return get_live_layout(state)
    except Exception as e:
        print("[DEBUG] ensure_layout_exists: layout missing, auto create. reason =", e)
        create_layout(state)
        return get_live_layout(state)


def get_live_map_item(state: dict):
    """
    Dynamically get the main map item in the layout, without relying on live map_item objects in state.
    If LLM calls add_scale_bar / add_legend before add_map_panel,
    auto-create map panel for robustness.
    """
    layout = ensure_layout_exists(state)

    for item in layout.items():
        if isinstance(item, QgsLayoutItemMap):
            return item

    # Auto-repair: add_map_panel hasn't been called yet, but decoration tools need map_item
    try:
        add_map_panel(state)
        for item in layout.items():
            if isinstance(item, QgsLayoutItemMap):
                return item
    except Exception as auto_e:
        print(f"[get_live_map_item] auto add_map_panel failed: {auto_e}")

    item_types = [type(item).__name__ for item in layout.items()]
    raise RuntimeError(
        "No QgsLayoutItemMap found in current layout. Please run add_map_panel first."
        f" Current layout item types: {item_types}"
    )



def _safe_remove_items_by_text(layout, text_list):
    """
    Remove old labels by text to avoid duplicates.
    """
    if not text_list:
        return

    for item in list(layout.items()):
        try:
            if isinstance(item, QgsLayoutItemLabel) and item.text() in text_list:
                layout.removeLayoutItem(item)
        except Exception:
            pass


def _safe_remove_first_item_by_type(layout, item_type):
    """
    Remove all layout items matching the given type. Used to avoid duplicate scale bars, legends, map panels, etc.
    """
    for item in list(layout.items()):
        try:
            if isinstance(item, item_type):
                layout.removeLayoutItem(item)
        except Exception:
            pass


# =========================================================
# Layout steps
# =========================================================
def create_layout(state: dict) -> dict:
    project = get_live_project()
    cfg = state["cfg"]
    layout_state = _ensure_layout_state(state)

    layout_manager = project.layoutManager()

    # Remove all old layouts with the same name to avoid addLayout failure
    try:
        for old_layout in list(layout_manager.printLayouts()):
            try:
                if old_layout.name() == cfg.paths.layout_name:
                    layout_manager.removeLayout(old_layout)
            except Exception:
                pass
    except Exception:
        pass

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(cfg.paths.layout_name)

    # Clear the add_map_panel completion flag for this layout id, preventing Python id reuse
    # from causing subsequent add_map_panel calls to be silently skipped
    # (when re-running experiments in the same QGIS session)
    _add_map_panel_done_layouts.discard(id(layout))

    added = layout_manager.addLayout(layout)
    print("[DEBUG] create_layout addLayout returned =", added)
    if not added:
        try:
            all_layout_names = [lyt.name() for lyt in layout_manager.printLayouts()]
        except Exception:
            all_layout_names = []

        raise RuntimeError(
            f"Failed to create layout, layoutManager.addLayout(layout) returned False."
            f" Target layout name: {cfg.paths.layout_name}."
            f" Current layout list: {all_layout_names}"
        )

    page = layout.pageCollection().pages()[0]
    page.setPageSize(
        cfg.layout.page_size,
        QgsLayoutItemPage.Orientation.Landscape
    )

    # Cache the layout object for fallback use within the same tool chain round
    layout_state["_layout_obj"] = layout
    layout_state["layout_name"] = cfg.paths.layout_name
    layout_state["created"] = True

    try:
        all_layout_names = [lyt.name() for lyt in layout_manager.printLayouts()]
    except Exception:
        all_layout_names = []

    print(f"[DEBUG] create_layout success, layout_name = {cfg.paths.layout_name}")
    print(f"[DEBUG] current layouts = {all_layout_names}")
    
    return state



def add_layout_title(state: dict) -> dict:
    cfg = state["cfg"]
    layout = ensure_layout_exists(state)
    layout_state = _ensure_layout_state(state)

    rows = state.get("analysis", {}).get("rows", [])

    subtitle_text = cfg.layout.subtitle_template.format(
        target=cfg.data.target_state,
        state_count=len(rows),
        w1=cfg.data.score_weight_direct_segment,
        w2=cfg.data.score_weight_airway_count,
    )

    # Avoid duplicate text items with the same name
    _safe_remove_items_by_text(layout, [cfg.layout.title_text, subtitle_text])

    title = QgsLayoutItemLabel(layout)
    title.setText(cfg.layout.title_text)
    title.setFont(QFont("Arial", 16))
    title.adjustSizeToText()
    title.attemptMove(QgsLayoutPoint(cfg.layout.title_x, cfg.layout.title_y, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(title)

    subtitle = QgsLayoutItemLabel(layout)
    subtitle.setText(subtitle_text)
    subtitle.setFont(QFont("Arial", 9))
    subtitle.adjustSizeToText()
    subtitle.attemptMove(QgsLayoutPoint(cfg.layout.subtitle_x, cfg.layout.subtitle_y, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(subtitle)

    layout_state["title_added"] = True
    layout_state["subtitle_added"] = True
    return state


def add_map_panel(state: dict) -> dict:
    cfg = state["cfg"]
    project = get_live_project()
    layout = ensure_layout_exists(state)
    layout_state = _ensure_layout_state(state)

    # Prevent LLM duplicate calls / concurrent race: atomic check+create+mark
    _layout_obj = layout
    # The entire critical section is serialized: check -> clean -> create -> mark must be atomic,
    # otherwise concurrent threads will all pass the check before the mark is set, leading to duplicate QgsLayoutItemMap creation.
    with _add_map_panel_lock:
        if id(_layout_obj) in _add_map_panel_done_layouts:
            print("[add_map_panel] SKIP (module-level flag): already called for this layout")
            return state
        for _item in layout.items():
            if isinstance(_item, QgsLayoutItemMap):
                _add_map_panel_done_layouts.add(id(_layout_obj))
                print("[add_map_panel] SKIP (layout check): layout already has QgsLayoutItemMap")
                return state

        state_display_layer = state["layers"]["state_display_layer"]
        visible_layers = state["analysis"]["visible_layers"]

        # Prefer analysis-strategy-specified map_extent, fall back to state_display_layer
        extent_layer_ext = state["analysis"].get("map_extent")
        print(f"[add_map_panel] map_extent from analysis: {extent_layer_ext is not None}")
        if extent_layer_ext is None:
            print(f"[add_map_panel] WARNING: map_extent not set, falling back to state_display_layer (full-US extent)")
            extent_layer_ext = state_display_layer.extent()
        else:
            print(f"[add_map_panel] map_extent = {extent_layer_ext.toString()}")

        extent = pad_rect_asym(
            extent_layer_ext,
            cfg.layout.extent_pad_left,
            cfg.layout.extent_pad_right,
            cfg.layout.extent_pad_bottom,
            cfg.layout.extent_pad_top
        )

        # Remove old map panel title
        _safe_remove_items_by_text(layout, [cfg.layout.map_panel_title])

        # Precision cleanup: only remove old QgsLayoutItemMap, keep title/subtitle/scale bar/legend/north arrow/footer etc.
        _safe_remove_first_item_by_type(layout, QgsLayoutItemMap)

        title_item = QgsLayoutItemLabel(layout)
        title_item.setText(cfg.layout.map_panel_title)
        title_item.setFont(QFont("Arial", 11))
        title_item.setBackgroundEnabled(True)
        title_item.setBackgroundColor(QColor("#f2f2f2"))
        title_item.setFrameEnabled(True)
        layout.addLayoutItem(title_item)
        title_item.attemptMove(QgsLayoutPoint(cfg.layout.map_x, cfg.layout.map_y, QgsUnitTypes.LayoutMillimeters))
        title_item.attemptResize(QgsLayoutSize(cfg.layout.map_w, cfg.layout.panel_title_h, QgsUnitTypes.LayoutMillimeters))

        map_item = QgsLayoutItemMap(layout)
        layout.addLayoutItem(map_item)
        map_item.setFrameEnabled(True)
        map_item.setBackgroundEnabled(True)
        map_item.setBackgroundColor(QColor("#f7f7f7"))
        map_item.setKeepLayerSet(True)
        map_item.setLayers(visible_layers)
        map_item.setCrs(project.crs())

        map_item.attemptMove(QgsLayoutPoint(
            cfg.layout.map_x,
            cfg.layout.map_y + cfg.layout.panel_title_h,
            QgsUnitTypes.LayoutMillimeters
        ))
        map_item.attemptResize(QgsLayoutSize(
            cfg.layout.map_w,
            cfg.layout.map_h - cfg.layout.panel_title_h,
            QgsUnitTypes.LayoutMillimeters
        ))
        map_item.zoomToExtent(extent)
        map_item.refresh()

        layout_state["map_panel_title_added"] = True
        layout_state["map_item"] = True
        layout_state["map_extent"] = extent
        _add_map_panel_done_layouts.add(id(_layout_obj))
    return state


def add_scale_bar(state: dict) -> dict:
    cfg = state["cfg"]
    layout = ensure_layout_exists(state)
    layout_state = _ensure_layout_state(state)
    map_item = get_live_map_item(state)

    _safe_remove_first_item_by_type(layout, QgsLayoutItemScaleBar)

    scale_bar = QgsLayoutItemScaleBar(layout)
    scale_bar.setStyle("Single Box")
    scale_bar.setLinkedMap(map_item)
    scale_bar.setUnits(QgsUnitTypes.DistanceKilometers)
    scale_bar.setNumberOfSegments(cfg.layout.scale_bar_segments)
    scale_bar.setNumberOfSegmentsLeft(0)
    scale_bar.setUnitsPerSegment(cfg.layout.scale_bar_units_per_segment)
    scale_bar.setUnitLabel(cfg.layout.scale_bar_unit_label)
    scale_bar.setFont(QFont("Arial", 8))
    scale_bar.setHeight(cfg.layout.scale_bar_height)
    layout.addLayoutItem(scale_bar)
    scale_bar.attemptMove(QgsLayoutPoint(cfg.layout.scale_bar_x, cfg.layout.scale_bar_y, QgsUnitTypes.LayoutMillimeters))
    scale_bar.attemptResize(QgsLayoutSize(cfg.layout.scale_bar_w, cfg.layout.scale_bar_h, QgsUnitTypes.LayoutMillimeters))
    scale_bar.update()

    layout_state["scale_bar"] = True
    return state


def add_legend(state: dict) -> dict:
    cfg = state["cfg"]
    layout = ensure_layout_exists(state)
    layout_state = _ensure_layout_state(state)
    map_item = get_live_map_item(state)

    _safe_remove_first_item_by_type(layout, QgsLayoutItemLegend)

    legend = QgsLayoutItemLegend(layout)
    legend.setTitle(cfg.layout.legend_title)
    legend.setLinkedMap(map_item)
    legend.setLegendFilterByMapEnabled(True)
    legend.setAutoUpdateModel(True)
    legend.setFrameEnabled(True)
    legend.setBackgroundEnabled(True)
    legend.setBackgroundColor(QColor(255, 255, 255, 230))
    layout.addLayoutItem(legend)
    legend.attemptMove(QgsLayoutPoint(cfg.layout.legend_x, cfg.layout.legend_y, QgsUnitTypes.LayoutMillimeters))
    legend.attemptResize(QgsLayoutSize(cfg.layout.legend_w, cfg.layout.legend_h, QgsUnitTypes.LayoutMillimeters))

    layout_state["legend"] = True
    return state


def add_simple_north_arrow(state: dict) -> dict:
    cfg = state["cfg"]
    layout = ensure_layout_exists(state)
    layout_state = _ensure_layout_state(state)

    _safe_remove_items_by_text(layout, ["N", "▲"])

    n_label = QgsLayoutItemLabel(layout)
    n_label.setText("N")
    n_label.setFont(QFont("Arial", 11, QFont.Bold))
    layout.addLayoutItem(n_label)
    n_label.adjustSizeToText()
    n_label.attemptMove(QgsLayoutPoint(cfg.layout.north_arrow_x + 4, cfg.layout.north_arrow_y, QgsUnitTypes.LayoutMillimeters))

    arrow_label = QgsLayoutItemLabel(layout)
    arrow_label.setText("▲")
    arrow_label.setFont(QFont("Arial", 22))
    layout.addLayoutItem(arrow_label)
    arrow_label.adjustSizeToText()
    arrow_label.attemptMove(QgsLayoutPoint(cfg.layout.north_arrow_x, cfg.layout.north_arrow_y + 4, QgsUnitTypes.LayoutMillimeters))

    layout_state["north_arrow"] = True
    return state


def add_footer_note(state: dict) -> dict:
    cfg = state["cfg"]
    layout = ensure_layout_exists(state)
    layout_state = _ensure_layout_state(state)

    _safe_remove_items_by_text(layout, [cfg.layout.footer_note])

    footer_note = QgsLayoutItemLabel(layout)
    footer_note.setText(cfg.layout.footer_note)
    footer_note.setFont(QFont("Arial", 8))
    footer_note.adjustSizeToText()
    footer_note.attemptMove(QgsLayoutPoint(cfg.layout.footer_x, cfg.layout.footer_y, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(footer_note)

    layout_state["footer_note"] = True
    return state


def force_refresh_layout(state: dict) -> dict:
    layout = ensure_layout_exists(state)
    layers = state.get("analysis", {}).get("refresh_layers", [])

    for lyr in layers:
        try:
            lyr.triggerRepaint()
        except Exception:
            pass

    QApplication.processEvents()

    try:
        from qgis.utils import iface
        if iface is not None:
            iface.mapCanvas().refresh()
    except Exception:
        pass

    QApplication.processEvents()

    for item in layout.items():
        if isinstance(item, QgsLayoutItemMap):
            try:
                item.refresh()
            except Exception:
                pass

    try:
        layout.refresh()
    except Exception:
        pass

    QApplication.processEvents()
    return state


def open_layout_designer_safe(state: dict) -> dict:
    """
    Safely handle the "open layout designer" step.

    Note:
    - Multi-agent tool steps typically run in background threads;
    - QGIS Designer is a GUI operation and should be opened manually on the main thread;
    - Therefore this function does not directly call iface.openLayoutDesigner(...), only records state.
    """
    layout_state = _ensure_layout_state(state)
    cfg = state["cfg"]

    if not getattr(cfg.run, "open_designer", False):
        layout_state["designer_opened"] = False
        layout_state["designer_message"] = "Skipped auto-opening layout designer (open_designer=False)."
        return state

    layout_state["designer_opened"] = False
    layout_state["designer_message"] = (
        "Layout created successfully, but did not auto-open QGIS Designer in the tool thread."
        " Please manually open the layout from the QGIS main thread after the task completes."
    )

    return state


def export_layout_if_needed(state: dict) -> dict:
    cfg = state["cfg"]
    layout = ensure_layout_exists(state)
    layout_state = _ensure_layout_state(state)

    if not cfg.run.auto_export:
        print("AUTO_EXPORT = False")
        print("Please inspect layout in Designer, then run export_layout_manual(state)")
        layout_state["auto_export_executed"] = False
        return state

    layout.refresh()

    # Pre-export debug: layout content check
    print("[export] layout items:", len(list(layout.items())))
    for item in layout.items():
        item_type = type(item).__name__
        if item_type == "QgsLayoutItemMap":
            lyrs = item.layers() if hasattr(item, 'layers') else []
            print(f"[export]   QgsLayoutItemMap: layers={[l.name() if l else 'None' for l in lyrs]}, "
                  f"extent={item.extent().toString()}, crs={item.crs().authid()}")
        else:
            print(f"[export]   {item_type}")

    # --- PNG export (direct render, QgsLayoutExporter renders blank in background thread, use manual rendering as in FAA script) ---
    from PyQt5.QtGui import QImage, QPainter, QColor
    from PyQt5.QtCore import QSize
    from qgis.core import QgsMapSettings, QgsMapRendererCustomPainterJob

    _page = layout.pageCollection().pages()[0]
    _ps = _page.pageSize()
    _dpi = max(cfg.run.export_dpi, 72)
    _pw = int(_ps.width() * _dpi / 25.4)
    _ph = int(_ps.height() * _dpi / 25.4)

    _ext = state["analysis"]["map_extent"]
    _ext_w = _ext.xMaximum() - _ext.xMinimum()
    _ext_h = _ext.yMaximum() - _ext.yMinimum()
    _ext_ratio = _ext_w / _ext_h if _ext_h > 0 else 1.0
    _page_ratio = _pw / _ph if _ph > 0 else 1.0

    if _ext_ratio > _page_ratio:
        _map_w = _pw
        _map_h = int(_pw / _ext_ratio)
        _map_x = 0
        _map_y = (_ph - _map_h) // 2
    else:
        _map_h = _ph
        _map_w = int(_ph * _ext_ratio)
        _map_y = 0
        _map_x = (_pw - _map_w) // 2

    _proj = get_live_project()
    _ms = QgsMapSettings()
    _ms.setLayers(state["analysis"]["visible_layers"])
    _ms.setDestinationCrs(_proj.crs())
    _ms.setExtent(_ext)
    _ms.setBackgroundColor(QColor(255, 255, 255))
    _ms.setOutputSize(QSize(_map_w, _map_h))
    _ms.setOutputDpi(_dpi)

    _png_img = QImage(_pw, _ph, QImage.Format_ARGB32_Premultiplied)
    _png_img.fill(QColor(255, 255, 255).rgba())
    _painter = QPainter(_png_img)
    _painter.translate(_map_x, _map_y)
    _job = QgsMapRendererCustomPainterJob(_ms, _painter)
    _job.start()
    _job.waitForFinished()
    _painter.end()

    _png_img.save(cfg.paths.png_path)
    img_result = 0
    print(f"[export] PNG via direct render: {cfg.paths.png_path} ({_map_w}x{_map_h}@{_map_x},{_map_y})")

    # --- PDF export ---
    exporter = QgsLayoutExporter(layout)
    pdf_settings = QgsLayoutExporter.PdfExportSettings()
    pdf_settings.dpi = cfg.run.export_dpi
    pdf_result = exporter.exportToPdf(cfg.paths.pdf_path, pdf_settings)

    print("====================================")
    print("Export finished")
    print("PNG:", cfg.paths.png_path, "result code =", img_result)
    print("PDF:", cfg.paths.pdf_path, "result code =", pdf_result)
    print("====================================")

    layout_state["auto_export_executed"] = True
    layout_state["png_export_result"] = img_result
    layout_state["pdf_export_result"] = pdf_result
    return state


def export_layout_manual(state: dict) -> dict:
    cfg = state["cfg"]
    layout = ensure_layout_exists(state)
    layout_state = _ensure_layout_state(state)

    from PyQt5.QtGui import QImage, QPainter
    _page = layout.pageCollection().pages()[0]
    _ps = _page.pageSize()
    _dpi = max(cfg.run.export_dpi, 72)
    _pw = int(_ps.width() * _dpi / 25.4)
    _ph = int(_ps.height() * _dpi / 25.4)
    _png_img = QImage(_pw, _ph, QImage.Format_ARGB32_Premultiplied)
    _png_img.fill(QColor(255, 255, 255).rgba())
    _png_painter = QPainter(_png_img)
    _png_painter.setRenderHint(QPainter.Antialiasing, True)
    exporter = QgsLayoutExporter(layout)
    exporter.renderPage(_png_painter, 0)
    _png_painter.end()
    _png_img.save(cfg.paths.png_path)

    pdf_settings = QgsLayoutExporter.PdfExportSettings()
    pdf_settings.dpi = cfg.run.export_dpi
    exporter.exportToPdf(cfg.paths.pdf_path, pdf_settings)

    layout_state["manual_export_executed"] = True
    return state