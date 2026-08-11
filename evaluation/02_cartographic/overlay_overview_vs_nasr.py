# -*- coding: utf-8 -*-
"""
Low-altitude Airway Overview Map -- Generated vs. Official NASR Overlay Comparison
===================================================================================
Addresses reviewer comment #2 (map validation): The overview map corresponds to the
official IFR low-altitude airway chart, so we provide an overlay visualization of
"generated airway geometry vs. official NASR airway geometry" as intuitive,
reproducible corroboration of quantitative results (e.g., length median rel-err 0.18%).

Fully reproducible, no manual chart screenshot needed:
  - Generated geometry: read faa_airways_visualization.gpkg airway_segments (WKB LINESTRING).
  - Official geometry: AWY_SEG_ALT segment-by-segment records, endpoint coordinates from
    Nov-2025 NASR (same cycle as the nav DB used for geolocation), forming official reference
    polylines.
  - Stratified sampling of N airways (by official sequence length: short/medium/long),
    side-by-side overlay: blue solid = generated, red dashed = official NASR;
    subplot title shows airway code.

Output (written to data/):
  overlay_overview_samples.png     Small-sample grid overlay
  overlay_overview_manifest.json   Sampling manifest + per-route quantitative metrics

Pure geometry + matplotlib, reuses evaluate_extraction_v2 loading/normalization tools.
"""
import os
import sys
import csv
import struct
import sqlite3
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# All figures use Times New Roman (journal typesetting)
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 10

# Page width (cm -> inch); generate figures at this physical width, paste at 100%
PAGE_W_CM = 17.0
CM = 1.0 / 2.54

HERE = os.path.dirname(os.path.abspath(__file__))
# evaluate_extraction_v2 is in 01_extraction
EVAL_DIR = os.path.normpath(os.path.join(HERE, "..", "01_extraction"))
sys.path.insert(0, EVAL_DIR)
from evaluate_extraction_v2 import (  # noqa: E402
    norm, norm_alt, haversine_km, find_csv, load_coords, to_id as _to_id,
)

_TOID_CACHE = {}


def to_id(name, cg):
    key = norm(name)
    if key in _TOID_CACHE:
        return _TOID_CACHE[key]
    r = _to_id(name, cg)
    _TOID_CACHE[key] = r
    return r


DATA_ROOT = os.environ.get(
    "FAA2CHART_DATA_ROOT",
    os.path.join(HERE, "..", "..", "..", "data"))
GPKG = os.path.join(DATA_ROOT, "maps", "faa_airways_visualization.gpkg")
FEB_CSV = os.path.join(DATA_ROOT, "airway_data", "28DaySubscription_Effective_2025-02-20", "CSV_Data", "20_Feb_2025_CSV")
NOV_CSV = os.path.join(DATA_ROOT, "airway_data", "28DaySubscription_Effective_2025-11-27", "CSV_Data", "27_Nov_2025_CSV")
OUT_DIR = os.path.join(HERE, "data")

N_SAMPLES = 12
SEED = 42


# ============================================================ Read generated layer
def _wkb_linestring_pts(blob):
    flags = blob[3]
    env_flag = (flags >> 1) & 0x07
    env_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    wkb = blob[8 + env_sizes[env_flag]:]
    bo = "<" if wkb[0] == 1 else ">"
    gtype = struct.unpack(bo + "I", wkb[1:5])[0]
    if gtype != 2:
        return None
    npt = struct.unpack(bo + "I", wkb[5:9])[0]
    pts, off = [], 9
    for _ in range(npt):
        x, y = struct.unpack(bo + "dd", wkb[off:off + 16])
        pts.append((x, y))
        off += 16
    return pts


def load_pred_segments(gpkg):
    c = sqlite3.connect(gpkg)
    cols = [x[1] for x in c.execute("PRAGMA table_info('airway_segments')")]
    segs = defaultdict(list)
    for row in c.execute("select * from airway_segments"):
        d = dict(zip(cols, row))
        pts = _wkb_linestring_pts(d["geom"]) if d.get("geom") else None
        if not pts:
            try:
                pts = [(float(d["from_lon"]), float(d["from_lat"])),
                       (float(d["to_lon"]), float(d["to_lat"]))]
            except (TypeError, ValueError):
                continue
        segs[norm(d["airway"])].append(dict(
            airway=norm(d["airway"]),
            from_name=d.get("from_name"), to_name=d.get("to_name"),
            pts=pts))
    c.close()
    return segs


# ============================================================ Official edges (gold)
def build_gold_edges(feb_csv, coord_gold):
    seg = find_csv(feb_csv, "AWY_SEG_ALT.csv")
    rows_by = defaultdict(list)
    with open(seg, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows_by[norm(r["AWY_ID"])].append(r)

    edges = defaultdict(list)
    for aid, rows in rows_by.items():
        def sk(x):
            try:
                return int(x.get("POINT_SEQ") or 0)
            except ValueError:
                return 0
        rows.sort(key=sk)
        for x in rows:
            u = norm(x.get("FROM_POINT"))
            v = norm(x.get("TO_POINT"))
            if not u or not v:
                continue
            u_id, _ = to_id(u, coord_gold)
            v_id, _ = to_id(v, coord_gold)
            edges[aid].append(dict(
                u=u_id, v=v_id,
                u_ll=coord_gold["coord"].get(u_id),
                v_ll=coord_gold["coord"].get(v_id)))
    return edges


def uv(a, b):
    return (a, b) if a <= b else (b, a)


def endpoint_dist_km(pred_edge, gold_edge):
    p = pred_edge["pts"]
    p0 = (p[0][1], p[0][0])
    p1 = (p[-1][1], p[-1][0])
    gu, gv = gold_edge["u_ll"], gold_edge["v_ll"]
    d_straight = haversine_km(p0[0], p0[1], gu[0], gu[1]) + haversine_km(p1[0], p1[1], gv[0], gv[1])
    d_swap = haversine_km(p0[0], p0[1], gv[0], gv[1]) + haversine_km(p1[0], p1[1], gu[0], gu[1])
    return min(d_straight, d_swap) / 2.0


# ============================================================ Sampling & metrics
def route_metrics(pred_segs, gold_edges, aid, cg):
    preds = pred_segs.get(aid, [])
    golds = gold_edges.get(aid, [])
    if not preds or not golds:
        return None
    gold_set = {}
    for ge in golds:
        if ge["u_ll"] and ge["v_ll"]:
            gold_set[uv(ge["u"], ge["v"])] = ge
    dists = []
    matched = 0
    for pe in preds:
        u_id, _ = to_id(pe["from_name"], cg)
        v_id, _ = to_id(pe["to_name"], cg)
        key = uv(u_id, v_id)
        if key in gold_set:
            matched += 1
            dists.append(endpoint_dist_km(pe, gold_set[key]))
    dists.sort()
    med = dists[len(dists) // 2] if dists else None
    return dict(airway=aid, n_pred=len(preds), n_gold=len(golds),
                n_matched=matched, endpoint_dist_median_km=med,
                n_gold_pts=len({p for e in golds for p in (e["u"], e["v"])}))


def stratified_sample(candidates, n, seed=SEED):
    import random
    rng = random.Random(seed)
    cand = sorted(candidates, key=lambda c: c["n_gold_pts"])
    if len(cand) <= n:
        return cand
    k = len(cand)
    lo = cand[:k // 3]
    mid = cand[k // 3:2 * k // 3]
    hi = cand[2 * k // 3:]
    per = n // 3
    pick = []
    for grp in (lo, mid, hi):
        pool = grp[:]
        rng.shuffle(pool)
        pick.extend(pool[:per])
    rest = [c for c in cand if c not in pick]
    rng.shuffle(rest)
    pick.extend(rest[:n - len(pick)])
    pick.sort(key=lambda c: c["n_gold_pts"])
    return pick


# ============================================================ Plotting
def nlon(lon):
    """Date-line normalization: positive longitudes (Alaska west/Guam) shift -360."""
    return lon - 360.0 if lon > 0 else lon


def plot_route(ax, pred_segs, gold_edges, m):
    aid = m["airway"]
    for ge in gold_edges.get(aid, []):
        if ge["u_ll"] and ge["v_ll"]:
            xs = [nlon(ge["u_ll"][1]), nlon(ge["v_ll"][1])]
            ys = [ge["u_ll"][0], ge["v_ll"][0]]
            ax.plot(xs, ys, color="#d62728", linestyle="--", linewidth=2.2,
                    zorder=2, solid_capstyle="round")
    for pe in pred_segs.get(aid, []):
        xs = [nlon(p[0]) for p in pe["pts"]]
        ys = [p[1] for p in pe["pts"]]
        ax.plot(xs, ys, color="#1f77b4", linestyle="-", linewidth=1.8,
                alpha=0.9, zorder=3)
    ax.set_title(aid, fontsize=10, fontweight="bold", pad=3)
    ax.set_aspect("auto")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.tick_params(labelsize=8, length=2, pad=1)


def main():
    global COORD_GOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    ap.add_argument("--codes", type=str, default="",
                    help="Comma-separated airway codes (optional, overrides stratified sampling)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading Nov-2025 coords (geometry reference)...")
    COORD_GOLD = load_coords(NOV_CSV)
    print("Loading generated segments from gpkg...")
    pred_segs = load_pred_segments(GPKG)
    print("Building official NASR edges (Feb AWY_SEG_ALT + Nov coords)...")
    gold_edges = build_gold_edges(FEB_CSV, COORD_GOLD)

    common = []
    for aid in set(pred_segs) & set(gold_edges):
        golds = gold_edges[aid]
        gold_pts = {p for e in golds for p in (e["u"], e["v"]) if e["u_ll"] and e["v_ll"]}
        n_gold_e = sum(1 for e in golds if e["u_ll"] and e["v_ll"])
        n_pred_e = len(pred_segs.get(aid) or [])
        if pred_segs.get(aid) and len(gold_pts) >= 3:
            common.append(dict(airway=aid, n_gold_pts=len(gold_pts),
                               n_gold_e=n_gold_e, n_pred_e=n_pred_e))
    print("Comparable airways:", len(common))

    if args.codes.strip():
        want = [norm(c) for c in args.codes.split(",") if c.strip()]
        by = {m["airway"]: m for m in common}
        picks = [by[c] for c in want if c in by]
    else:
        MIN_COVER = 0.6
        full = [c for c in common
                if c["n_gold_e"] > 0 and c["n_pred_e"] >= MIN_COVER * c["n_gold_e"]]
        print("Reasonably complete airways (>= {:.0%} coverage):".format(MIN_COVER), len(full))
        picks = stratified_sample(full, args.n)

    sample = []
    for c in picks:
        m = route_metrics(pred_segs, gold_edges, c["airway"], COORD_GOLD)
        if m:
            sample.append(m)

    # ---- Small-sample grid overlay ----
    n = len(sample)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig_w = PAGE_W_CM * CM
    panel_w = fig_w / ncol
    panel_h = panel_w * 0.85
    fig_h = panel_h * nrow + 0.9
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, m in enumerate(sample):
        plot_route(axes[i], pred_segs, gold_edges, m)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=1.8, label="Generated (this work)"),
        plt.Line2D([0], [0], color="#d62728", lw=2.2, ls="--", label="Official FAA NASR"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Generated airway geometry vs. official FAA NASR",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0.045, 1, 0.965])
    p_samples = os.path.join(OUT_DIR, "overlay_overview_samples.png")
    fig.savefig(p_samples, dpi=300)
    plt.close(fig)
    print("Saved:", p_samples)

    # ---- manifest ----
    import json
    meds = sorted(m["endpoint_dist_median_km"] for m in sample
                  if m["endpoint_dist_median_km"] is not None)
    overall_med = meds[len(meds) // 2] if meds else None
    manifest = {
        "meta": {
            "gpkg": os.path.basename(GPKG),
            "official_topology": "Feb-2025 AWY_SEG_ALT",
            "official_coords": "Nov-2025 NASR (same cycle as localization nav DB)",
            "n_comparable_airways": len(common),
            "n_sampled": len(sample),
            "sampled_matched_endpoint_dist_median_km": overall_med,
            "note": ("Overlay compares generated airway geometry against official NASR "
                     "geometry; provides a visual, reproducible corroboration of the "
                     "independent geometric validation."),
        },
        "sample": sample,
        "figures": [os.path.basename(p_samples)],
    }
    p_man = os.path.join(OUT_DIR, "overlay_overview_manifest.json")
    with open(p_man, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Saved:", p_man)

    print("\n==================== SUMMARY ====================")
    print("Comparable airways:", len(common))
    print("Sampled routes:", [m["airway"] for m in sample])
    print("Overall matched-edge endpoint distance median (km):", overall_med)


if __name__ == "__main__":
    main()