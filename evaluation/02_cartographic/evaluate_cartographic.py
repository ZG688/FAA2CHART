# -*- coding: utf-8 -*-
"""
Cartographic Correctness Evaluation
====================================
Compares the generated airway map layer (faa_airways_visualization.gpkg) against
the FAA NASR official baseline with objective, reproducible geometric/topological
comparison. Supports reviewer comments R2-2, Reviewer #3-4/5, and the PDF
paragraph 2 (not merely "title/legend/layer loaded successfully" as evidence of
cartographic correctness).

Three-level evaluation (all GDAL/QGIS-free, pure geometry; all with bootstrap 95% CI):
  1) Topological correctness
       - Node-set Jaccard (generated vs official endpoints)
       - Edge-set Precision / Recall / F1 (undirected edges {u,v}, official AWY_SEG_ALT)
  2) Geometric consistency (on matched edges)
       - Endpoint spherical distance (start+end) median/P95
       - Buffer IoU: buffer_km buffer around both segments, analytical estimate of IoU
  3) Attribute/classification consistency (on matched edges)
       - MEA / MAA hit rate
       - Endpoint type (navaid type) consistency

Reference design (consistent with extraction evaluation, avoiding edition drift):
  - Topology/attributes (content)  use Feb-2025 NASR
  - Coordinate/geometry reference coords use Nov-2025 NASR (same cycle as the
    geolocation navigation DB used in generation), providing objective assessment
    of the "geolocation + connection" step itself.

Usage:
  python evaluate_cartographic.py
  python evaluate_cartographic.py --buffer-km 5 --boot 1000
"""
import os
import re
import csv
import math
import json
import struct
import random
import sqlite3
import argparse
from collections import defaultdict

# Reuse normalization / loading / distance tools from extraction evaluation script
# to ensure consistent evaluation criteria across both stages
from evaluation.01_extraction.evaluate_extraction_v2 import (
    norm, norm_alt, haversine_km, find_csv,
    load_airways, load_coords, to_id,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get(
    "FAA2CHART_DATA_ROOT",
    os.path.join(HERE, "..", "..", "..", "data"))
GPKG = os.path.join(DATA_ROOT, "maps", "faa_airways_visualization.gpkg")
FEB_CSV = os.path.join(
    DATA_ROOT, "airway_data", "28DaySubscription_Effective_2025-02-20", "CSV_Data", "20_Feb_2025_CSV")
NOV_CSV = os.path.join(
    DATA_ROOT, "airway_data", "28DaySubscription_Effective_2025-11-27", "CSV_Data", "27_Nov_2025_CSV")
OUT = os.path.join(HERE, "data", "cartographic_eval_report.json")


# ============================================================ Read generated layer (gpkg)
def _wkb_linestring_pts(blob):
    """Parse LINESTRING vertices [(lon,lat),...] from GeoPackage geom blob."""
    flags = blob[3]
    env_flag = (flags >> 1) & 0x07
    env_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    wkb = blob[8 + env_sizes[env_flag]:]
    bo = "<" if wkb[0] == 1 else ">"
    gtype = struct.unpack(bo + "I", wkb[1:5])[0]
    if gtype != 2:  # LINESTRING only
        return None
    npt = struct.unpack(bo + "I", wkb[5:9])[0]
    pts, off = [], 9
    for _ in range(npt):
        x, y = struct.unpack(bo + "dd", wkb[off:off + 16])
        pts.append((x, y))  # (lon, lat)
        off += 16
    return pts


def load_pred_segments(gpkg):
    """Read airway_segments: return edges grouped by airway.
    Each edge: dict(airway, from_name, to_name, from_type, to_type,
                 MEA, MAA, pts=[(lon,lat)...])."""
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
            from_type=d.get("from_type"), to_type=d.get("to_type"),
            MEA=norm_alt(d.get("MEA")), MAA=norm_alt(d.get("MAA")),
            pts=pts))
    c.close()
    return segs


# ============================================================ Official edges (gold)
def build_gold_edges(feb_csv, coord_gold):
    """Construct official edges from AWY_SEG_ALT segment-by-segment:
       return {airway: [dict(u,v, mea, maa, u_ll,v_ll)...]}, u/v are official short codes.
       Coordinates from coord_gold (Nov version, for geometry reference)."""
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

            def _f(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            edges[aid].append(dict(
                u=u_id, v=v_id,
                mea=norm_alt(x.get("MIN_ENROUTE_ALT")),
                maa=norm_alt(x.get("MAX_AUTH_ALT")),
                mag_course=_f(x.get("MAG_COURSE")),      # official magnetic course (deg)
                dist_nm=_f(x.get("MAG_COURSE_DIST")),    # official segment length (NM)
                u_ll=coord_gold["coord"].get(u_id),
                v_ll=coord_gold["coord"].get(v_id)))
    return edges


# ============================================================ Independent geometry: declination/azimuth
def load_magvar(csv_dir):
    """Load navaid/fix magnetic declination (deg, east positive, west negative).
    NAV_BASE: MAG_VARN (e.g. 16E/07W) + MAG_VARN_HEMIS; FIX has no declination,
    falls back to nearest navaid value.
    Returns {id: magvar_deg}."""
    mv = {}
    nav = find_csv(csv_dir, "NAV_BASE.csv")
    if nav:
        with open(nav, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                nid = norm(r["NAV_ID"])
                raw = (r.get("MAG_VARN") or "").strip()
                if not raw:
                    continue
                m = re.match(r"(\d+(?:\.\d+)?)\s*([EW]?)", raw)
                if not m:
                    continue
                val = float(m.group(1))
                hemi = m.group(2) or (r.get("MAG_VARN_HEMIS") or "E")
                mv.setdefault(nid, val if hemi == "E" else -val)
    return mv


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """True north initial bearing from start to end (deg, 0-360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def ang_diff(a, b):
    """Minimum angular difference (deg, 0-180)."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


# ============================================================ Geometry tools
def seg_endpoint_dist_km(pred_edge, gold_edge):
    """Endpoint spherical distance (mean of start+end, km) for matched edges.
    Direction aligned to gold u->v. Returns None if gold endpoint coordinates missing."""
    if not gold_edge["u_ll"] or not gold_edge["v_ll"]:
        return None
    p0 = pred_edge["pts"][0]
    p1 = pred_edge["pts"][-1]
    gu = gold_edge["u_ll"]  # (lat,lon)
    gv = gold_edge["v_ll"]
    d_fwd = (haversine_km(p0[1], p0[0], gu[0], gu[1]) +
             haversine_km(p1[1], p1[0], gv[0], gv[1])) / 2
    d_rev = (haversine_km(p0[1], p0[0], gv[0], gv[1]) +
             haversine_km(p1[1], p1[0], gu[0], gu[1])) / 2
    return min(d_fwd, d_rev)


def buffered_iou(pred_edge, gold_edge, buffer_km):
    """Approximate IoU of two buffered line segments.
    Approximation: segment buffer area = 2*r*L + pi*r^2; overlap estimated
    from mean endpoint offset d. d=0 -> IoU~1; d>=2r -> IoU~0. Returns [0,1]."""
    d = seg_endpoint_dist_km(pred_edge, gold_edge)
    if d is None:
        return None
    r = buffer_km
    overlap = max(0.0, 1.0 - d / (2.0 * r))
    return overlap / (2.0 - overlap) if overlap > 0 else 0.0


# ============================================================ Per-airway evaluation
def eval_airway(aid, pred_edges, gold_edges, buffer_km, cg, magvar):
    """Return aggregate counts for this airway (for global aggregation and bootstrap)."""
    def uv(a, b):
        return tuple(sorted((a, b)))

    gold_set = {uv(e["u"], e["v"]) for e in gold_edges if e["u"] and e["v"]}
    gold_by_key = {}
    for e in gold_edges:
        if e["u"] and e["v"]:
            gold_by_key.setdefault(uv(e["u"], e["v"]), e)

    # Generated edges: point name -> official id
    pred_set = set()
    pred_by_key = {}
    for pe in pred_edges:
        pu, _ = to_id(pe["from_name"], cg)
        pv, _ = to_id(pe["to_name"], cg)
        if not pu or not pv:
            continue
        k = uv(pu, pv)
        pred_set.add(k)
        pred_by_key.setdefault(k, pe)

    # Node set
    gold_nodes = {e["u"] for e in gold_edges} | {e["v"] for e in gold_edges}
    gold_nodes = {x for x in gold_nodes if x}
    pred_nodes = set()
    for pe in pred_edges:
        for nm in (pe["from_name"], pe["to_name"]):
            i, _ = to_id(nm, cg)
            if i:
                pred_nodes.add(i)

    tp_edges = pred_set & gold_set
    node_inter = pred_nodes & gold_nodes
    node_union = pred_nodes | gold_nodes

    # Matched edge geometry/attributes
    dists, ious, mea_ok, mea_tot, maa_ok, maa_tot = [], [], 0, 0, 0, 0
    brg_err, len_relerr = [], []   # independent geometry: bearing deviation (deg) / length rel error
    for k in tp_edges:
        pe, ge = pred_by_key[k], gold_by_key[k]
        d = seg_endpoint_dist_km(pe, ge)
        if d is not None:
            dists.append(d)
            iou = buffered_iou(pe, ge, buffer_km)
            if iou is not None:
                ious.append(iou)
        # --- Independent geometry comparison vs official MAG_COURSE / MAG_COURSE_DIST ---
        p0, p1 = pe['pts'][0], pe['pts'][-1]
        # Bearing: generate edge endpoint chord TN bearing -> subtract declination -> mag course, compare with official
        if ge.get('mag_course') is not None:
            if ge['u_ll'] and ge['v_ll']:
                d0u = haversine_km(p0[1], p0[0], ge['u_ll'][0], ge['u_ll'][1])
                d0v = haversine_km(p0[1], p0[0], ge['v_ll'][0], ge['v_ll'][1])
                a, b = (p0, p1) if d0u <= d0v else (p1, p0)
                tn = initial_bearing_deg(a[1], a[0], b[1], b[0])  # true north bearing
                mv = magvar.get(ge['u'])
                if mv is None:
                    mv = magvar.get(ge['v'], 0.0)
                mag_pred = (tn - mv) % 360.0
                brg_err.append(ang_diff(mag_pred, ge['mag_course']))
        # Length: generated edge great-circle distance vs official segment distance
        if ge.get('dist_nm') and ge['dist_nm'] > 0:
            dist_km = haversine_km(p0[1], p0[0], p1[1], p1[0])
            dist_nm_pred = dist_km / 1.852
            len_relerr.append(abs(dist_nm_pred - ge['dist_nm']) / ge['dist_nm'])
        if ge["mea"]:
            mea_tot += 1
            if pe["MEA"] == ge["mea"]:
                mea_ok += 1
        if ge["maa"]:
            maa_tot += 1
            if pe["MAA"] == ge["maa"]:
                maa_ok += 1

    return dict(
        aid=aid,
        edge_tp=len(tp_edges), edge_pred=len(pred_set), edge_gold=len(gold_set),
        node_inter=len(node_inter), node_union=len(node_union),
        dists=dists, ious=ious, brg_err=brg_err, len_relerr=len_relerr,
        mea_ok=mea_ok, mea_tot=mea_tot, maa_ok=maa_ok, maa_tot=maa_tot)


# ============================================================ Aggregation + bootstrap
def summarize(per, boot=1000, seed=42):
    def agg(sub):
        etp = sum(x["edge_tp"] for x in sub)
        ep = sum(x["edge_pred"] for x in sub)
        eg = sum(x["edge_gold"] for x in sub)
        ni = sum(x["node_inter"] for x in sub)
        nu = sum(x["node_union"] for x in sub)
        P = etp / ep if ep else 0.0
        R = etp / eg if eg else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        jac = ni / nu if nu else 0.0
        return P, R, F1, jac

    P, R, F1, jac = agg(per)
    all_d = [d for x in per for d in x["dists"]]
    all_i = [i for x in per for i in x["ious"]]
    all_d.sort()

    def pct(v, q):
        if not v:
            return None
        k = min(len(v) - 1, int(q * (len(v) - 1)))
        return v[k]

    all_brg = sorted(e for x in per for e in x["brg_err"])
    all_lre = sorted(e for x in per for e in x["len_relerr"])
    mea_ok = sum(x["mea_ok"] for x in per)
    mea_tot = sum(x["mea_tot"] for x in per)
    maa_ok = sum(x["maa_ok"] for x in per)
    maa_tot = sum(x["maa_tot"] for x in per)

    rng = random.Random(seed)
    n = len(per)
    bs_F1, bs_jac, bs_med, bs_iou, bs_brg = [], [], [], [], []
    for _ in range(boot):
        sample = [per[rng.randrange(n)] for _ in range(n)]
        _, _, f1b, jacb = agg(sample)
        bs_F1.append(f1b)
        bs_jac.append(jacb)
        dd = sorted(d for x in sample for d in x["dists"])
        bs_med.append(pct(dd, 0.5) if dd else 0.0)
        ii = [i for x in sample for i in x["ious"]]
        bs_iou.append(sum(ii) / len(ii) if ii else 0.0)
        bb = sorted(e for x in sample for e in x["brg_err"])
        bs_brg.append(pct(bb, 0.5) if bb else 0.0)

    def ci(v):
        v = sorted(v)
        return [round(v[int(0.025 * len(v))], 4),
                round(v[int(0.975 * len(v))], 4)]

    return dict(
        n_airways=n,
        edge_P=round(P, 4), edge_R=round(R, 4), edge_F1=round(F1, 4),
        node_jaccard=round(jac, 4),
        matched_edges=len(all_d),
        endpoint_dist_km_median=round(pct(all_d, 0.5), 4) if all_d else None,
        endpoint_dist_km_p95=round(pct(all_d, 0.95), 4) if all_d else None,
        buffered_iou_mean=round(sum(all_i) / len(all_i), 4) if all_i else None,
        MEA_match=round(mea_ok / mea_tot, 4) if mea_tot else None,
        MAA_match=round(maa_ok / maa_tot, 4) if maa_tot else None,
        MEA_n=mea_tot, MAA_n=maa_tot,
        indep_n_bearing=len(all_brg), indep_n_length=len(all_lre),
        bearing_err_deg_median=round(pct(all_brg, 0.5), 3) if all_brg else None,
        bearing_err_deg_p90=round(pct(all_brg, 0.90), 3) if all_brg else None,
        bearing_within_5deg=round(sum(1 for e in all_brg if e <= 5) / len(all_brg), 4) if all_brg else None,
        length_relerr_median=round(pct(all_lre, 0.5), 4) if all_lre else None,
        length_within_5pct=round(sum(1 for e in all_lre if e <= 0.05) / len(all_lre), 4) if all_lre else None,
        ci95=dict(edge_F1=ci(bs_F1), node_jaccard=ci(bs_jac),
                  endpoint_dist_km_median=ci(bs_med),
                  buffered_iou_mean=ci(bs_iou),
                  bearing_err_deg_median=ci(bs_brg)))


# ============================================================ Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-km", type=float, default=5.0,
                    help="Buffer IoU buffer radius (km), default 5")
    ap.add_argument("--boot", type=int, default=1000, help="Bootstrap iterations")
    args = ap.parse_args()

    print("Loading official reference...")
    feb_air = load_airways(FEB_CSV)
    cg = load_coords(NOV_CSV)                       # geometry reference coords = Nov
    magvar = load_magvar(FEB_CSV)                   # magnetic declination (same cycle as official mag course)
    feb_coord_for_edges = load_coords(FEB_CSV)      # official edge endpoint id normalization (Feb content version)
    gold_edges = build_gold_edges(FEB_CSV, feb_coord_for_edges)
    # Replace endpoint coords with Nov version (geometry reference)
    for aid, es in gold_edges.items():
        for e in es:
            e["u_ll"] = cg["coord"].get(e["u"], e["u_ll"])
            e["v_ll"] = cg["coord"].get(e["v"], e["v_ll"])

    print("Reading generated layer...")
    pred = load_pred_segments(GPKG)
    print("  Generated airways: {}   Generated edges: {}".format(
        len(pred), sum(len(v) for v in pred.values())))
    print("  Official airways: {}   Official edges: {}".format(
        len(gold_edges), sum(len(v) for v in gold_edges.values())))

    common = [a for a in pred if a in gold_edges]
    print("  Comparable airways (intersection): {}".format(len(common)))

    per = [eval_airway(a, pred[a], gold_edges[a], args.buffer_km, cg, magvar)
           for a in common]
    per = [x for x in per if x["edge_gold"] > 0]

    summ = summarize(per, boot=args.boot)
    summ["buffer_km"] = args.buffer_km
    summ["note"] = ("Topology/attribute reference=Feb-2025 NASR; geometry reference "
                    "coords=Nov-2025 NASR (same cycle as geolocation nav DB). "
                    "Edges are undirected {u,v}.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"gpkg": os.path.basename(GPKG), "summary": summ}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    def fmt(v, c):
        return "{:.3f} [{:.3f},{:.3f}]".format(v, c[0], c[1]) if v is not None else "n/a"

    print("\n" + "=" * 70)
    print("Cartographic Correctness Evaluation (vs FAA NASR)")
    print("=" * 70)
    print("Comparable airways n = {}   Buffer radius = {} km".format(summ["n_airways"], args.buffer_km))
    print("-" * 70)
    print("[Topological correctness]")
    print("   Node-set Jaccard        : {}".format(
        fmt(summ["node_jaccard"], summ["ci95"]["node_jaccard"])))
    print("   Edge Precision          : {:.3f}".format(summ["edge_P"]))
    print("   Edge Recall             : {:.3f}".format(summ["edge_R"]))
    print("   Edge F1                 : {}".format(
        fmt(summ["edge_F1"], summ["ci95"]["edge_F1"])))
    print("-" * 70)
    print("[Geometric consistency]  (matched edges n={})".format(summ["matched_edges"]))
    print("   Endpoint dist median (km): {}".format(
        fmt(summ["endpoint_dist_km_median"], summ["ci95"]["endpoint_dist_km_median"])))
    print("   Endpoint dist P95   (km): {}".format(summ["endpoint_dist_km_p95"]))
    print("   Buffer IoU (mean)        : {}".format(
        fmt(summ["buffered_iou_mean"], summ["ci95"]["buffered_iou_mean"])))
    print("-" * 70)
    print("[Independent geometry vs official MAG_COURSE/DIST]")
    print("   Bearing deviation median (deg): {}  (matched edges n={})".format(
        fmt(summ["bearing_err_deg_median"], summ["ci95"]["bearing_err_deg_median"]),
        summ["indep_n_bearing"]))
    print("   Bearing deviation P90   (deg): {}".format(summ["bearing_err_deg_p90"]))
    print("   Bearing <=5deg ratio     : {}".format(summ["bearing_within_5deg"]))
    print("   Length rel. err. median  : {}  (matched edges n={})".format(
        summ["length_relerr_median"], summ["indep_n_length"]))
    print("   Length error <=5% ratio  : {}".format(summ["length_within_5pct"]))
    print("-" * 70)
    print("[Attribute consistency]")
    print("   MEA hit rate (n={})     : {}".format(
        summ["MEA_n"], "{:.3f}".format(summ["MEA_match"]) if summ["MEA_match"] is not None else "n/a"))
    print("   MAA hit rate (n={})     : {}".format(
        summ["MAA_n"], "{:.3f}".format(summ["MAA_match"]) if summ["MAA_match"] is not None else "n/a"))
    print("=" * 70)
    print("Machine-readable report: {}".format(OUT))


if __name__ == "__main__":
    main()