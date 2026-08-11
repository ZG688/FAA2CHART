# -*- coding: utf-8 -*-
"""
Interstate Connectivity Thematic Map -- Result Reproducibility & State-Attribution
Consistency Evaluation
=============================================================================
Addresses reviewer comment #2 (map validation): The thematic map (Interstate
connectivity map) has no official counterpart, so instead of visual comparison
with official charts, we validate (1) that its values can be deterministically
reproduced from source data (traceability), and (2) the "waypoint state
attribution" field driving the thematic map is consistent with independent
geometric judgment (point-in-polygon coordinate overlay).

Three-part evaluation:
  A. Score reproducibility: Independently reproduce the connectivity score formula
     score = 2 x direct_interstate_adjacent_segments + airway_co_occurrence
     Confirming that the thematic map's state scores can be precisely and
     deterministically derived from the source JSON.
  B. Region-label geometric validation: For waypoints with both valid coordinates
     and a state label, use Natural Earth state boundaries with point-in-polygon
     to independently determine the state, compare with the extracted region field,
     and report agreement rate + 1,000 bootstrap 95% CI. This is independent
     evidence of thematic map correctness (the map depends entirely on the region field).
  C. Propagation: Recompute the target state's connectivity scores using the
     geometrically-determined state, compare ranking consistency (Spearman) and
     Top-K overlap with the original scores, assessing the actual impact of
     region errors on the thematic map conclusion.

Pure standard library implementation (struct .shp/.dbf parsing + ray-casting PIP),
no GDAL/QGIS/geopandas dependency.
"""
import os
import json
import struct
import random

DATA_ROOT = os.environ.get(
    "FAA2CHART_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))
JSON_PATH = os.path.join(DATA_ROOT, "airway_data", "text", "parsed_routes_newnew_5_Claude4.5_updated_merged.json")
SHP = os.path.join(DATA_ROOT, "us_vector_data", "ne_10m_admin_1_states_provinces", "ne_10m_admin_1_states_provinces.shp")
DBF = SHP[:-4] + ".dbf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "connectivity_consistency_report.json")

TARGET_STATE = "CA"
LABEL_TOP_K = 5
N_BOOT = 1000
SEED = 42

US_STATE_ABBR = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC",
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
    "DISTRICT OF COLUMBIA":"DC","WASHINGTON, D.C.":"DC","WASHINGTON DC":"DC",
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


# =========================================================
# Shapefile / DBF pure stdlib parsing
# =========================================================
def read_dbf(path):
    """Return (fields, records). records is a list of dicts."""
    with open(path, "rb") as f:
        header = f.read(32)
        num_rec = struct.unpack("<I", header[4:8])[0]
        header_size = struct.unpack("<H", header[8:10])[0]
        rec_size = struct.unpack("<H", header[10:12])[0]
        n_fields = (header_size - 33) // 32
        fields = []
        for _ in range(n_fields):
            fd = f.read(32)
            name = fd[:11].split(b"\x00")[0].decode("latin-1")
            flen = fd[16]
            fields.append((name, flen))
        f.read(1)  # header terminator 0x0D
        records = []
        for _ in range(num_rec):
            rec = f.read(rec_size)
            if not rec or rec[:1] == b"\x1a":
                break
            vals = {}
            off = 1
            for name, flen in fields:
                raw = rec[off:off + flen]
                vals[name] = raw.decode("latin-1").replace("\x00", "").strip()
                off += flen
            records.append(vals)
    return [fn for fn, _ in fields], records


def read_shp_polygons(path):
    """Parse shapefile Polygon (type 5). Return list of parts per record:
    [ [ (x,y),... ], ... ] (one ring per part). Index aligned with dbf records."""
    with open(path, "rb") as f:
        data = f.read()
    shapes = []
    pos = 100
    n = len(data)
    while pos < n:
        rec_num, content_len = struct.unpack(">II", data[pos:pos + 8])
        pos += 8
        rec_start = pos
        shape_type = struct.unpack("<I", data[pos:pos + 4])[0]
        if shape_type == 5:  # Polygon
            num_parts = struct.unpack("<I", data[pos + 36:pos + 40])[0]
            num_points = struct.unpack("<I", data[pos + 40:pos + 44])[0]
            p = pos + 44
            parts = list(struct.unpack("<%dI" % num_parts, data[p:p + 4 * num_parts]))
            p += 4 * num_parts
            coords = struct.unpack("<%dd" % (2 * num_points), data[p:p + 16 * num_points])
            rings = []
            for i in range(num_parts):
                start = parts[i]
                end = parts[i + 1] if i + 1 < num_parts else num_points
                ring = [(coords[2 * j], coords[2 * j + 1]) for j in range(start, end)]
                rings.append(ring)
            shapes.append(rings)
        else:
            shapes.append([])
        pos = rec_start + content_len * 2
    return shapes


def point_in_ring(x, y, ring):
    """Ray-casting: is point inside a single ring?"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def point_in_polygon(x, y, rings):
    """Multi-ring polygon: odd count of rings hit = inside (approximate hole handling)."""
    cnt = 0
    for ring in rings:
        if point_in_ring(x, y, ring):
            cnt += 1
    return cnt % 2 == 1


def ring_bbox(rings):
    xs = [p[0] for ring in rings for p in ring]
    ys = [p[1] for ring in rings for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


# =========================================================
# State boundary index (US 50 states + DC)
# =========================================================
def build_state_index():
    fields, recs = read_dbf(DBF)
    shapes = read_shp_polygons(SHP)
    idx = []
    for rec, rings in zip(recs, shapes):
        if rec.get("adm0_a3") != "USA":
            continue
        if not rings:
            continue
        abbr = normalize_state_abbr(rec.get("postal") or rec.get("iso_3166_2"))
        if abbr is None:
            abbr = normalize_state_abbr(rec.get("name"))
        if abbr not in US_STATE_ABBR:
            continue
        idx.append((abbr, rings, ring_bbox(rings)))
    return idx


def locate_state(lat, lon, idx):
    """Given (lat, lon), return state abbreviation, or None if not found."""
    x, y = lon, lat
    for abbr, rings, (xmin, ymin, xmax, ymax) in idx:
        if x < xmin or x > xmax or y < ymin or y > ymax:
            continue
        if point_in_polygon(x, y, rings):
            return abbr
    return None


# =========================================================
# Connectivity score (independent reproduction of map script formula)
# =========================================================
def compute_scores(routes, region_getter):
    """region_getter(point) -> state abbreviation or None. Returns {state: {...}} and diagnostics."""
    metrics = {}

    def ensure(st):
        if st not in metrics:
            metrics[st] = {"direct_segment_count": 0, "airway_codes": set(), "airway_count": 0, "score": 0}

    diag = {"total_adjacent_pairs": 0, "valid_adjacent_pairs": 0,
            "skipped_missing": 0, "target_direct_segments": 0}

    for airway in routes:
        code = airway.get("airway_code", "UNKNOWN")
        pts = airway.get("airway_point", []) or []

        states_in_airway = set()
        for p in pts:
            reg = region_getter(p)
            pos = p.get("position") or [None, None]
            lat = clean_coord(pos[0])
            lon = clean_coord(pos[1])
            if reg in US_STATE_ABBR and lat is not None and lon is not None:
                states_in_airway.add(reg)

        if TARGET_STATE in states_in_airway:
            for st in states_in_airway:
                if st == TARGET_STATE:
                    continue
                ensure(st)
                metrics[st]["airway_codes"].add(code)

        for i in range(len(pts) - 1):
            diag["total_adjacent_pairs"] += 1
            p1, p2 = pts[i], pts[i + 1]
            reg1 = region_getter(p1)
            reg2 = region_getter(p2)
            pos1 = p1.get("position") or [None, None]
            pos2 = p2.get("position") or [None, None]
            lat1, lon1 = clean_coord(pos1[0]), clean_coord(pos1[1])
            lat2, lon2 = clean_coord(pos2[0]), clean_coord(pos2[1])
            if None in (lat1, lon1, lat2, lon2):
                diag["skipped_missing"] += 1
                continue
            if reg1 not in US_STATE_ABBR or reg2 not in US_STATE_ABBR:
                continue
            diag["valid_adjacent_pairs"] += 1
            if reg1 != reg2 and TARGET_STATE in {reg1, reg2}:
                other = reg2 if reg1 == TARGET_STATE else reg1
                ensure(other)
                metrics[other]["direct_segment_count"] += 1
                diag["target_direct_segments"] += 1

    for st, m in metrics.items():
        m["airway_count"] = len(m["airway_codes"])
        m["score"] = 2 * m["direct_segment_count"] + m["airway_count"]
        m["airway_codes"] = sorted(m["airway_codes"])
    return metrics, diag


def rows_from_metrics(metrics, valid_states):
    rows = []
    for st, m in metrics.items():
        if st not in valid_states:
            continue
        if m["score"] <= 0:
            continue
        rows.append({"state": st, "direct_segment_count": m["direct_segment_count"],
                     "airway_count": m["airway_count"], "score": m["score"]})
    rows.sort(key=lambda x: (-x["score"], x["state"]))
    return rows


def spearman(rank_a, rank_b):
    """Spearman rank correlation on common states."""
    common = sorted(set(rank_a) & set(rank_b))
    n = len(common)
    if n < 2:
        return None
    d2 = sum((rank_a[s] - rank_b[s]) ** 2 for s in common)
    return 1 - 6 * d2 / (n * (n * n - 1))


def bootstrap_ci(flags, n_boot=N_BOOT, seed=SEED):
    if not flags:
        return (None, None)
    rng = random.Random(seed)
    n = len(flags)
    means = []
    for _ in range(n_boot):
        s = sum(flags[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (round(lo, 4), round(hi, 4))


# =========================================================
# MAIN
# =========================================================
def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        routes = json.load(f)

    # ---- A. Score reproducibility ----
    metrics_ext, diag_ext = compute_scores(routes, lambda p: normalize_state_abbr(p.get("region")))
    valid_states = set(US_STATE_ABBR)
    rows_ext = rows_from_metrics(metrics_ext, valid_states)

    # ---- B. Region geometric validation ----
    print("Building state boundary index (pure-Python shapefile parse)...")
    idx = build_state_index()
    print("US state polygons indexed:", len(idx))

    flags = []
    n_pts_total = 0
    n_pts_labeled = 0
    n_pts_coord = 0
    n_eval = 0
    n_geo_none = 0
    mismatches = []
    cache = {}
    for airway in routes:
        for p in airway.get("airway_point", []) or []:
            n_pts_total += 1
            reg = normalize_state_abbr(p.get("region"))
            pos = p.get("position") or [None, None]
            lat = clean_coord(pos[0])
            lon = clean_coord(pos[1])
            if reg is not None:
                n_pts_labeled += 1
            if lat is not None and lon is not None:
                n_pts_coord += 1
            if reg is None or lat is None or lon is None:
                continue
            key = (round(lat, 5), round(lon, 5))
            if key in cache:
                geo = cache[key]
            else:
                geo = locate_state(lat, lon, idx)
                cache[key] = geo
            if geo is None:
                n_geo_none += 1
                continue
            n_eval += 1
            ok = 1 if geo == reg else 0
            flags.append(ok)
            if not ok and len(mismatches) < 60:
                mismatches.append({"name": p.get("name"), "extracted_region": reg,
                                   "geometric_state": geo, "lat": lat, "lon": lon})

    agree = sum(flags) / len(flags) if flags else None
    ci = bootstrap_ci(flags)

    # ---- C. Propagation: recompute scores with geometric states ----
    def geo_region_getter(p):
        pos = p.get("position") or [None, None]
        lat = clean_coord(pos[0])
        lon = clean_coord(pos[1])
        if lat is None or lon is None:
            return None
        key = (round(lat, 5), round(lon, 5))
        if key in cache:
            return cache[key]
        g = locate_state(lat, lon, idx)
        cache[key] = g
        return g

    metrics_geo, diag_geo = compute_scores(routes, geo_region_getter)
    rows_geo = rows_from_metrics(metrics_geo, valid_states)

    rank_ext = {r["state"]: i + 1 for i, r in enumerate(rows_ext)}
    rank_geo = {r["state"]: i + 1 for i, r in enumerate(rows_geo)}
    rho = spearman(rank_ext, rank_geo)
    topk_ext = [r["state"] for r in rows_ext[:LABEL_TOP_K]]
    topk_geo = [r["state"] for r in rows_geo[:LABEL_TOP_K]]
    topk_overlap = len(set(topk_ext) & set(topk_geo)) / float(LABEL_TOP_K)

    report = {
        "meta": {
            "target_state": TARGET_STATE,
            "json_source": os.path.basename(JSON_PATH),
            "state_boundary": os.path.basename(SHP),
            "n_routes": len(routes),
            "n_boot": N_BOOT,
            "note": ("Thematic connectivity map has no official chart counterpart; "
                     "validated by (A) deterministic score reproducibility, "
                     "(B) independent geometric validation of the region label, "
                     "(C) propagation of region error to the map ranking."),
        },
        "A_score_reproducibility": {
            "diagnostics": diag_ext,
            "n_connected_states": len(rows_ext),
            "target_direct_segments": diag_ext["target_direct_segments"],
            "top_states": rows_ext[:15],
        },
        "B_region_geometric_consistency": {
            "n_points_total": n_pts_total,
            "n_points_with_region": n_pts_labeled,
            "n_points_with_coord": n_pts_coord,
            "n_points_evaluated": n_eval,
            "n_points_geo_unresolved": n_geo_none,
            "region_agreement": round(agree, 4) if agree is not None else None,
            "region_agreement_95CI": ci,
            "n_mismatch": len([1 for x in flags if x == 0]),
            "mismatch_examples": mismatches,
        },
        "C_propagation": {
            "n_connected_states_geo": len(rows_geo),
            "spearman_rank_corr": round(rho, 4) if rho is not None else None,
            "top{}_overlap".format(LABEL_TOP_K): round(topk_overlap, 4),
            "top_states_extracted": topk_ext,
            "top_states_geometric": topk_geo,
            "top_states_geo_detail": rows_geo[:15],
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==================== SUMMARY ====================")
    print("A. Score reproducibility: connected states =", len(rows_ext),
          "| target direct segments =", diag_ext["target_direct_segments"])
    print("   Top5 (extracted):", [(r["state"], r["score"]) for r in rows_ext[:5]])
    print("B. Region geometric consistency:")
    print("   points total/labeled/coord/evaluated =",
          n_pts_total, n_pts_labeled, n_pts_coord, n_eval)
    print("   geo-unresolved (outside US polygons) =", n_geo_none)
    print("   region agreement = {} 95% CI {}".format(
        round(agree, 4) if agree is not None else None, ci))
    print("C. Propagation to map ranking:")
    print("   Spearman rank corr =", round(rho, 4) if rho is not None else None,
          "| Top{} overlap =".format(LABEL_TOP_K), round(topk_overlap, 4))
    print("   Top5 (geometric):", topk_geo)
    print("Report written to:", OUT)


if __name__ == "__main__":
    main()