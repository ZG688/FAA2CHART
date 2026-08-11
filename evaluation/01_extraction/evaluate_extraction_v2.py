# -*- coding: utf-8 -*-
"""
FAA2CHART Extraction Stage Full Evaluation Script v2 (Dual Gold Source, Objective)

Uses FAA official NASR data as ground truth, evaluating all airway records on
  record-level EM / field-level P/R/F1 / waypoint sequence-order accuracy /
  coordinate localization accuracy / MEA / MAA accuracy, with bootstrap 95% CI.

Dual gold source design (key to "objective" evaluation):
  Source PDF is the February 2025 edition. Two gold sources avoid edition drift:

  (1) Content/structure metrics -> gold = Feb 2025 NASR (same as PDF cycle)
      - airway_code P/R/F1, waypoint set P/R/F1, sequence similarity (LCS),
        order consistency (Kendall-tau), MEA/MAA accuracy, Record-level EM.
      Rationale: these reflect whether the model faithfully extracted PDF content;
        must compare with the official registry from the same cycle.

  (2) Coordinate localization metrics -> gold = Nov 2025 NASR (same as exp nav DB)
      - coordinate localization accuracy (pred vs. official coordinates <= threshold).
      Rationale: the experiment's lat/lon came from the November navigation DB lookup.
        To objectively evaluate localization accuracy, the same November coordinates
        must be used; using Feb coordinates would conflate DB version differences.

Prediction source:
    parsed_routes_newnew_5_<model>_updated_merged.json
    Structure: list[{airway_code, MEA, MAA, airway_point:[{name,region,type,position:[lat,lon]}]}]

Usage:
    python evaluate_extraction_v2.py                # evaluate all models
    python evaluate_extraction_v2.py --coord-km 3   # coordinate hit threshold (default 2km)
    python evaluate_extraction_v2.py --boot 1000    # bootstrap iterations
    # Edit FEB_CSV_DIR / NOV_CSV_DIR below to switch gold directories.
"""

import os
import re
import csv
import json
import glob
import math
import random
import argparse
from collections import defaultdict

random.seed(20260714)
csv.field_size_limit(10 ** 7)

# ------------------------------------------------------------------ Path Configuration
# Data root directory for experiment data
# Set this to the root directory containing airway data subdirectories
DATA_ROOT = os.environ.get(
    "FAA2CHART_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))

# This script / report output directory
OUT_ROOT = os.path.dirname(os.path.abspath(__file__))

# gold: Feb 2025 (content/structure) - same as source PDF cycle
FEB_CSV_DIR = os.path.join(
    DATA_ROOT, "airway_data", "28DaySubscription_Effective_2025-02-20",
    "CSV_Data", "20_Feb_2025_CSV")
# gold: Nov 2025 (coordinate localization reference) - same as exp nav DB cycle
NOV_CSV_DIR = os.path.join(
    DATA_ROOT, "airway_data", "28DaySubscription_Effective_2025-11-27",
    "CSV_Data", "27_Nov_2025_CSV")

PRED_DIR = os.path.join(DATA_ROOT, "airway_data", "text")
PRED_FILES = {
    "Claude4.5":         "parsed_routes_newnew_5_Claude4.5_updated_merged.json",
    "deepseek_chat":     "parsed_routes_newnew_5_deepseek_chat_updated_merged.json",
    "deepseek_reasoner": "parsed_routes_newnew_5_deepseek_reasoner_updated_merged.json",
    "gpt5.2":            "parsed_routes_newnew_5_gpt5.2_updated_merged.json",
    "gpt5.2-new":        "parsed_routes_newnew_5_gpt5.2-new_updated_merged.json",
}


# ------------------------------------------------------------------ Utilities

def find_csv(csv_dir, name):
    """Locate a CSV in a given directory (including subdirectories)."""
    p = os.path.join(csv_dir, name)
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(csv_dir, "**", name), recursive=True)
    return hits[0] if hits else None


def norm(s):
    """Normalize: strip whitespace, uppercase. For airway_code / waypoint id comparison."""
    if s is None:
        return ""
    return re.sub(r"\s+", "", str(s)).upper()


def norm_alt(v):
    """Normalize altitude value: extract digits only (strip commas/units). Returns str or ''."""
    if v is None:
        return ""
    m = re.search(r"\d+", str(v).replace(",", ""))
    return m.group(0) if m else ""


def is_noise_code(code):
    """Noise/table-of-contents entry check (none_table_*, none_text_*, empty)."""
    c = norm(code)
    return c == "" or c.startswith("NONE_TABLE") or c.startswith("NONE_TEXT")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def lcs_len(a, b):
    """Longest common subsequence length (order-preserving), for waypoint seq similarity."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if ai == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def kendall_tau_order(pred_seq, gold_seq):
    """Kendall tau order consistency of common points between pred and gold, [-1,1].
    Returns None if <2 common points (excluded from averaging)."""
    gpos = {x: i for i, x in enumerate(gold_seq)}
    common = [x for x in pred_seq if x in gpos and pred_seq.count(x) == 1]
    if len(common) < 2:
        return None
    ppos = list(range(len(common)))
    gp = [gpos[x] for x in common]
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            s = (ppos[i] - ppos[j]) * (gp[i] - gp[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else None


# ------------------------------------------------------------------ Load gold

def load_airways(csv_dir):
    """Load airway structure gold (sequence / MEA / MAA). Returns dict.

    Modifications (2026-07):
      - Sequence gold prefers AWY_BASE.AIRWAY_STRING (avoiding multi-segment interleaving)
      - SEG_ALT only used for MEA/MAA + fallback sequence when no AWY_BASE
      - Multi-segment detection: split into independent segments when POINT_SEQ reverses
    """
    awy_base = find_csv(csv_dir, "AWY_BASE.csv")
    awy_seg = find_csv(csv_dir, "AWY_SEG_ALT.csv")

    # (1) Load AWY_BASE (primary sequence gold)
    base_seq = {}
    if awy_base:
        with open(awy_base, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                base_seq[norm(r["AWY_ID"])] = [
                    norm(t) for t in (r.get("AIRWAY_STRING") or "").split()]

    # (2) Load SEG_ALT (MEA/MAA + multi-segment detection)
    seg_rows = defaultdict(list)
    if awy_seg:
        with open(awy_seg, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                seg_rows[norm(r["AWY_ID"])].append(r)

    awy_seq, awy_mea, awy_maa = {}, {}, {}

    # Preload MEA/MAA (all segments merged into a set, deduplicated)
    for aid, rows in seg_rows.items():
        awy_mea[aid] = {norm_alt(x.get("MIN_ENROUTE_ALT")) for x in rows
                        if norm_alt(x.get("MIN_ENROUTE_ALT"))}
        awy_maa[aid] = {norm_alt(x.get("MAX_AUTH_ALT")) for x in rows
                        if norm_alt(x.get("MAX_AUTH_ALT"))}

    # Empty set fallback
    for aid in list(base_seq.keys()) + list(seg_rows.keys()):
        awy_mea.setdefault(aid, set())
        awy_maa.setdefault(aid, set())

    # (3) Sequence gold: AWY_BASE first
    for aid, seq in base_seq.items():
        if seq:
            awy_seq[aid] = seq

    # (4) Airways without AWY_BASE, use SEG_ALT to build sequence (with multi-segment detection)
    for aid, rows in seg_rows.items():
        if aid in awy_seq:
            continue  # Already have AWY_BASE sequence
        rows_sorted = sorted(rows, key=lambda x: int(x.get("POINT_SEQ") or 0))
        segments = []
        cur = []
        prev_seq = -1
        seen_from = set()
        for r in rows_sorted:
            seq_n = int(r.get("POINT_SEQ") or 0)
            fp = norm(r.get("FROM_POINT"))
            if (seq_n < prev_seq) or (fp and fp in seen_from and cur):
                if cur:
                    segments.append(cur)
                cur = []
                seen_from = set()
            if fp:
                cur.append(fp)
                seen_from.add(fp)
            prev_seq = seq_n
        last_to = norm(rows[-1].get("TO_POINT")) if rows else ""
        if cur:
            segments.append(cur)
        if segments:
            best = max(segments, key=len)
            if last_to and best[-1] != last_to:
                best.append(last_to)
            awy_seq[aid] = best

    # (5) Fallback: clear MEA/MAA for airways with no sequence (extremely rare)
    for aid in list(awy_seq.keys()):
        awy_mea.setdefault(aid, set())
        awy_maa.setdefault(aid, set())

    return dict(awy_seq=awy_seq, awy_mea=awy_mea, awy_maa=awy_maa)


def load_coords(csv_dir):
    """Load coordinate gold + full-name->id mapping (for normalizing predicted point names)."""
    coord = {}
    fix = find_csv(csv_dir, "FIX_BASE.csv")
    if fix:
        with open(fix, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    coord[norm(r["FIX_ID"])] = (float(r["LAT_DECIMAL"]),
                                                float(r["LONG_DECIMAL"]))
                except (ValueError, KeyError):
                    pass

    name2id = {}
    nav = find_csv(csv_dir, "NAV_BASE.csv")
    if nav:
        with open(nav, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                nid = norm(r["NAV_ID"])
                try:
                    coord.setdefault(nid, (float(r["LAT_DECIMAL"]),
                                           float(r["LONG_DECIMAL"])))
                except (ValueError, KeyError):
                    pass
                nm = norm(r.get("NAME"))
                if nm:
                    name2id.setdefault(nm, nid)
                combo = norm((r.get("NAME") or "") + (r.get("CITY") or ""))
                if combo:
                    name2id.setdefault(combo, nid)

    return dict(coord=coord, name2id=name2id, ids=set(coord.keys()))


# Manual alias mapping (OCR typos / PDF format deviations)
MANUAL_ALIAS = {
    "BARRETSMOUNTAIN": "BARRETTSMOUNTAIN",
    "BOYSENRESEROIR":  "BOYSENRESERVOIR",
    "SANANTHONIO":     "SANANTONIO",
    "TRVERSECITY":     "TRAVERSECITY",
    "NEBND":           "NEBND",
    "NWBND":           "NWBND",
    "SEBND":           "SEBND",
    "SWBND":           "SWBND",
    "U.S.CANADIANBORDER": "USCANADIANBORDER",
    "U.S.MEXICANBORDER":  "USMEXICANBORDER",
    "AKWPMUPVE":       "AKWPMUPVE",
}


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i, ch1 in enumerate(a, 1):
        curr = [i] * (m + 1)
        for j, ch2 in enumerate(b, 1):
            cost = 0 if ch1 == ch2 else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def fuzzy_match(n, ids, name2id, max_dist=2):
    if max_dist <= 0:
        return None
    best, best_d = None, max_dist + 1
    for key in name2id:
        d = levenshtein(n, key)
        if d < best_d:
            best_d = d
            best = key
    if best:
        return name2id[best]
    for key in ids:
        d = levenshtein(n, key)
        if d < best_d:
            best_d = d
            best = key
    return best


def to_id(name, cg):
    """Predicted waypoint name -> official short code id. Returns (id, mapped_flag).
    cg is coordinate gold (with ids / name2id).
    Mapping chain: exact -> alias -> fuzzy (Levenshtein <= 2)."""
    n = norm(name)
    if not n:
        return "", False
    # (1) Exact match
    if n in cg["ids"]:
        return n, True
    if n in cg["name2id"]:
        return cg["name2id"][n], True
    n2 = re.sub(r"(NDB|VOR|DME|VORTAC|TACAN|WP|FIX)+$", "", n)
    if n2 and n2 in cg["name2id"]:
        return cg["name2id"][n2], True
    # (2) Manual alias mapping (OCR typos)
    if n in MANUAL_ALIAS:
        alias = MANUAL_ALIAS[n]
        if alias in cg["ids"]:
            return alias, True
        if alias in cg["name2id"]:
            return cg["name2id"][alias], True
    # (3) Fuzzy match fallback (Levenshtein <= 2)
    fuzzy = fuzzy_match(n, cg["ids"], cg["name2id"], max_dist=2)
    if fuzzy:
        return fuzzy, True
    return n, False


# ------------------------------------------------------------------ Single-model evaluation

def evaluate(pred_path, feb, nov_coord, coord_km):
    """feb: Feb airway gold (content); nov_coord: Nov coordinate gold (localization reference)."""
    preds = json.load(open(pred_path, encoding="utf-8"))

    pred_map = {}
    n_pred_raw = 0
    for rec in preds:
        if is_noise_code(rec.get("airway_code")):
            continue
        n_pred_raw += 1
        pred_map.setdefault(norm(rec["airway_code"]), rec)

    gold_codes = set(feb["awy_seq"].keys())
    pred_codes = set(pred_map.keys())

    tp_code = len(pred_codes & gold_codes)
    fp_code = len(pred_codes - gold_codes)
    fn_code = len(gold_codes - pred_codes)

    per = []
    map_hit = map_tot = 0
    for c in sorted(pred_codes & gold_codes):
        prec = pred_map[c]
        gseq = feb["awy_seq"][c]           # content sequence uses Feb
        gset = set(gseq)

        pseq, coord_hit, coord_tot = [], 0, 0
        for p in prec.get("airway_point", []):
            pid, mapped = to_id(p.get("name"), nov_coord)
            if not pid:
                continue
            pseq.append(pid)
            map_tot += 1
            map_hit += 1 if mapped else 0
            # Coordinate localization accuracy: pred coords vs Nov official coords
            pos = p.get("position") or ["none", "none"]
            try:
                plat, plon = float(pos[0]), float(pos[1])
            except (ValueError, TypeError):
                plat = None
            if plat is not None and pid in nov_coord["coord"]:
                coord_tot += 1
                glat, glon = nov_coord["coord"][pid]
                if haversine_km(plat, plon, glat, glon) <= coord_km:
                    coord_hit += 1

        pset = set(pseq)
        tp, fp, fn = len(pset & gset), len(pset - gset), len(gset - pset)

        # --- Tier A metrics ---
        # OSV: Ordered Subsequence Validity = LCS(pred, gold) / pred_len
        l = lcs_len(pseq, gseq)
        osv = l / len(pseq) if len(pseq) > 0 else 0.0
        # Truncation detection: last position in gold hit by a predicted point
        last_idx = -1
        for gi, gp in enumerate(gseq):
            if gp in pset:
                last_idx = gi
        trunc_ratio = (last_idx + 1) / len(gseq) if last_idx >= 0 else 0.0
        is_truncated = 1 if (trunc_ratio < 0.8 and len(gseq) > 5) else 0
        # ----

        l = lcs_len(pseq, gseq)
        seq_ratio = l / max(len(pseq), len(gseq)) if max(len(pseq), len(gseq)) else 0.0
        tau = kendall_tau_order(pseq, gseq)

        pmea, pmaa = norm_alt(prec.get("MEA")), norm_alt(prec.get("MAA"))
        mea_ok = 1 if (pmea and pmea in feb["awy_mea"].get(c, set())) else 0
        maa_ok = 1 if (pmaa and pmaa in feb["awy_maa"].get(c, set())) else 0
        mea_has = 1 if feb["awy_mea"].get(c) else 0
        maa_has = 1 if feb["awy_maa"].get(c) else 0

        seq_exact = 1 if pseq == gseq else 0
        record_em = 1 if (seq_exact and mea_ok and maa_ok) else 0

        # Data-driven recall stratification (objective, reproducible):
        #   complete = pred points >= gold points (full point set, tests recognition quality)
        #   endpoint = pred exactly 2 points while gold > 2 (source table only lists endpoints)
        #   under    = remaining (partial omission)
        n_pred_pts, n_gold_pts = len(pseq), len(gseq)
        if n_gold_pts > 0 and n_pred_pts >= n_gold_pts:
            stratum = "complete"
        elif n_pred_pts == 2 and n_gold_pts > 2:
            stratum = "endpoint"
        else:
            stratum = "under"
        endpoint_ok = 1 if (n_pred_pts >= 2 and n_gold_pts >= 2
                            and pseq[0] == gseq[0] and pseq[-1] == gseq[-1]) else 0

        per.append(dict(code=c, tp=tp, fp=fp, fn=fn,
                        n_pred_pts=n_pred_pts, n_gold_pts=n_gold_pts,
                        stratum=stratum, endpoint_ok=endpoint_ok,
                        seq_ratio=seq_ratio, tau=tau, seq_exact=seq_exact,
                        coord_hit=coord_hit, coord_tot=coord_tot,
                        mea_ok=mea_ok, mea_has=mea_has,
                        maa_ok=maa_ok, maa_has=maa_has, record_em=record_em,
                        osv=osv, trunc_ratio=trunc_ratio, is_truncated=is_truncated))

    return dict(
        model_file=os.path.basename(pred_path),
        n_pred_raw=n_pred_raw, n_pred_codes=len(pred_codes),
        n_gold_codes=len(gold_codes),
        code_tp=tp_code, code_fp=fp_code, code_fn=fn_code, per=per,
        map_cov=(map_hit / map_tot if map_tot else 0.0), map_tot=map_tot)


# ------------------------------------------------------------------ Aggregation + CI

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def summarize(res, n_boot=1000):
    per = res["per"]
    n = len(per)

    code_p, code_r, code_f = prf(res["code_tp"], res["code_fp"], res["code_fn"])
    wp_tp = sum(x["tp"] for x in per)
    wp_fp = sum(x["fp"] for x in per)
    wp_fn = sum(x["fn"] for x in per)
    wp_p, wp_r, wp_f = prf(wp_tp, wp_fp, wp_fn)

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    seq_ratio = mean([x["seq_ratio"] for x in per])
    tau = mean([x["tau"] for x in per])
    seq_exact = mean([x["seq_exact"] for x in per])
    record_em = mean([x["record_em"] for x in per])
    osv_mean = mean([x["osv"] for x in per])
    trunc_ratio_mean = mean([x["trunc_ratio"] for x in per])
    trunc_pct = sum(x["is_truncated"] for x in per) / len(per) if per else 0.0

    # Stratified Record-EM / Seq-Exact rate
    strata = {}
    for key in ("complete", "endpoint", "under"):
        grp = [x for x in per if x["stratum"] == key]
        strata[key] = dict(
            n=len(grp),
            frac=(len(grp) / len(per) if per else 0.0),
            record_EM=mean([x["record_em"] for x in grp]),
            seq_exact=mean([x["seq_exact"] for x in grp]),
            endpoint_ok=mean([x["endpoint_ok"] for x in grp]),
        )
    # Fairer metrics on complete stratum (full point set)
    comp = [x for x in per if x["stratum"] == "complete"]
    record_em_complete = mean([x["record_em"] for x in comp])
    seq_exact_complete = mean([x["seq_exact"] for x in comp])
    coord_t = sum(x["coord_tot"] for x in per)
    coord_acc = sum(x["coord_hit"] for x in per) / coord_t if coord_t else 0.0
    mea_has = sum(x["mea_has"] for x in per)
    maa_has = sum(x["maa_has"] for x in per)
    mea_acc = sum(x["mea_ok"] for x in per) / mea_has if mea_has else 0.0
    maa_acc = sum(x["maa_ok"] for x in per) / maa_has if maa_has else 0.0

    def boot(fn_metric):
        out = []
        for _ in range(n_boot):
            sample = [per[random.randrange(n)] for _ in range(n)] if n else []
            out.append(fn_metric(sample))
        out.sort()
        return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]

    ci = {
        "record_EM":   boot(lambda s: mean([x["record_em"] for x in s])),
        "waypoint_F1": boot(lambda s: prf(sum(x["tp"] for x in s),
                                          sum(x["fp"] for x in s),
                                          sum(x["fn"] for x in s))[2]),
        "waypoint_R":  boot(lambda s: prf(sum(x["tp"] for x in s),
                                          sum(x["fp"] for x in s),
                                          sum(x["fn"] for x in s))[1]),
        "seq_ratio":   boot(lambda s: mean([x["seq_ratio"] for x in s])),
        "order_tau":   boot(lambda s: mean([x["tau"] for x in s])),
        "coord_acc":   boot(lambda s: (sum(x["coord_hit"] for x in s) /
                                       sum(x["coord_tot"] for x in s))
                            if sum(x["coord_tot"] for x in s) else 0.0),
        "MEA_acc":     boot(lambda s: (sum(x["mea_ok"] for x in s) /
                                       sum(x["mea_has"] for x in s))
                            if sum(x["mea_has"] for x in s) else 0.0),
        "MAA_acc":     boot(lambda s: (sum(x["maa_ok"] for x in s) /
                                       sum(x["maa_has"] for x in s))
                            if sum(x["maa_has"] for x in s) else 0.0),
        "OSV":         boot(lambda s: mean([x["osv"] for x in s])),
    }

    return dict(
        n_matched=n,
        code_P=code_p, code_R=code_r, code_F1=code_f,
        waypoint_P=wp_p, waypoint_R=wp_r, waypoint_F1=wp_f,
        seq_ratio=seq_ratio, order_tau=tau, seq_exact=seq_exact,
        record_EM=record_em, coord_acc=coord_acc, coord_n=coord_t,
        MEA_acc=mea_acc, MEA_n=mea_has, MAA_acc=maa_acc, MAA_n=maa_has,
        map_cov=res["map_cov"], map_tot=res["map_tot"],
        strata=strata, record_EM_complete=record_em_complete,
        seq_exact_complete=seq_exact_complete, ci=ci,
        OSV=osv_mean, trunc_ratio=trunc_ratio_mean, trunc_pct=trunc_pct)


# ------------------------------------------------------------------ Print

def fmt(v, ci):
    return "{:.3f} [{:.3f},{:.3f}]".format(v, ci[0], ci[1])


def print_report(name, res, s):
    print("\n" + "=" * 76)
    print("Model: {}   File: {}".format(name, res["model_file"]))
    print("-" * 76)
    print("Predicted airways (denoised): {}   Unique codes: {}   Gold airways (Feb): {}   Matched: {}".format(
        res["n_pred_raw"], res["n_pred_codes"], res["n_gold_codes"], s["n_matched"]))
    print("Waypoint-name -> official ID mapping coverage (Nov DB): {:.1%} (n={})".format(
        s["map_cov"], s["map_tot"]))
    print("-" * 76)
    print("gold=Feb2025 | [Airway Code]        P={:.3f} R={:.3f} F1={:.3f}".format(
        s["code_P"], s["code_R"], s["code_F1"]))
    print("gold=Feb2025 | [Waypoint Set]       P={:.3f}  (P = point validity, usable)".format(s["waypoint_P"]))
    print("  [!] The following R / F1 have a reference-system mismatch "
          "(numerator=PDF-extracted, denominator=AWY_BASE full DB). "
          "They are NOT extraction capability metrics; archived only:")
    print("      R(misleading)={} F1(misleading)={}".format(
        fmt(s["waypoint_R"], s["ci"]["waypoint_R"]),
        fmt(s["waypoint_F1"], s["ci"]["waypoint_F1"])))
    print("gold=Feb2025 | [Seq Similarity]     LCS-ratio={}".format(
        fmt(s["seq_ratio"], s["ci"]["seq_ratio"])))
    print("gold=Feb2025 | [Order Consistency]  Kendall-tau={}".format(
        fmt(s["order_tau"], s["ci"]["order_tau"])))
    print("gold=Feb2025 | [OSV]                Ordered Subseq Validity={}  (pred->gold LCS / pred_len)".format(
        fmt(s["OSV"], s["ci"]["OSV"])))
    print("gold=Feb2025 | [Truncation]         trunc_ratio={:.3f}  truncated(pct)={:.1%}".format(
        s["trunc_ratio"], s["trunc_pct"]))
    print("gold=Feb2025 | [Point Validity]     waypoint Precision={:.3f}  (fraction of pred points in official point set)".format(
        s["waypoint_P"]))
    print("gold=Feb2025 | [MEA Accuracy]       Acc={} (n={})".format(
        fmt(s["MEA_acc"], s["ci"]["MEA_acc"]), s["MEA_n"]))
    print("gold=Feb2025 | [MAA Accuracy]       Acc={} (n={})".format(
        fmt(s["MAA_acc"], s["ci"]["MAA_acc"]), s["MAA_n"]))
    print("  [!] Seq-exact / Record-EM have reference-system mismatch "
          "(require exact match against AWY_BASE full DB). "
          "Not extraction capability metrics; archived only:")
    print("      Seq-exact(misleading)={:.3f}  Record-EM(misleading)={}".format(
        s["seq_exact"], fmt(s["record_EM"], s["ci"]["record_EM"])))
    print("gold=Nov2025 | [Coord Loc.<=threshold] Acc={} (comparable pts n={})".format(
        fmt(s["coord_acc"], s["ci"]["coord_acc"]), s["coord_n"]))
    print("-" * 76)
    print("[Recall-stratified Record-EM / Seq-Exact] (gold=Feb2025) "
          "[!]Reference mismatch; for error analysis only, not capability metrics")
    for key in ("complete", "endpoint", "under"):
        st = s["strata"][key]
        print("   {:<9} n={:>4} ({:>5.1%})  RecordEM={:.3f}  seqExact={:.3f}  endpointOK={:.3f}".format(
            key, st["n"], st["frac"], st["record_EM"], st["seq_exact"], st["endpoint_ok"]))
    print("   >> Complete stratum only (full point set): Record-EM={:.3f}  Seq-Exact={:.3f}".format(
        s["record_EM_complete"], s["seq_exact_complete"]))


def print_compare(rows):
    print("\n" + "#" * 96)
    print("# Model cross-comparison (point estimate [95%CI]); content=Feb2025, coord=Nov2025")
    print("#" * 96)
    print("{:<18}{:>8}{:>16}{:>16}{:>16}{:>12}{:>14}{:>13}{:>8}{:>10}".format(
        "model", "code_F1", "wp_F1", "coord_acc", "record_EM(all)",
        "EM(complete)", "seqExact(compl)", "complete%", "OSV", "trunc%"))
    for name, s in rows:
        print("{:<18}{:>8.3f}{:>16}{:>16}{:>16.3f}{:>12.3f}{:>14.3f}{:>12.1%}{:>8.3f}{:>10.1%}".format(
            name, s["code_F1"],
            fmt(s["waypoint_F1"], s["ci"]["waypoint_F1"]),
            fmt(s["coord_acc"], s["ci"]["coord_acc"]),
            s["record_EM"], s["record_EM_complete"],
            s["seq_exact_complete"], s["strata"]["complete"]["frac"],
            s["OSV"], s["trunc_pct"]))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="Evaluate a single prediction json")
    ap.add_argument("--coord-km", type=float, default=2.0, help="Coordinate hit threshold (km)")
    ap.add_argument("--boot", type=int, default=1000, help="Bootstrap iterations")
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "extraction_eval_report.json"))
    args = ap.parse_args()

    print("=" * 76)
    print("Dual gold source evaluation:")
    print("  Content/sequence/MEA/MAA/RecordEM  <- Feb 2025 (same as source PDF cycle)")
    print("  Coordinate localization            <- Nov 2025 (same as experiment nav DB cycle)")
    print("=" * 76)

    feb = load_airways(FEB_CSV_DIR)
    nov_coord = load_coords(NOV_CSV_DIR)
    print("Feb airway gold: {}   Nov coord points: {}   Nov full-name aliases: {}".format(
        len(feb["awy_seq"]), len(nov_coord["coord"]), len(nov_coord["name2id"])))

    targets = ({"single": os.path.basename(args.pred)} if args.pred else PRED_FILES)
    report, rows = {}, []
    for name, fn in targets.items():
        path = args.pred if args.pred else os.path.join(PRED_DIR, fn)
        if not os.path.exists(path):
            print("  [SKIP] not found:", path)
            continue
        res = evaluate(path, feb, nov_coord, args.coord_km)
        s = summarize(res, n_boot=args.boot)
        print_report(name, res, s)
        report[name] = s
        rows.append((name, s))

    if len(rows) > 1:
        print_compare(rows)

    meta = {
        "gold_content_version": "2025-02-20 (Feb, same as source PDF)",
        "gold_coord_version": "2025-11-27 (Nov, same as experiment geolocation DB)",
        "coord_km": args.coord_km, "boot": args.boot,
        "note": "Content metrics use Feb to eliminate edition drift; coord localization uses Nov (same DB as locator), objective assessment of localization step.",
        "metric_validity_warning": (
            "waypoint_R / waypoint_F1 / seq_exact / record_EM have reference-system mismatch: "
            "numerator = model-extracted points from PDF chart (FAA selective printing), "
            "denominator = AWY_BASE DB full set. These two are structurally inequivalent. "
            "These metrics measure FAA printing policy, not model extraction capability; "
            "they must not be used as extraction capability indicators. "
            "For true recall, see Tier B (PDF sampling gold standard). "
            "Valid extraction capability metrics: OSV / point validity (waypoint_P) / "
            "coord localization / order (tau) / MEA/MAA (intersection)."),
        "reliable_metrics": ["code_F1", "waypoint_P", "OSV", "order_tau", "coord_acc", "MEA_acc", "MAA_acc"],
        "unreliable_metrics": ["waypoint_R", "waypoint_F1", "seq_exact", "record_EM", "record_EM_complete"],
    }
    slim = {}
    for name, s in report.items():
        slim[name] = {k: v for k, v in s.items() if k != "ci"}
        slim[name]["ci95"] = {k: list(v) for k, v in s["ci"].items()}
    json.dump({"meta": meta, "results": slim},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nReport saved:", args.out)


if __name__ == "__main__":
    main()
