# -*- coding: utf-8 -*-
"""
Tier B Full Model Evaluation

Covers all metrics:
  - record-level exact match (Record-EM)
  - field-level precision / recall / F1 (waypoint-level)
  - waypoint-sequence accuracy (ordered exact match)
  - waypoint-order accuracy (order correctness: Kendall tau / LCS for matched points)
  - cross-page merging accuracy (EM / recall for cross-page airway subsets)
  - confidence intervals (bootstrap 95% CI)
  - inter-annotator agreement (IAA) + adjudication

Gold standard: tierB manual annotation table (final edition) with two annotators
         (annotator1 / annotator2) after normalization and adjudication.
"""
import numpy as np
np.float = float; np.int = int; np.bool = bool; np.object = object; np.str = str

import openpyxl
import json
import os
import re
import random

random.seed(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
# Prediction data directory
PRED_DIR = os.environ.get(
    "FAA2CHART_PRED_DIR",
    os.path.join(HERE, "..", "..", "..", "data", "airway_data", "text"))
EXCEL = os.path.join(HERE, "data", "tierB_labels.xlsx")

MODELS = {
    "Claude4.5":         "parsed_routes_newnew_5_Claude4.5_updated_merged.json",
    "gpt5.2":            "parsed_routes_newnew_5_gpt5.2_updated_merged.json",
    "gpt5.2-new":        "parsed_routes_newnew_5_gpt5.2-new_updated_merged.json",
    "deepseek_chat":     "parsed_routes_newnew_5_deepseek_chat_updated_merged.json",
    "deepseek_reasoner": "parsed_routes_newnew_5_deepseek_reasoner_updated_merged.json",
}


# ------------------------------------------------------------------ Normalization
def norm(s):
    """Point name normalization: strip spaces/punctuation/asterisks, uppercase. Eliminates tokenization differences."""
    s = str(s).upper()
    s = s.replace("O'", "O").replace("U.S.", "US")
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def norm_list(pts):
    out = []
    for p in pts:
        n = norm(p)
        if n:
            out.append(n)
    return out


def sig(pts):
    """Normalized signature of an entire airway (ordered concatenation), for difference-free comparison."""
    return "|".join(norm_list(pts))


def concat_sig(pts):
    """Full concatenation (no separator), for detecting tokenization-only differences."""
    return "".join(norm_list(pts))


# ------------------------------------------------------------------ Read annotations
wb = openpyxl.load_workbook(EXCEL)
ws = wb.active

rows = []  # {code, a1[], a2[], raw, span}
for r in range(2, ws.max_row + 1):
    code = str(ws.cell(r, 1).value or "").strip().upper()
    if not code:
        continue
    a1 = str(ws.cell(r, 8).value).strip().split() if ws.cell(r, 8).value else []
    a2 = str(ws.cell(r, 9).value).strip().split() if ws.cell(r, 9).value else []
    raw = str(ws.cell(r, 6).value or "")
    span = str(ws.cell(r, 10).value or "").strip()
    rows.append({
        "code": code, "a1": a1, "a2": a2,
        "raw_norm": norm(raw), "span": (span.strip().lower() in ("yes", "true", "1")),
    })

print(f"Loaded annotations: {len(rows)} records")


# ------------------------------------------------------------------ IAA + adjudication
def adjudicate(rec):
    """
    Adjudication rules:
      1. If both annotators have the same normalized concatenated signature
         (tokenization-only difference) -> use the form with fewer points
         (merged multi-word names are more consistent with model/official naming).
      2. If concatenated signatures differ (genuine content disagreement) -> use A2
         as gold standard (A2 is more complete and standardized, and raw_text from
         single-page PDF is unreliable).
         IAA is already computed on independent annotations; disagreements are
         marked in content_diff_routes.
    Returns: gold_points(list), resolved_type('same'/'token'/'content')
    """
    a1, a2 = rec["a1"], rec["a2"]
    s1, s2 = sig(a1), sig(a2)
    if s1 == s2:
        return norm_list(a2), "same"
    c1, c2 = concat_sig(a1), concat_sig(a2)
    if c1 == c2:
        # Tokenization-only difference: use the canonical form with fewer points (merged words)
        chosen = a2 if len(norm_list(a2)) <= len(norm_list(a1)) else a1
        return norm_list(chosen), "token"
    # Genuine content disagreement: use A2 as gold (A2 is more complete and standardized)
    return norm_list(a2), "content"


gold = {}
iaa_record_em = 0        # Both annotators fully agree (normalized, order-included)
iaa_tp = iaa_a1 = iaa_a2 = 0
resolved_counts = {"same": 0, "token": 0, "content": 0}
content_diffs = []

for rec in rows:
    g, rtype = adjudicate(rec)
    gold[rec["code"]] = {"pts": g, "span": rec["span"]}
    resolved_counts[rtype] += 1
    if rtype == "content":
        content_diffs.append(rec["code"])

    n1, n2 = norm_list(rec["a1"]), norm_list(rec["a2"])
    if n1 == n2:
        iaa_record_em += 1
    # Point-level agreement (greedy 1-1, exact equality)
    used = [False] * len(n2)
    tp = 0
    for x in n1:
        for j, y in enumerate(n2):
            if not used[j] and x == y:
                used[j] = True
                tp += 1
                break
    iaa_tp += tp
    iaa_a1 += len(n1)
    iaa_a2 += len(n2)

iaa_p = iaa_tp / iaa_a1 if iaa_a1 else 0
iaa_r = iaa_tp / iaa_a2 if iaa_a2 else 0
iaa_f1 = 2 * iaa_p * iaa_r / (iaa_p + iaa_r) if (iaa_p + iaa_r) else 0

print("\n" + "=" * 70)
print("Inter-Annotator Agreement (IAA) and Adjudication")
print("=" * 70)
print(f"Record-level exact match (normalized, order-included): {iaa_record_em}/{len(rows)} = {iaa_record_em/len(rows):.4f}")
print(f"Point-level agreement F1 (A1 vs A2):                {iaa_f1:.4f} (P={iaa_p:.4f} R={iaa_r:.4f})")
print(f"Adjudication distribution: identical={resolved_counts['same']}  "
      f"tokenization-only={resolved_counts['token']}  content-disagreement={resolved_counts['content']}")
print(f"Content-disagreement airways: {content_diffs}")


# ------------------------------------------------------------------ Load model predictions
def load_model(path):
    d = json.load(open(path, encoding="utf-8"))
    mp = {}
    for item in d:
        c = str(item.get("airway_code", "")).strip().upper()
        if not c:
            continue
        pts = [pt.get("name", "") for pt in item.get("airway_point", [])]
        mp[c] = norm_list(pts)
    return mp


# ------------------------------------------------------------------ Point-level matching
def match_points(gold_pts, pred_pts):
    """
    Greedy 1-1 matching (bidirectionally lenient: handles model tokenization differences).
    Returns tp, and list of matched (gold_idx, pred_idx) pairs (for order evaluation).
    """
    used = [False] * len(pred_pts)
    pairs = []
    for gi, g in enumerate(gold_pts):
        for pj, p in enumerate(pred_pts):
            if used[pj]:
                continue
            if g == p or (len(g) >= 3 and len(p) >= 3 and (g in p or p in g)):
                used[pj] = True
                pairs.append((gi, pj))
                break
    return len(pairs), pairs


def kendall_tau_order(pairs):
    """Order consistency of matched point pairs: normalized Kendall tau (1=perfect order)."""
    if len(pairs) < 2:
        return 1.0
    pred_order = [pj for (gi, pj) in pairs]  # pred indices in gold order
    n = len(pred_order)
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pred_order[i] < pred_order[j]:
                concord += 1
            else:
                discord += 1
    total = concord + discord
    return concord / total if total else 1.0


def eval_route(gold_pts, pred_pts):
    """All metrics for a single airway."""
    if not pred_pts:
        return dict(p=0, r=0, f1=0, em=False, seq_exact=False, tau=0.0,
                    n_gold=len(gold_pts), n_pred=0, tp=0)
    tp, pairs = match_points(gold_pts, pred_pts)
    p = tp / len(pred_pts)
    r = tp / len(gold_pts) if gold_pts else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    em = (set(gold_pts) == set(pred_pts))          # point set exact match
    seq_exact = (gold_pts == pred_pts)             # ordered exact match
    tau = kendall_tau_order(pairs)                 # order consistency
    return dict(p=p, r=r, f1=f1, em=em, seq_exact=seq_exact, tau=tau,
                n_gold=len(gold_pts), n_pred=len(pred_pts), tp=tp)


def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    """Bootstrap mean 95% CI on per-route metrics."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values, dtype=float)
    means = []
    n = len(arr)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        means.append(arr[idx].mean())
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return (round(float(lo), 4), round(float(hi), 4))


# ------------------------------------------------------------------ Full model evaluation
codes = [rec["code"] for rec in rows]
span_codes = [c for c in codes if gold[c]["span"]]

report = {}
for mname, fn in MODELS.items():
    mp = load_model(os.path.join(PRED_DIR, fn))
    per = []
    covered = 0
    for c in codes:
        gpts = gold[c]["pts"]
        ppts = mp.get(c, [])
        if c in mp:
            covered += 1
        m = eval_route(gpts, ppts)
        m["code"] = c
        m["span"] = gold[c]["span"]
        per.append(m)

    # micro
    tp = sum(m["tp"] for m in per)
    ngold = sum(m["n_gold"] for m in per)
    npred = sum(m["n_pred"] for m in per)
    micro_p = tp / npred if npred else 0
    micro_r = tp / ngold if ngold else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0
    # macro
    macro_p = float(np.mean([m["p"] for m in per]))
    macro_r = float(np.mean([m["r"] for m in per]))
    macro_f1 = float(np.mean([m["f1"] for m in per]))
    # record EM & seq
    em_cnt = sum(1 for m in per if m["em"])
    seq_cnt = sum(1 for m in per if m["seq_exact"])
    tau_mean = float(np.mean([m["tau"] for m in per]))
    # cross-page
    span_per = [m for m in per if m["span"]]
    span_em = sum(1 for m in span_per if m["em"])
    span_recall = float(np.mean([m["r"] for m in span_per])) if span_per else 0
    # CI
    ci_f1 = bootstrap_ci([m["f1"] for m in per])
    ci_em = bootstrap_ci([1.0 if m["em"] else 0.0 for m in per])
    ci_r = bootstrap_ci([m["r"] for m in per])

    report[mname] = dict(
        n_routes=len(per), covered=covered,
        micro_p=round(micro_p, 4), micro_r=round(micro_r, 4), micro_f1=round(micro_f1, 4),
        macro_p=round(macro_p, 4), macro_r=round(macro_r, 4), macro_f1=round(macro_f1, 4),
        record_em=em_cnt, record_em_rate=round(em_cnt / len(per), 4),
        seq_exact=seq_cnt, seq_exact_rate=round(seq_cnt / len(per), 4),
        order_tau=round(tau_mean, 4),
        crosspage_n=len(span_per), crosspage_em=span_em,
        crosspage_em_rate=round(span_em / len(span_per), 4) if span_per else None,
        crosspage_recall=round(span_recall, 4),
        ci95_f1=ci_f1, ci95_em=ci_em, ci95_recall=ci_r,
        per_route=per,
    )

# ------------------------------------------------------------------ Print comparison
print("\n" + "=" * 100)
print("Tier B Full Model Evaluation (gold = dual-annotator adjudicated PDF print points)")
print("=" * 100)
hdr = ("{:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}"
       .format("Model", "Cover", "microF1", "microR", "macroF1", "EM_Rate", "seq_Rate", "order_tau", "Xpage_EM_Rate", "Xpage_R"))
print(hdr)
print("-" * 100)
for mname in MODELS:
    s = report[mname]
    print("{:<18} {:>4}/{:<3} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>10} {:>8.4f}".format(
        mname, s["covered"], s["n_routes"], s["micro_f1"], s["micro_r"], s["macro_f1"],
        s["record_em_rate"], s["seq_exact_rate"], s["order_tau"],
        f"{s['crosspage_em']}/{s['crosspage_n']}", s["crosspage_recall"]))

print("\n95% Bootstrap CI (per-route):")
print("{:<18} {:>18} {:>18} {:>18}".format("Model", "F1 CI", "Record-EM CI", "Recall CI"))
print("-" * 74)
for mname in MODELS:
    s = report[mname]
    print("{:<18} {:>18} {:>18} {:>18}".format(
        mname, str(s["ci95_f1"]), str(s["ci95_em"]), str(s["ci95_recall"])))

# ------------------------------------------------------------------ Save
out = {
    "gold_standard": {
        "source": os.path.basename(EXCEL),
        "n_routes": len(rows),
        "iaa_record_em": round(iaa_record_em / len(rows), 4),
        "iaa_point_f1": round(iaa_f1, 4),
        "iaa_point_p": round(iaa_p, 4),
        "iaa_point_r": round(iaa_r, 4),
        "resolution": resolved_counts,
        "content_diff_routes": content_diffs,
    },
    "models": {m: {k: v for k, v in s.items() if k != "per_route"} for m, s in report.items()},
    "per_route": {m: report[m]["per_route"] for m in MODELS},
}
outp = os.path.join(HERE, "data", "tierB_eval_all_results.json")
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nResults saved: {outp}")
