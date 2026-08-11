# -*- coding: utf-8 -*-
"""
Tier B 150-Route -- Baseline Comparison Evaluation
===================================================
Evaluates the multi-agent primary result and 6 baselines (rule-based/rag/codegen/
react/autogen) against the 150-route human-annotated gold standard.

Gold source: tierB_labels.xlsx (150 routes)
  - Column 8: annotator1_pdf_points
  - Column 9: annotator2_pdf_points
  - Column 10: cross-page flag
  After IAA adjudication + OCR spell correction, forms final gold.

Metrics:
  - code_P / code_R / code_F1      airway-code level
  - waypoint_micro_P/R/F1          point-level micro-average (substring match)
  - waypoint_macro_P/R/F1          point-level macro-average
  - record_EM                      record-level exact match (point set equality)
  - seq_exact                      sequence exact match (ordered equality)
  - OSV                            Ordered Subsequence Validity (LCS/len(pred))
  - order_tau                      Kendall tau order consistency
  - trunc%                         truncation rate
  - crosspage_EM                   cross-page airway record EM rate

Note: No MEA/MAA/coord metrics (human annotations lack altitude/coordinate fields).
"""
import numpy as np; np.float = float; np.int = int; np.bool = bool; np.object = object; np.str = str  # numpy compatibility patch
import openpyxl, json, os, re, csv, glob, math, random
from collections import defaultdict

random.seed(20260714)

HERE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.environ.get(
    "FAA2CHART_PRED_DIR",
    os.path.join(HERE, "..", "..", "..", "data", "airway_data", "text"))
EXCEL = os.path.join(HERE, "data", "tierB_labels.xlsx")

TARGETS = {
    "FAA2CHART":    "parsed_routes_newnew_5_deepseek_chat_updated_merged.json",
    "Rule-based":   "parsed_routes_baseline_rule_based.json",
    "RAG":          "parsed_routes_baseline_rag.json",
    "Codegen":      "parsed_routes_baseline_codegen.json",
    "ReAct":        "parsed_routes_baseline_react.json",
    "AutoGen":      "parsed_routes_baseline_autogen.json",
}


# ------------------------------------------------------------------ Normalization
def norm(s):
    s = str(s).upper().replace("O'", "O").replace("U.S.", "US")
    return re.sub(r"[^A-Z0-9]", "", s)


def norm_list(pts):
    return [norm(p) for p in pts if norm(p)]


def sig(pts):
    return "|".join(norm_list(pts))


def concat_sig(pts):
    return "".join(norm_list(pts))


# ------------------------------------------------------------------ Sequence tools
def lcs_len(a, b):
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


# ------------------------------------------------------------------ OCR spell correction
OCR_ERRORS = {
    "LBBOCK": "LUBBOCK", "LUBOCK": "LUBBOCK", "LBBOOK": "LUBBOCK",
    "WTLLROGERS": "WILLROGERS", "WILLROGER": "WILLROGERS", "WILLROGERSS": "WILLROGERS",
    "BROWNSVILL": "BROWNSVILLE", "BROWNSVILE": "BROWNSVILLE", "BROWNSVLLE": "BROWNSVILLE",
    "BROWNSVILLW": "BROWNSVILLE", "BROWNSVILLEE": "BROWNSVILLE", "BROWNSVILLS": "BROWNSVILLE",
    "JACKSONVILL": "JACKSONVILLE", "JACKSONVILE": "JACKSONVILLE", "JACKSONVLLE": "JACKSONVILLE",
    "SACRAMENT": "SACRAMENTO", "SACRAMNTO": "SACRAMENTO",
    "OKLAHOMASITY": "OKLAHOMACITY", "OKLAHOMACTIY": "OKLAHOMACITY",
    "PHILADELPHI": "PHILADELPHIA",
    "ALBUQUERQE": "ALBUQUERQUE",
    "KNXVLLE": "KNOXVILLE",
    "NASHVILL": "NASHVILLE", "NASHVILE": "NASHVILLE",
    "11N": "L1N",
}


def auto_correct_points(pts_list):
    out = []
    for p in pts_list:
        pn = norm(p)
        out.append(OCR_ERRORS.get(pn, pn))
    return out


# ------------------------------------------------------------------ Load gold
def load_gold():
    """Read human annotation Excel, IAA adjudication + OCR correction, return gold dict."""
    wb = openpyxl.load_workbook(EXCEL)
    ws = wb.active
    rows = []
    for r in range(2, ws.max_row + 1):
        code = str(ws.cell(r, 1).value or "").strip().upper()
        if not code:
            continue
        a1 = str(ws.cell(r, 8).value or "").strip().split()
        a2 = str(ws.cell(r, 9).value or "").strip().split()
        span = str(ws.cell(r, 10).value or "").strip()
        rows.append({"code": code, "a1": a1, "a2": a2, "span": (span.strip().lower() in ("yes", "true", "1"))})

    for rec in rows:
        rec["a1_corrected"] = auto_correct_points(rec["a1"])
        rec["a2_corrected"] = auto_correct_points(rec["a2"])

    gold = {}
    iaa_record_em = 0
    iaa_tp = iaa_a1 = iaa_a2 = 0
    resolved = {"same": 0, "token": 0, "content": 0}

    for rec in rows:
        if not rec["a1"]:
            g = rec["a2_corrected"]
            rtype = "same"
        else:
            s1 = "|".join(rec["a1_corrected"])
            s2 = "|".join(rec["a2_corrected"])
            if s1 == s2:
                g = rec["a2_corrected"]
                rtype = "same"
            elif "".join(rec["a1_corrected"]) == "".join(rec["a2_corrected"]):
                g = rec["a2_corrected"] if len(rec["a2_corrected"]) <= len(rec["a1_corrected"]) else rec["a1_corrected"]
                rtype = "token"
            else:
                g = rec["a2_corrected"]
                rtype = "content"
        gold[rec["code"]] = {"pts": g, "span": rec["span"]}
        resolved[rtype] += 1

        n1, n2 = rec["a1_corrected"], rec["a2_corrected"]
        if n1 == n2:
            iaa_record_em += 1
        used = [False] * len(n2)
        tp = 0
        for x in n1:
            for j, y in enumerate(n2):
                if not used[j] and x == y:
                    used[j] = True; tp += 1; break
        iaa_tp += tp
        iaa_a1 += len(n1)
        iaa_a2 += len(n2)

    iaa_p = iaa_tp / iaa_a1 if iaa_a1 else 0
    iaa_r = iaa_tp / iaa_a2 if iaa_a2 else 0
    iaa_f1 = 2 * iaa_p * iaa_r / (iaa_p + iaa_r) if (iaa_p + iaa_r) else 0

    print(f"Loaded annotations: {len(rows)} records")
    print(f"IAA: record-level agreement={iaa_record_em}/{len(rows)}={iaa_record_em/len(rows):.4f}, "
          f"point-level F1={iaa_f1:.4f} (P={iaa_p:.4f} R={iaa_r:.4f})")
    print(f"Adjudication: same={resolved['same']} token={resolved['token']} content={resolved['content']}")

    return gold, dict(n=len(rows), iaa_record_em=iaa_record_em / len(rows),
                      iaa_point_f1=iaa_f1, iaa_p=iaa_p, iaa_r=iaa_r,
                      resolution=resolved)


# ------------------------------------------------------------------ Load predictions
def load_pred(path):
    if not os.path.exists(path):
        print(f"  [WARNING] not found: {path}")
        return {}
    d = json.load(open(path, encoding="utf-8"))
    mp = {}
    for item in d:
        c = norm(item.get("airway_code", ""))
        if not c or c.startswith("NONETABLE") or c.startswith("NONETEXT"):
            continue
        pts = [pt.get("name", "") for pt in item.get("airway_point", [])]
        mp[c] = norm_list(pts)
    return mp


# ------------------------------------------------------------------ Point matching
def match_points(gold_pts, pred_pts):
    used = [False] * len(pred_pts)
    tp = 0
    for g in gold_pts:
        for pj, p in enumerate(pred_pts):
            if used[pj]:
                continue
            if g == p or (len(g) >= 3 and len(p) >= 3 and (g in p or p in g)):
                used[pj] = True; tp += 1; break
    return tp


# ------------------------------------------------------------------ Single-route evaluation
def eval_route(gold_pts, pred_pts):
    n_gold = len(gold_pts)
    n_pred = len(pred_pts)
    if n_pred == 0:
        return dict(p=0, r=0, f1=0, em=False, seq_exact=False,
                    n_gold=n_gold, n_pred=0, tp=0,
                    osv=0.0, tau=None, trunc_ratio=0.0, is_truncated=0)

    tp = match_points(gold_pts, pred_pts)
    p = tp / n_pred
    r = tp / n_gold if n_gold else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    em = (set(gold_pts) == set(pred_pts))
    seq_exact = (gold_pts == pred_pts)

    lcs = lcs_len(pred_pts, gold_pts)
    osv = lcs / n_pred if n_pred else 0.0
    tau = kendall_tau_order(pred_pts, gold_pts)

    pset = set(pred_pts)
    last_idx = -1
    for gi, gp in enumerate(gold_pts):
        if gp in pset:
            last_idx = gi
    trunc_ratio = (last_idx + 1) / n_gold if last_idx >= 0 and n_gold else 0.0
    is_truncated = 1 if (trunc_ratio < 0.8 and n_gold > 5) else 0

    return dict(p=p, r=r, f1=f1, em=em, seq_exact=seq_exact,
                n_gold=n_gold, n_pred=n_pred, tp=tp,
                osv=osv, tau=tau, trunc_ratio=trunc_ratio,
                is_truncated=is_truncated)


# ------------------------------------------------------------------ Aggregation + bootstrap
def summarize(per, gold, pred_map, n_boot=2000):
    n = len(per)
    if n == 0:
        return None

    tp = sum(x["tp"] for x in per)
    ngold = sum(x["n_gold"] for x in per)
    npred = sum(x["n_pred"] for x in per)
    micro_p = tp / npred if npred else 0
    micro_r = tp / ngold if ngold else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0

    macro_p = float(np.mean([x["p"] for x in per]))
    macro_r = float(np.mean([x["r"] for x in per]))
    macro_f1 = float(np.mean([x["f1"] for x in per]))

    em_cnt = sum(1 for x in per if x["em"])
    seq_cnt = sum(1 for x in per if x["seq_exact"])

    osv_mean = float(np.mean([x["osv"] for x in per]))
    tau_vals = [x["tau"] for x in per if x["tau"] is not None]
    tau_mean = float(np.mean(tau_vals)) if tau_vals else 0.0
    trunc_pct = sum(x["is_truncated"] for x in per) / n

    span_per = [x for x in per if x["span"]]
    span_em = sum(1 for x in span_per if x["em"])
    span_recall = float(np.mean([x["r"] for x in span_per])) if span_per else 0

    covered = sum(1 for x in per if x["n_pred"] > 0)
    n_extracted = len(pred_map)

    def boot(fn):
        out = []
        for _ in range(n_boot):
            sample = [per[random.randrange(n)] for _ in range(n)]
            out.append(fn(sample))
        out.sort()
        return [round(out[int(0.025 * len(out))], 4),
                round(out[int(0.975 * len(out)) - 1], 4)]

    def mean_f1(s):
        tp_s = sum(x["tp"] for x in s)
        npred_s = sum(x["n_pred"] for x in s)
        ngold_s = sum(x["n_gold"] for x in s)
        p_s = tp_s / npred_s if npred_s else 0
        r_s = tp_s / ngold_s if ngold_s else 0
        return 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) else 0

    ci = {
        "micro_F1": boot(mean_f1),
        "record_EM": boot(lambda s: sum(1 for x in s if x["em"]) / len(s) if s else 0),
        "OSV": boot(lambda s: float(np.mean([x["osv"] for x in s])) if s else 0),
        "order_tau": boot(lambda s: float(np.mean([x["tau"] for x in s if x["tau"] is not None]))
                          if any(x["tau"] is not None for x in s) else 0),
    }

    return dict(
        n_routes=n, covered=covered,
        n_extracted=n_extracted,
        micro_P=round(micro_p, 4), micro_R=round(micro_r, 4), micro_F1=round(micro_f1, 4),
        macro_P=round(macro_p, 4), macro_R=round(macro_r, 4), macro_F1=round(macro_f1, 4),
        record_EM=em_cnt, record_EM_rate=round(em_cnt / n, 4),
        seq_exact=seq_cnt, seq_exact_rate=round(seq_cnt / n, 4),
        OSV=round(osv_mean, 4), order_tau=round(tau_mean, 4),
        trunc_pct=round(trunc_pct, 4),
        crosspage_n=len(span_per),
        crosspage_EM=span_em,
        crosspage_EM_rate=round(span_em / len(span_per), 4) if span_per else None,
        crosspage_recall=round(span_recall, 4),
        ci95=ci,
    )


# ------------------------------------------------------------------ Main
def main():
    print("=" * 80)
    print("Tier B 150-Route Gold Standard -- Baseline Comparison Evaluation")
    print("gold = human-annotated 150 routes (IAA adjudicated + OCR corrected)")
    print("=" * 80)

    gold, gold_meta = load_gold()
    codes = list(gold.keys())
    print(f"\nGold airway count: {len(codes)}, cross-page: {sum(1 for c in codes if gold[c]['span'])}")

    report = {}
    for name, fn in TARGETS.items():
        path = os.path.join(PRED_DIR, fn)
        pred_map = load_pred(path)
        per = []
        for c in codes:
            gpts = gold[c]["pts"]
            ppts = pred_map.get(c, [])
            m = eval_route(gpts, ppts)
            m["code"] = c
            m["span"] = gold[c]["span"]
            per.append(m)
        s = summarize(per, gold, pred_map)
        if s:
            report[name] = s
            print(f"\n{name}: covered={s['covered']}/{s['n_routes']}  "
                  f"micro_F1={s['micro_F1']:.4f}  record_EM={s['record_EM_rate']:.4f}  "
                  f"OSV={s['OSV']:.4f}  tau={s['order_tau']:.4f}  trunc={s['trunc_pct']:.1%}")

    # Cross-comparison table
    print(f"\n{'=' * 140}")
    print(f"{'Model':<16} {'Extracted':>8} {'Cover':>6} {'micro_P':>8} {'micro_R':>8} {'micro_F1':>8} "
          f"{'EM_Rate':>8} {'seq_Rate':>8} {'OSV':>8} {'tau':>8} {'trunc%':>8} {'Xpage_EM':>8}")
    print(f"{'-' * 140}")
    for name in TARGETS:
        if name not in report:
            continue
        s = report[name]
        print(f"{name:<16} {s['n_extracted']:>8} {s['covered']:>4}/{s['n_routes']:<3} "
              f"{s['micro_P']:>8.4f} {s['micro_R']:>8.4f} {s['micro_F1']:>8.4f} "
              f"{s['record_EM_rate']:>8.4f} {s['seq_exact_rate']:>8.4f} "
              f"{s['OSV']:>8.4f} {s['order_tau']:>8.4f} {s['trunc_pct']:>8.1%} "
              f"{s['crosspage_EM']}/{s['crosspage_n']}")

    # Save JSON
    out = {
        "meta": {
            "gold_source": "human-annotated 150 routes (tierB_labels.xlsx)",
            "gold_pipeline": "IAA adjudication + OCR spell correction",
            "iaa": gold_meta,
            "note": "gold is human-annotated airway point sequences from PDF, preserving printed order, independent of NASR DB",
        },
        "results": report,
    }
    outp = os.path.join(HERE, "data", "tierB_150_baselines_report.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nResults saved: {outp}")


if __name__ == "__main__":
    main()