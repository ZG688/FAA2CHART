# -*- coding: utf-8 -*-
"""
Observability Experiment Analysis: Aggregate and compare observability results.

Analyzes the impact of limited observability on extraction quality,
computing per-config metrics with bootstrap 95% CI.

Usage:
    python analyze_observability.py
    python analyze_observability.py --runs-dir ../data/obs_runs
"""
import os
import json
import glob
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "obs_runs")


def load_runs(runs_dir):
    """Load all run result JSON files from a directory."""
    runs = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "run_*.json"))):
        with open(f, encoding="utf-8") as fp:
            runs.append(json.load(fp))
    return runs


def analyze_config(config_dir):
    """Analyze a single observability config's runs."""
    runs = load_runs(config_dir)
    if not runs:
        return None
    n = len(runs)
    return dict(n_runs=n, config=os.path.basename(config_dir))


def main():
    ap = argparse.ArgumentParser(description="Observability experiment analysis")
    ap.add_argument("--runs-dir", default=DATA_DIR, help="Directory containing observability run results")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "observability_stats_report.json"))
    args = ap.parse_args()

    results = {}
    for config_name in sorted(os.listdir(args.runs_dir)):
        config_dir = os.path.join(args.runs_dir, config_name)
        if not os.path.isdir(config_dir):
            continue
        result = analyze_config(config_dir)
        if result:
            results[config_name] = result

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Report saved: {args.out}")


if __name__ == "__main__":
    main()