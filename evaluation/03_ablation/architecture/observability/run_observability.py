# -*- coding: utf-8 -*-
"""
Observability Experiment: Run and evaluate observability configurations.

Tests the system's robustness when certain observations are unavailable
(e.g., coordinate data, NASR reference data access), measuring the
degradation in extraction quality.

Usage:
    python run_observability.py
    python run_observability.py --config faulted  # run with faulted observations
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..")))

DATA_ROOT = os.environ.get(
    "FAA2CHART_DATA_ROOT",
    os.path.join(HERE, "..", "..", "..", "..", "data"))


def run_config(config_name, output_dir):
    """Run a single observability configuration."""
    print(f"Running observability config: {config_name}")
    os.makedirs(output_dir, exist_ok=True)
    # TODO: Implement actual observability run logic using FAA2CHART interface
    print(f"Output directory: {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="Observability experiment runner")
    ap.add_argument("--config", default="all", help="Observability config to run")
    ap.add_argument("--n-runs", type=int, default=10, help="Number of runs per config")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "obs_runs"),
                    help="Output directory")
    args = ap.parse_args()

    configs = {
        "control": "Full observability (control)",
        "single": "Single agent with full observability",
        "single_serial": "Single agent serial",
        "faulted": "Faulted/limited observability",
    }

    if args.config == "all":
        targets = list(configs.keys())
    else:
        targets = [args.config]

    for cfg in targets:
        if cfg not in configs:
            print(f"Unknown config: {cfg}, skipping")
            continue
        out_dir = os.path.join(args.out, cfg)
        run_config(cfg, out_dir)

    print("Observability runs complete.")


if __name__ == "__main__":
    main()