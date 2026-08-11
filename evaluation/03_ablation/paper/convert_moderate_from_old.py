# -*- coding: utf-8 -*-
"""Convert old format (run_ablation_final control/single_serial run_*.json)
to new three-task format. Only processes moderate tasks (existing data is moderate mapping tasks).
"""
from __future__ import annotations
import os
import json

_SRC = os.environ.get("FAA2CHART_ABLATION_RUNS", "ablation_runs")
_DST = os.environ.get("FAA2CHART_CONVERTED_OUTPUT", "converted/moderate")
_TASK = "moderate"

# run 31-40 -> new indices 1-10 (same batch, similar API latency)

for variant_src, variant_dst in [("control", "multi_agent"), ("single_serial", "single_agent")]:
    agent_label = "multi" if variant_src == "control" else "single"
    out_folder = os.path.join(_DST, variant_dst)
    os.makedirs(out_folder, exist_ok=True)
    for new_idx, src_idx in enumerate(range(31, 41), start=1):
        src_path = os.path.join(_SRC, variant_src, f"run_{src_idx}.json")
        if not os.path.isfile(src_path):
            print(f"[Skip] {src_path} does not exist")
            continue
        with open(src_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        metrics = rec.get("metrics", {})
        # approximate observable_points with steps + ai_messages when missing
        obs = metrics.get("total_steps", 0) + metrics.get("ai_messages", 0)
        new = {
            "task": _TASK,
            "agent": agent_label,
            "run_idx": new_idx,
            "variant_in_ablation": variant_src,
            "start_time": rec.get("start_time", ""),
            "elapsed_sec": rec.get("elapsed_sec", 0),
            "success": rec.get("success", False),
            "error": rec.get("error", ""),
            "fault_injection_enabled": False,
            "metrics": {
                "total_steps": int(metrics.get("total_steps", 0)),
                "tool_calls": int(metrics.get("tool_calls", 0)),
                "subagent_delegations": int(metrics.get("subagent_delegations", 0)),
                "tool_errors": float(metrics.get("tool_errors", 0)),
                "ai_messages": int(metrics.get("ai_messages", 0)),
                "observable_points": int(obs),
                "n_errors_detected": 0,
                "first_error_step": None,
                "error_localization_rate": None,
                "recovery_rate": None,
                "mean_recovery_steps": None,
                "mean_recovery_sec": None,
                "call_result_pairing_rate": None,
            },
            "obs_trace_collected": False,
            "task_description": "CA connectivity map reused from ablation (approx. moderate)",
            "events_file": None,
            "_note": "Reused from run_ablation_final run_{} (CA, no PDF, geocode skipped).  Observable_points approx. steps + ai_messages (no trace_collected).".format(src_idx),
        }
        out_path = os.path.join(out_folder, f"run_{new_idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new, f, ensure_ascii=False, indent=2)
        print(f"[OK] {variant_src} run_{src_idx} -> {variant_dst} run_{new_idx} | success={rec.get('success')} elapsed={rec.get('elapsed_sec')}s")

# Append index.jsonl
idx_path = os.path.join(os.path.dirname(_DST), "index.jsonl")
import glob
if os.path.exists(idx_path):
    os.remove(idx_path)
for task_folder in ["simple", "moderate", "complex"]:
    for agent_folder in ["multi_agent", "single_agent"]:
        folder = os.path.join(os.path.dirname(_DST), task_folder, agent_folder)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(glob.glob(os.path.join(folder, "run_*.json")),
                         key=lambda s: int(os.path.basename(s)[4:-5].split("_")[-1])
                         if "_" in os.path.basename(s) else int(os.path.basename(s)[4:-5])):
            with open(fn, "r", encoding="utf-8") as f:
                rec = json.load(f)
            with open(idx_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"\n[Done] Summary index written to {idx_path}")