# -*- coding: utf-8 -*-
"""
HITL validation cases (Case A: explicit fault; Case B: silent error + HITL)

Case A: explicit fault — geocode fails on the first call; observe whether it retries,
        stops, or invokes downstream tools prematurely (R5-4).
Case B: silent error + HITL — tamper a waypoint identifier, correct it manually at the
        HITL review point, and verify the corrected JSON is written (R5-3 / R5-4).

Dependency: this file lives alongside run_three_tasks.py
(evaluation/03_ablation/architecture/), which provides _make_user_request / _make_cfg /
_GEOCODE_INPUT / _JSON_DIR.

Usage (QGIS Python console):
    import sys
    sys.path.insert(0, r"<path_to_evaluation>/03_ablation/architecture")
    from hitl_validation import run_caseA, run_caseB

    run_caseA()
    run_caseB("V6", "GRAND ISLAND", "ABERDEEN")   # rename the waypoint GRAND ISLAND to ABERDEEN in V6
"""

from __future__ import annotations

import json
import os
import sys
import uuid

# allow importing the sibling run_three_tasks (provides _make_user_request / _make_cfg / _GEOCODE_INPUT / _JSON_DIR)
try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _THIS_DIR = os.getcwd()
if _THIS_DIR and _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from langchain.agents.middleware import AgentMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from domap.connectivity.multi_agents import build_model, build_subagents
from domap.connectivity.virtual_paths import to_virtual_path
from domap.connectivity.hitl import (
    build_hitl_supervisor,
    run_pipeline_with_hitl,
    FileReviewQueue,
    DEFAULT_QUEUE_DIR,
)
from run_three_tasks import _make_user_request, _make_cfg, _GEOCODE_INPUT, _JSON_DIR


# ==================================================================
# Behavior observation middleware: records the tool-call sequence
# ==================================================================
class BehaviorObserver(AgentMiddleware):
    """Record each tool call's tool name / sub-agent type / execution status."""

    def __init__(self):
        self.events = []

    def wrap_tool_call(self, request, handler):
        tc = request.tool_call
        name = tc["name"]
        sub = tc.get("args", {}).get("subagent_type", "")
        ev = {"tool": name, "sub": sub, "status": None}
        self.events.append(ev)
        try:
            result = handler(request)
            ev["status"] = "ok"
        except Exception as e:
            ev["status"] = f"FAIL({type(e).__name__}: {e})"
            raise
        return result

    def report(self):
        return "\n".join(
            f"  {i}. {e['tool']} sub={e['sub'] or '-'} -> {e['status']}"
            for i, e in enumerate(self.events, 1)
        )


# ==================================================================
# Fault injection (Case A)
# ==================================================================
def inject_geocode_fault(subagents, fail_times=1):
    """Make the geocode sub-agent raise on the first `fail_times` invokes (simulated fault)."""
    geocode = next(s for s in subagents if s["name"] == "geocode")
    runnable = geocode["runnable"]
    orig = runnable.invoke
    cnt = {"n": 0}

    def faulty(*a, **kw):
        cnt["n"] += 1
        if cnt["n"] <= fail_times:
            raise RuntimeError(f"[Case A] simulated explicit geocode fault (call #{cnt['n']})")
        return orig(*a, **kw)

    runnable.invoke = faulty
    return cnt


# ==================================================================
# Identifier tampering (Case B)
# ==================================================================
def tamper_waypoint_identifier(json_path, airway_code, old_name, new_name, out_path=None):
    """Replace a waypoint identifier with another real NASR waypoint (simulated name-association error).

    Returns (tampered JSON path, number of hits).
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    hit = 0
    for r in data:
        if r.get("airway_code") == airway_code:
            for p in r.get("airway_point", []):
                if p.get("name") == old_name:
                    p["name"] = new_name
                    hit += 1
    out_path = out_path or json_path.replace(".json", "_tampered.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path, hit


# ==================================================================
# Case A: explicit fault
# ==================================================================
def run_caseA(fail_times=1):
    model = build_model()
    subagents = build_subagents(model)
    cnt = inject_geocode_fault(subagents, fail_times=fail_times)

    obs = BehaviorObserver()
    checkpointer = InMemorySaver()
    supervisor = build_hitl_supervisor(
        model, subagents, checkpointer,
        extra_middleware=[obs],
    )

    task = _make_user_request("simple")
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    payload = {"messages": [{"role": "user", "content": task}]}

    try:
        for _ in supervisor.stream(payload, config=config, stream_mode="updates"):
            pass
        print("[Case A] Pipeline finished normally")
    except Exception as e:
        print(f"[Case A] Pipeline aborted: {type(e).__name__}: {e}")

    print(f"\n[Case A] geocode total calls: {cnt['n']}")
    print("[Case A] Tool call sequence:")
    print(obs.report())

    geocode_calls = [e for e in obs.events if e["tool"] == "task" and e["sub"] == "geocode"]
    downstream = [e for e in obs.events if e["tool"] in ("create_connectivity_job_tool",)]
    exposed = any(e["status"].startswith("FAIL") for e in geocode_calls)
    retried = len(geocode_calls) >= 2 and geocode_calls[1]["status"] == "ok"

    print("\n[Case A] Conclusion:")
    print(f"  - Fault exposed to coordinator: {'yes' if exposed else 'no'}")
    print(f"  - Coordinator re-delegated geocode and succeeded: {'yes' if retried else 'no'}")
    print(f"  - Downstream mapping tool invoked prematurely: {'yes' if downstream else 'no'}")
    return obs


# ==================================================================
# Case B: silent error + HITL
# ==================================================================
def run_caseB(airway_code, old_name, new_name, use_dialog=True):
    """Tamper a waypoint identifier → correct it manually at the HITL review point → verify the corrected JSON.

    use_dialog=True (default): QGIS uses a Qt dialog; select "Manual correction" to enter a
                               natural-language fix.
    use_dialog=False: file queue, reproducible/automated.
    """
    # 1) tamper the identifier (write a new JSON, do not overwrite the original)
    tampered_path, hit = tamper_waypoint_identifier(
        _GEOCODE_INPUT, airway_code, old_name, new_name,
        out_path=os.path.join(_JSON_DIR, "parsed_routes_caseB_tampered.json"),
    )
    if hit == 0:
        print(f"[Case B] No waypoint found with airway_code={airway_code}, name={old_name}; pick another pair")
        return None

    # 2) build a geocode request that points only at the tampered file
    geocode_input_vp = to_virtual_path(tampered_path)
    user_req = (
        f"Check and complete the spatial coordinates of the waypoints in \"{geocode_input_vp}\".\n"
        f"Coordinates may only be matched from the CSV table; waypoints not found in the CSV are skipped.\n"
    )

    print(f"\n[Case B] Tampered {hit} location(s): {old_name} -> {new_name}")
    print(f"[Case B] Tampered input: {tampered_path}")

    if use_dialog:
        # no review_callback → QGIS automatically uses the Qt dialog
        run_pipeline_with_hitl(
            user_request=user_req,
            cfg=_make_cfg("simple"),
            review_subagents={"geocode"},
        )
        print("\n[Case B] Finished. After choosing 'Manual correction' in the dialog, "
              "the fix is written to hitl_queue\\geocode_reviewed_*.json")
    else:
        queue_dir = os.path.join(DEFAULT_QUEUE_DIR, "caseB")
        q = FileReviewQueue(queue_dir=queue_dir, timeout=0)
        run_pipeline_with_hitl(
            user_request=user_req,
            cfg=_make_cfg("simple"),
            review_subagents={"geocode"},
            review_callback=q.review,
        )
        print(f"\n[Case B] Finished. Check {queue_dir} for geocode_reviewed_*.json")

    return tampered_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="HITL validation cases")
    p.add_argument("--case", choices=["A", "B"], default="A")
    p.add_argument("--airway_code", default="V6")
    p.add_argument("--old_name", default="GRAND ISLAND")
    p.add_argument("--new_name", default="ABERDEEN")
    p.add_argument("--use_dialog", action="store_true")
    args = p.parse_args()

    if args.case == "A":
        run_caseA()
    else:
        run_caseB(args.airway_code, args.old_name, args.new_name, use_dialog=args.use_dialog)
