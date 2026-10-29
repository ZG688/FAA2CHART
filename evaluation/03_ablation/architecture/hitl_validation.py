# -*- coding: utf-8 -*-
"""
HITL 验证案例（案例A：显式故障；案例B：静默错误 + HITL）

案例A：显式故障 —— geocode 首次失败，观察「重试 / 停止 / 提前调用下游」（R5-4）
案例B：静默错误 + HITL —— 篡改航路点标识符，人工纠正，验证下游读取修正 JSON（R5-3 / R5-4）

依赖：本文件与 run_three_tasks.py 同目录（evaluation/03_ablation/architecture/），
      run_three_tasks.py 提供 _make_user_request / _make_cfg / _GEOCODE_INPUT / _JSON_DIR。

用法（QGIS Python 控制台）：
    import sys
    sys.path.insert(0, r"<path_to_evaluation>/03_ablation/architecture")
    from hitl_validation import run_caseA, run_caseB

    run_caseA()
    run_caseB("V6", "GRAND ISLAND", "ABERDEEN")   # 把 V6 里名为 GRAND ISLAND 的航路点改成 ABERDEEN
"""

from __future__ import annotations

import json
import os
import sys
import uuid

# 允许直接 import 同目录的 run_three_tasks（提供 _make_user_request / _make_cfg / _GEOCODE_INPUT / _JSON_DIR）
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
# 行为观测中间件：记录工具调用序列
# ==================================================================
class BehaviorObserver(AgentMiddleware):
    """记录每个工具调用的工具名 / 子智能体类型 / 执行状态。"""

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
# 故障注入（案例A）
# ==================================================================
def inject_geocode_fault(subagents, fail_times=1):
    """让 geocode 子智能体前 fail_times 次 invoke 抛异常（模拟显式故障）。"""
    geocode = next(s for s in subagents if s["name"] == "geocode")
    runnable = geocode["runnable"]
    orig = runnable.invoke
    cnt = {"n": 0}

    def faulty(*a, **kw):
        cnt["n"] += 1
        if cnt["n"] <= fail_times:
            raise RuntimeError(f"【案例A】模拟 geocode 显式故障（第 {cnt['n']} 次调用）")
        return orig(*a, **kw)

    runnable.invoke = faulty
    return cnt


# ==================================================================
# 标识符篡改（案例B）
# ==================================================================
def tamper_waypoint_identifier(json_path, airway_code, old_name, new_name, out_path=None):
    """把指定航路点标识符改成另一个真实 NASR 航路点（模拟名称关联错误）。

    返回 (篡改后 JSON 路径, 命中数量)。
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
# 案例A：显式故障
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
        print("[案例A] 流水线正常结束")
    except Exception as e:
        print(f"[案例A] 流水线异常终止: {type(e).__name__}: {e}")

    print(f"\n[案例A] geocode 总调用次数: {cnt['n']}")
    print("[案例A] 工具调用序列:")
    print(obs.report())

    geocode_calls = [e for e in obs.events if e["tool"] == "task" and e["sub"] == "geocode"]
    downstream = [e for e in obs.events if e["tool"] in ("create_connectivity_job_tool",)]
    exposed = any(e["status"].startswith("FAIL") for e in geocode_calls)
    retried = len(geocode_calls) >= 2 and geocode_calls[1]["status"] == "ok"

    print("\n[案例A] 结论:")
    print(f"  - 故障是否暴露给协调器: {'是' if exposed else '否'}")
    print(f"  - 协调器是否重新委派 geocode 并成功: {'是' if retried else '否'}")
    print(f"  - 失败后是否提前调用下游制图工具: {'是' if downstream else '否'}")
    return obs


# ==================================================================
# 案例B：静默错误 + HITL
# ==================================================================
def run_caseB(airway_code, old_name, new_name, use_dialog=True):
    """篡改航路点标识符 → 人工在 HITL 复核点纠正 → 验证修正 JSON 生成。

    use_dialog=True（默认）：QGIS 里用 Qt 弹窗，点「反馈输入」直接输入自然语言修复指令。
    use_dialog=False：文件队列，可复现/自动化。
    """
    # 1) 篡改标识符（写一个新 JSON，不覆盖原文件）
    tampered_path, hit = tamper_waypoint_identifier(
        _GEOCODE_INPUT, airway_code, old_name, new_name,
        out_path=os.path.join(_JSON_DIR, "parsed_routes_caseB_tampered.json"),
    )
    if hit == 0:
        print(f"[案例B] 未找到 airway_code={airway_code}, name={old_name}，请换一对标识符")
        return None

    # 2) 构造只指向篡改文件的 geocode 请求
    geocode_input_vp = to_virtual_path(tampered_path)
    user_req = (
        f"请检查并补全文件“{geocode_input_vp}”中航路点的空间坐标。\n"
        f"坐标只能从 CSV 表格中查找匹配，CSV 中没有的航路点直接跳过。\n"
    )

    print(f"\n[案例B] 篡改命中 {hit} 处：{old_name} -> {new_name}")
    print(f"[案例B] 篡改后输入: {tampered_path}")

    if use_dialog:
        # 不传 review_callback → QGIS 下自动用 Qt 弹窗复核
        run_pipeline_with_hitl(
            user_request=user_req,
            cfg=_make_cfg("simple"),
            review_subagents={"geocode"},
        )
        print("\n[案例B] 运行结束。弹窗点「反馈输入」后，修正会写入 hitl_queue\\geocode_reviewed_*.json")
    else:
        queue_dir = os.path.join(DEFAULT_QUEUE_DIR, "caseB")
        q = FileReviewQueue(queue_dir=queue_dir, timeout=0)
        run_pipeline_with_hitl(
            user_request=user_req,
            cfg=_make_cfg("simple"),
            review_subagents={"geocode"},
            review_callback=q.review,
        )
        print(f"\n[案例B] 运行结束。请检查 {queue_dir} 下的 geocode_reviewed_*.json")

    return tampered_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="HITL 验证案例")
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
