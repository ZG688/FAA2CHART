# -*- coding: utf-8 -*-
"""
Human-in-the-Loop (HITL) 人机交互模块 —— 回应审稿意见 R5-3。

设计要点
--------
1. 复核「上一步结果」而非「审批下一步」：用 LangGraph 节点级 `interrupt_after=["tools"]`
   在 tools 节点执行完成后暂停，通过 `get_state()` 读取「上一步的执行结果」呈现给用户复核。

2. 恢复：`interrupt_after` 的标准恢复方式是传 `None`。节点后暂停、恢复不重跑节点，
   因此无需幂等缓存（子智能体不会重复执行）。

3. 编辑落到真实数据：用户修正后的结果覆盖写回原 JSON 路径（下游制图读同一路径，
   自动拿到修正数据），并另存一份 `*_reviewed_*.json` 副本留档。

4. 不自动放行：文件队列默认无超时、超时默认驳回；Qt 弹窗关窗默认驳回。

5. 检查点：默认 InMemorySaver；装了 langgraph-checkpoint-sqlite 且给 db_path 则用 SqliteSaver。

审核粒度：从 task 工具的 subagent_type 参数区分 event_extract / geocode / layer_loader /
analysis / map_publish，按 review_subagents 过滤。

依赖：
    domap（connectivity.multi_agents / virtual_paths）
    deepagents（create_deep_agent / FilesystemBackend）
    langgraph（checkpoint / store）
    langchain-core（messages）

用法（QGIS Python 控制台）：
    from domap.connectivity.hitl import run_pipeline_with_hitl, QtDialogReviewer
    run_pipeline_with_hitl(user_request="解析航报并生成加州航路连通图",
                           review_subagents={"geocode"})
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

from domap.connectivity.multi_agents import (
    build_model,
    build_subagents,
    build_user_task_from_natural_language,
    get_supervisor_tools,
    SUPERVISOR_PROMPT,
)
from domap.connectivity.virtual_paths import HOST_ROOT, to_host_path
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.store.memory import InMemoryStore

DEFAULT_QUEUE_DIR = os.path.join(os.getcwd(), "hitl_queue")
DEFAULT_REVIEW_SUBAGENTS = {"event_extract", "geocode", "layer_loader", "analysis", "map_publish"}
ReviewCallback = Callable[[Dict[str, Any]], List[Dict[str, Any]]]


# ==================================================================
# 检查点
# ==================================================================
def _make_checkpointer(db_path: Optional[str] = None):
    if db_path:
        try:
            # QGIS 自带 Python 的 _sqlite3.pyd 依赖 sqlite3.dll，需加入其 bin 目录
            qgis_bin = os.environ.get("QGIS_BIN_DIR") or os.path.join(
                os.environ.get("QGIS_PREFIX_PATH", ""), "bin"
            )
            if qgis_bin and os.path.isdir(qgis_bin):
                os.add_dll_directory(qgis_bin)
            import sqlite3  # noqa: E402
            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
            # 直接建连接 + saver（from_conn_string 是 context manager，会随 with 关闭）
            conn = sqlite3.connect(db_path, check_same_thread=False)
            return SqliteSaver(conn)
        except (ImportError, OSError) as e:
            print(f"[HITL] SqliteSaver 不可用（{e}），回退 InMemorySaver（仅进程内）")
    return InMemorySaver()


# ==================================================================
# 编辑结果：把修正数据写成新 JSON
# ==================================================================
def _write_edited_result(subagent_type: str, edited_content: str) -> str:
    os.makedirs(DEFAULT_QUEUE_DIR, exist_ok=True)
    try:
        parsed = json.loads(edited_content)
        content = json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        content = edited_content
    path = os.path.join(DEFAULT_QUEUE_DIR, f"{subagent_type}_reviewed_{uuid.uuid4().hex[:8]}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _concise_summary(summary: str, limit: int = 400) -> str:
    """把子智能体的长报告压成几行关键信息（路径 + 统计），不显示讨论性文字。"""
    s = str(summary)
    parts = []
    jp = _extract_json_path(s)
    if jp:
        parts.append(f"输出 JSON 路径: {jp}")
    for label in ("总航路点", "成功补全"):
        m = re.search(rf"{label}[^0-9]*?([0-9]+)", s)
        if m:
            parts.append(f"{label}: {m.group(1)}")
    if parts:
        return "\n".join(parts) + f"\n（完整报告 {len(s)} 字符，已截断）"
    if len(s) > limit:
        return s[:limit] + f"\n（完整报告 {len(s)} 字符，已截断）"
    return s


def _extract_json_path(summary: str) -> Optional[str]:
    """从子智能体报告里提取 JSON 虚拟路径（如 /文本/xxx.json）。"""
    for line in str(summary).splitlines():
        if re.search(r"geocoded_json_path|extracted_json_path|json_path", line, re.I):
            m = re.search(r"/[^\s`*]+", line)
            if m:
                return m.group(0).rstrip("*`：:")
    return None


def _apply_nl_edit(nl_instruction: str, subagent_type: str, summary: str) -> str:
    """用 LLM 把自然语言修复指令解析成结构化修改，应用到当前 JSON，写新文件。"""
    # 1) 从 summary 提取当前 JSON 的虚拟路径并转宿主路径
    current_vp = _extract_json_path(summary)
    host_path = to_host_path(current_vp) if current_vp else None

    data = None
    if host_path and os.path.exists(host_path):
        with open(host_path, encoding="utf-8") as f:
            data = json.load(f)

    # 2) LLM 解析自然语言 → 结构化修改指令
    model = build_model()
    parse_prompt = (
        "根据用户的自然语言修复指令，对航路 JSON 生成结构化修改。\n"
        "JSON 结构：顶层是航路数组，每条含 airway_code 和 airway_point；"
        "每个航路点含 name/region/type/position（position 为 [lat,lon] 或 [\"none\",\"none\"]）。\n\n"
        f"用户修复指令：{nl_instruction}\n\n"
        "请只输出一个 JSON 数组，每个元素形如：\n"
        '[{"airway_code":"V6","find_name":"ABERDEEN","set_name":"GRAND ISLAND","set_position":[40.96,-98.32]}]\n'
        "字段含义：find_name=要查找的当前航路点名；set_name=改成的新名字（可省略）；"
        "set_position=改成的新坐标（可省略）。只输出 JSON 数组本身，不要解释文字或代码块标记。"
    )
    resp = model.invoke(parse_prompt)
    raw = str(resp.content).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        fixes = json.loads(raw)
    except json.JSONDecodeError:
        s = raw.find("[")
        e = raw.rfind("]")
        fixes = json.loads(raw[s:e + 1]) if s >= 0 and e > s else []

    # 3) 应用修改
    if data is not None and fixes:
        for fix in fixes:
            ac = fix.get("airway_code")
            fn = fix.get("find_name")
            for r in data:
                if ac and r.get("airway_code") != ac:
                    continue
                for p in r.get("airway_point", []):
                    if p.get("name") == fn:
                        if fix.get("set_name"):
                            p["name"] = fix["set_name"]
                        if fix.get("set_position"):
                            p["position"] = fix["set_position"]

    # 4) 写回：覆盖原路径（下游读同一路径，自动拿到修正数据）；
    #    再留一份 reviewed 副本（满足「新的 JSON 版本」的留档要求）
    payload = data if data is not None else {}
    if host_path and data is not None:
        try:
            with open(host_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[HITL] 覆盖写回 {host_path} 失败: {e}")

    os.makedirs(DEFAULT_QUEUE_DIR, exist_ok=True)
    copy_path = os.path.join(DEFAULT_QUEUE_DIR, f"{subagent_type}_reviewed_{uuid.uuid4().hex[:8]}.json")
    with open(copy_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 返回原路径（下游据此读取修正后的数据），若无原路径则返回副本
    return host_path or copy_path


# ==================================================================
# 构建带 HITL 的主控
# ==================================================================
def build_hitl_supervisor(
    model, subagents, checkpointer=None,
    interrupt_on: Optional[Dict[str, Any]] = None,
    extra_middleware: Optional[list] = None,
):
    """构建 supervisor（复核采用节点级 `interrupt_after=["tools"]`，不在工具线程内 interrupt）。"""
    middleware = list(extra_middleware or [])

    agent = create_deep_agent(
        model=model,
        tools=get_supervisor_tools(),
        backend=FilesystemBackend(root_dir=str(HOST_ROOT), virtual_mode=True),
        store=InMemoryStore(),
        system_prompt=SUPERVISOR_PROMPT,
        subagents=subagents,
        middleware=middleware,
        checkpointer=checkpointer,
        interrupt_on=interrupt_on,
    )
    return agent.with_config({"recursion_limit": 5000})


# ==================================================================
# 节点级复核：从暂停状态提取待审项
# ==================================================================
def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _extract_review_items_from_state(state, review_subagents):
    """从 interrupt_after 暂停后的状态里，提取本轮需要复核的 task 委派结果。"""
    messages = state.values.get("messages", [])
    ai_msg = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            ai_msg = m
            break
    if ai_msg is None:
        return []

    items = []
    for tc in ai_msg.tool_calls:
        if tc.get("name") != "task":
            continue
        subagent_type = tc.get("args", {}).get("subagent_type", "")
        if review_subagents and subagent_type not in review_subagents:
            continue
        content = ""
        for m in messages:
            if isinstance(m, AIMessage) or not hasattr(m, "tool_call_id"):
                continue
            if getattr(m, "tool_call_id", None) == tc.get("id"):
                content = _to_text(m.content)
                break
        items.append({
            "name": "task",
            "subagent_type": subagent_type,
            "args": tc.get("args", {}),
            "tool_call_id": tc.get("id"),
            "description": f"子智能体 [{subagent_type}] 执行结果（摘要）：\n{_concise_summary(content)}",
            "full_result": content,
        })
    return items


def _to_hitl_request(items):
    return {
        "action_requests": [{
            "name": it["name"],
            "subagent_type": it["subagent_type"],
            "args": it["args"],
            "description": it["description"],
            "full_result": it["full_result"],
        } for it in items],
        "review_configs": [{
            "action_name": it["name"],
            "allowed_decisions": ["approve", "edit", "reject"],
        } for it in items],
    }


def _apply_decisions(items, decisions):
    """按决策处理副作用（编辑写新 JSON）；恢复由主循环用 None 完成。"""
    for it, decision in zip(items, decisions):
        t = decision.get("type")
        print(f"[HITL] 决策: {t}  子智能体={it['subagent_type']}")
        if t == "edit":
            nl = decision.get("nl_instruction")
            if nl:
                new_path = _apply_nl_edit(str(nl), it["subagent_type"], it["full_result"])
            else:
                edited = decision.get("edited_result") or ""
                new_path = _write_edited_result(it["subagent_type"], str(edited))
            print(f"[HITL] 编辑完成，新 JSON: {new_path}")
        elif t == "reject":
            print(f"[HITL] 驳回：{decision.get('message', '')}")


# ==================================================================
# HITL 运行主循环
# ==================================================================
def run_pipeline_with_hitl(
    user_request: str,
    cfg=None,
    extra_requirement: str = "",
    review_callback: Optional[ReviewCallback] = None,
    review_subagents: Optional[set] = None,
    interrupt_on: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
    thread_id: Optional[str] = None,
    profile=None,
    source_document: Optional[str] = None,
    max_iterations: int = 30,
) -> List[Dict[str, Any]]:
    """以 HITL 方式运行（节点级 `interrupt_after=["tools"]`，主线程复核，避免二次中断）。"""
    if review_callback is None:
        review_callback = _default_reviewer()
    if review_subagents is None:
        review_subagents = DEFAULT_REVIEW_SUBAGENTS

    model = build_model()
    subagents = build_subagents(model, profile=profile)
    checkpointer = _make_checkpointer(db_path)
    supervisor = build_hitl_supervisor(
        model, subagents, checkpointer,
        interrupt_on=interrupt_on,
    )

    user_task = build_user_task_from_natural_language(
        user_request=user_request, cfg=cfg, extra_requirement=extra_requirement,
        profile=profile, source_document=source_document,
    )
    input_data = {"messages": [{"role": "user", "content": user_task}]}

    if thread_id is None:
        thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    chunks: List[Dict[str, Any]] = []
    payload = input_data
    reviewed: set = set()  # 已复核过的子智能体类型，避免模型重复委派导致反复弹窗

    for iteration in range(1, max_iterations + 1):
        interrupted = False
        for chunk in supervisor.stream(
            payload, config=config, stream_mode="updates", interrupt_after=["tools"]
        ):
            chunks.append(chunk)
            if "__interrupt__" in chunk:
                state = supervisor.get_state(config)
                items = _extract_review_items_from_state(state, review_subagents)
                # 过滤掉已复核过的子智能体类型（自动放行）
                pending = [it for it in items if it["subagent_type"] not in reviewed]
                if pending:
                    print(f"[HITL] 第{iteration}轮 待复核子智能体: {[it['subagent_type'] for it in pending]}")
                    decisions = review_callback(_to_hitl_request(pending))
                    if len(decisions) != len(pending):
                        raise ValueError(f"决策数 {len(decisions)} 与待审动作数 {len(pending)} 不一致")
                    _apply_decisions(pending, decisions)
                    reviewed.update(it["subagent_type"] for it in pending)
                else:
                    print(f"[HITL] 第{iteration}轮 无新待审（已复核过 {sorted(reviewed)}），自动放行")
                # interrupt_after 的标准恢复：传 None
                payload = None
                interrupted = True
                break
        if not interrupted:
            print(f"[HITL] 第{iteration}轮 图执行结束（无中断），正常退出")
            break
    else:
        print(f"[HITL] 达到迭代上限 {max_iterations}，强制结束（模型可能陷入重复委派循环）")

    return chunks


# ==================================================================
# 渲染待审内容
# ==================================================================
def _render_hitl_request(hitl_request: Dict[str, Any]) -> str:
    lines = ["=" * 60, "以下步骤的执行结果需要人工复核：", "=" * 60]
    for req in hitl_request.get("action_requests", []):
        lines.append(f"  阶段  : {req.get('subagent_type', req.get('name'))}")
        desc = req.get("description")
        if desc:
            lines.append(f"  结果  :\n{desc}")
        lines.append("-" * 60)
    return "\n".join(lines)


# ==================================================================
# 评审接口 1：控制台
# ==================================================================
def console_reviewer(hitl_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        import sys as _sys
        _sys.stdin.fileno()
    except (AttributeError, ValueError, OSError, RuntimeError):
        raise RuntimeError("当前环境无交互 stdin（如 QGIS），请改用 QtDialogReviewer 或 FileReviewQueue")

    print(_render_hitl_request(hitl_request))
    decisions = []
    for req in hitl_request.get("action_requests", []):
        name = req.get("subagent_type", req.get("name"))
        print(f"\n待复核: {name}")
        while True:
            c = input("  决策(approve/edit/reject)[回车=approve]: ").strip().lower() or "approve"
            if c in ("approve", "edit", "reject"):
                break
        if c == "approve":
            decisions.append({"type": "approve"})
        elif c == "edit":
            decisions.append({"type": "edit", "edited_result": input("  修正后的结果: ").strip()})
        else:
            decisions.append({"type": "reject", "message": input("  驳回说明: ").strip() or f"用户驳回 {name}"})
    return decisions


# ==================================================================
# 评审接口 2：文件队列
# ==================================================================
def _wait_poll(seconds: float) -> None:
    try:
        from qgis.PyQt.QtWidgets import QApplication
    except ImportError:
        time.sleep(seconds)
        return
    end = time.time() + seconds
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.05)


class FileReviewQueue:
    """文件队列评审接口。默认无超时——关键审核一直等人工决策，不自动放行。"""

    def __init__(self, queue_dir: str = None, poll_interval: float = 1.0,
                 timeout: Optional[float] = None, default_decision: str = "reject"):
        self.queue_dir = queue_dir or DEFAULT_QUEUE_DIR
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.default_decision = default_decision
        os.makedirs(self.queue_dir, exist_ok=True)

    def review(self, hitl_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        req_file = os.path.join(self.queue_dir, "pending_request.json")
        dec_file = os.path.join(self.queue_dir, "decision.json")

        with open(req_file, "w", encoding="utf-8") as f:
            json.dump(hitl_request, f, ensure_ascii=False, indent=2)
        if os.path.exists(dec_file):
            os.remove(dec_file)

        print(_render_hitl_request(hitl_request))
        print(f"\n[HITL] 待审请求: {req_file}")
        print(f"[HITL] 请在 {dec_file} 写入决策，如 {{'decisions': [{{'type': 'approve'}}]}}\n")

        deadline = time.time() + self.timeout if self.timeout else None
        while not os.path.exists(dec_file):
            if deadline and time.time() > deadline:
                n = len(hitl_request.get("action_requests", []))
                if self.default_decision == "approve":
                    decisions = [{"type": "approve"} for _ in range(n)]
                    print(f"[HITL] 超时 {self.timeout}s 自动放行（演示）")
                else:
                    decisions = [{"type": "reject", "message": "超时未审核"} for _ in range(n)]
                    print(f"[HITL] 超时 {self.timeout}s 自动驳回")
                return decisions
            _wait_poll(self.poll_interval)

        for _ in range(10):
            try:
                with open(dec_file, "r", encoding="utf-8") as f:
                    response = json.load(f)
                break
            except json.JSONDecodeError:
                time.sleep(self.poll_interval)
        else:
            raise RuntimeError("决策文件 JSON 解析失败")

        decisions = response.get("decisions", [])
        print(f"[HITL] 已读取 {len(decisions)} 条决策，恢复执行...")
        return decisions


# ==================================================================
# 评审接口 3：QGIS Qt 弹窗
# ==================================================================
class QtDialogReviewer:
    """QGIS Qt 弹窗评审接口。"""

    def __call__(self, hitl_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        from qgis.PyQt.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QInputDialog,
        )
        n = len(hitl_request.get("action_requests", []))
        dialog = QDialog()
        dialog.setWindowTitle("人工复核（HITL）")
        dialog.resize(720, 480)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setPlainText(_render_hitl_request(hitl_request))
        text.setReadOnly(True)
        layout.addWidget(text)
        btns = QHBoxLayout()
        approve_btn = QPushButton("批准")
        edit_btn = QPushButton("反馈输入")
        reject_btn = QPushButton("驳回")
        btns.addWidget(approve_btn)
        btns.addWidget(edit_btn)
        btns.addWidget(reject_btn)
        layout.addLayout(btns)

        result: Dict[str, Any] = {"decisions": []}

        def _approve():
            result["decisions"] = [{"type": "approve"} for _ in range(n)]
            dialog.accept()

        def _reject():
            msg, _ = QInputDialog.getText(dialog, "驳回", "驳回说明：")
            result["decisions"] = [{"type": "reject", "message": msg or "用户驳回"} for _ in range(n)]
            dialog.accept()

        def _edit():
            if n == 0:
                return
            req = hitl_request["action_requests"][0]
            name = req.get("subagent_type", req.get("name"))
            txt, ok = QInputDialog.getMultiLineText(
                dialog, "反馈输入",
                f"阶段 [{name}] —— 请输入你的反馈/修复意见：",
                "",
            )
            if not ok or not txt.strip():
                return
            result["decisions"] = [{"type": "edit", "nl_instruction": txt}] + \
                                  [{"type": "approve"} for _ in range(n - 1)]
            dialog.accept()

        approve_btn.clicked.connect(_approve)
        edit_btn.clicked.connect(_edit)
        reject_btn.clicked.connect(_reject)
        dialog.exec_()

        if not result["decisions"]:
            result["decisions"] = [{"type": "reject", "message": "用户关闭复核窗口"} for _ in range(n)]
        return result["decisions"]


def _default_reviewer():
    try:
        import qgis  # noqa: F401
        return QtDialogReviewer()
    except ImportError:
        return FileReviewQueue().review


if __name__ == "__main__":
    run_pipeline_with_hitl(
        user_request="解析给定航报并生成加州航路连通图",
        review_callback=FileReviewQueue().review,
    )
