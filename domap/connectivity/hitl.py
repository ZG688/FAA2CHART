# -*- coding: utf-8 -*-
r"""
Human-in-the-Loop (HITL) interface.

Reviews the result of the previous step (rather than merely approving the next
action): after a selected sub-agent completes, its result is surfaced to the
operator, who can approve, reject, or correct it before the pipeline proceeds.

At each review point, an LLM generates a focused review question together with
2-4 selectable options (mapped to approve / reject / edit), so the interaction
adapts to the actual result of the current step instead of offering a fixed set
of buttons.

Implementation
--------------
- Node-level pause via LangGraph `interrupt_after=["tools"]` + `checkpointer` +
  `get_state()`; resume with `None` (the node is not re-run on resume).
- Review granularity per sub-agent via the `subagent_type` argument of the
  `task` tool, filtered by `review_subagents`.
- On `edit`, the corrected data is written back to the original JSON path (so
  downstream mapping reads the corrected data) and a `*_reviewed_*.json` copy is
  archived.
- Checkpoint: `InMemorySaver` by default; `SqliteSaver` when `db_path` is given
  (requires `langgraph-checkpoint-sqlite`).

Usage (QGIS Python console):
    from domap.connectivity.hitl import run_pipeline_with_hitl
    run_pipeline_with_hitl(user_request="...", review_subagents={"geocode"})
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
# Checkpointer
# ==================================================================
def _make_checkpointer(db_path: Optional[str] = None):
    if db_path:
        try:
            # QGIS-bundled Python's _sqlite3.pyd depends on sqlite3.dll in QGIS/bin
            qgis_bin = os.environ.get("QGIS_BIN_DIR") or os.path.join(
                os.environ.get("QGIS_PREFIX_PATH", ""), "bin"
            )
            if qgis_bin and os.path.isdir(qgis_bin):
                os.add_dll_directory(qgis_bin)
            import sqlite3  # noqa: E402
            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
            conn = sqlite3.connect(db_path, check_same_thread=False)
            return SqliteSaver(conn)
        except (ImportError, OSError) as e:
            print(f"[HITL] SqliteSaver unavailable ({e}), falling back to InMemorySaver")
    return InMemorySaver()


# ==================================================================
# Edit result: write corrected data as a new JSON
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


# Chinese virtual-directory names used by the framework, mapped to English for display
_CN_DIR_MAP = {
    "文本": "text", "图层": "layer", "临时": "temp", "输出": "output",
    "地图": "map", "结果": "result", "数据": "data", "输入": "input",
    "航报": "notam", "航路": "airway", "截图": "screenshot",
}


def _clean_path_for_display(path: str) -> str:
    """Translate Chinese directory names in a virtual path to English for display;
    unknown Chinese segments are replaced with 'out'."""
    p = str(path)
    for cn, en in _CN_DIR_MAP.items():
        p = p.replace(f"/{cn}/", f"/{en}/")
    p = re.sub(r"[\u4e00-\u9fff]+", "out", p)
    return p


def _concise_summary(summary: str, limit: int = 600) -> str:
    """Condense a long sub-agent report into key lines (path + statistics), in English."""
    s = str(summary)
    parts = []
    jp = _extract_json_path(s)
    if jp:
        parts.append(f"Output JSON path: {_clean_path_for_display(jp)}")

    # match both Chinese and English labels, then output in English
    stats = [
        ("Total waypoints", r"(?:总航路点|Total waypoints)"),
        ("Coordinates completed", r"(?:成功补全|Coordinates completed|Successfully geocoded)"),
        ("Skipped / failed", r"(?:跳过|失败|Skipped|Failed|unmatched)"),
    ]
    for label, pat in stats:
        m = re.search(rf"{pat}[^0-9]*?([0-9]+)", s, re.I)
        if m:
            parts.append(f"{label}: {m.group(1)}")

    if parts:
        return "\n".join(parts) + f"\n(full report {len(s)} chars, truncated)"
    if len(s) > limit:
        return s[:limit] + f"\n(full report {len(s)} chars, truncated)"
    return s


def _extract_json_path(summary: str) -> Optional[str]:
    """Extract a JSON virtual path (e.g. /text/xxx.json) from a sub-agent report."""
    for line in str(summary).splitlines():
        if re.search(r"geocoded_json_path|extracted_json_path|json_path", line, re.I):
            m = re.search(r"/[^\s`*]+", line)
            if m:
                return m.group(0).rstrip("*`：:")
    return None


def _apply_nl_edit(nl_instruction: str, subagent_type: str, summary: str) -> str:
    """Parse a natural-language correction into structured edits and apply them to the JSON."""
    current_vp = _extract_json_path(summary)
    host_path = to_host_path(current_vp) if current_vp else None

    data = None
    if host_path and os.path.exists(host_path):
        with open(host_path, encoding="utf-8") as f:
            data = json.load(f)

    model = build_model()
    parse_prompt = (
        "Given the user's natural-language correction instruction, generate structured edits "
        "for the airway JSON.\n"
        "JSON structure: top-level is an array of airways; each airway has airway_code and "
        "airway_point; each waypoint has name/region/type/position "
        "(position is [lat,lon] or [\"none\",\"none\"]).\n\n"
        f"User instruction: {nl_instruction}\n\n"
        "Output only a JSON array, each element of the form:\n"
        '[{"airway_code":"V6","find_name":"ABERDEEN","set_name":"GRAND ISLAND","set_position":[40.96,-98.32]}]\n'
        "Fields: find_name=current waypoint name to locate; set_name=new name (optional); "
        "set_position=new coordinates (optional). Output only the JSON array, no explanation or code fences."
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

    payload = data if data is not None else {}
    if host_path and data is not None:
        try:
            with open(host_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[HITL] Failed to overwrite {host_path}: {e}")

    os.makedirs(DEFAULT_QUEUE_DIR, exist_ok=True)
    copy_path = os.path.join(DEFAULT_QUEUE_DIR, f"{subagent_type}_reviewed_{uuid.uuid4().hex[:8]}.json")
    with open(copy_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return host_path or copy_path


# ==================================================================
# Build the HITL-enabled supervisor
# ==================================================================
def build_hitl_supervisor(
    model, subagents, checkpointer=None,
    interrupt_on: Optional[Dict[str, Any]] = None,
    extra_middleware: Optional[list] = None,
):
    """Build the supervisor (review uses node-level `interrupt_after=["tools"]`)."""
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
# Node-level review: extract pending items from the paused state
# ==================================================================
def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _extract_review_items_from_state(state, review_subagents):
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
            "description": f"Sub-agent [{subagent_type}] result (summary):\n{_concise_summary(content)}",
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
    for it, decision in zip(items, decisions):
        t = decision.get("type")
        print(f"[HITL] Decision: {t}  sub-agent={it['subagent_type']}")
        if t == "edit":
            nl = decision.get("nl_instruction")
            if nl:
                new_path = _apply_nl_edit(str(nl), it["subagent_type"], it["full_result"])
            else:
                edited = decision.get("edited_result") or ""
                new_path = _write_edited_result(it["subagent_type"], str(edited))
            print(f"[HITL] Edit complete, new JSON: {new_path}")
        elif t == "reject":
            print(f"[HITL] Rejected: {decision.get('message', '')}")


# ==================================================================
# HITL main loop
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
    """Run with HITL (node-level `interrupt_after=["tools"]`, main-thread review)."""
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
    reviewed: set = set()

    for iteration in range(1, max_iterations + 1):
        interrupted = False
        for chunk in supervisor.stream(
            payload, config=config, stream_mode="updates", interrupt_after=["tools"]
        ):
            chunks.append(chunk)
            if "__interrupt__" in chunk:
                state = supervisor.get_state(config)
                items = _extract_review_items_from_state(state, review_subagents)
                pending = [it for it in items if it["subagent_type"] not in reviewed]
                if pending:
                    print(f"[HITL] Round {iteration} — pending review: {[it['subagent_type'] for it in pending]}")
                    decisions = review_callback(_to_hitl_request(pending))
                    if len(decisions) != len(pending):
                        raise ValueError(f"Decision count {len(decisions)} != pending actions {len(pending)}")
                    _apply_decisions(pending, decisions)
                    reviewed.update(it["subagent_type"] for it in pending)
                else:
                    print(f"[HITL] Round {iteration} — no new review (already reviewed {sorted(reviewed)}), auto-approve")
                payload = None
                interrupted = True
                break
        if not interrupted:
            print(f"[HITL] Round {iteration} — graph finished (no interrupt), exiting normally")
            break
    else:
        print(f"[HITL] Iteration cap {max_iterations} reached, forcing stop")

    return chunks


# ==================================================================
# Render the pending review
# ==================================================================
def _render_hitl_request(hitl_request: Dict[str, Any]) -> str:
    lines = ["=" * 60, "The following step results require human review:", "=" * 60]
    for req in hitl_request.get("action_requests", []):
        lines.append(f"  Stage  : {req.get('subagent_type', req.get('name'))}")
        desc = req.get("description")
        if desc:
            lines.append(f"  Result :\n{desc}")
        lines.append("-" * 60)
    return "\n".join(lines)


# ==================================================================
# Review front-end 1: console
# ==================================================================
def console_reviewer(hitl_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        import sys as _sys
        _sys.stdin.fileno()
    except (AttributeError, ValueError, OSError, RuntimeError):
        raise RuntimeError("No interactive stdin (e.g. QGIS); use QtDialogReviewer or FileReviewQueue")

    print(_render_hitl_request(hitl_request))
    decisions = []
    for req in hitl_request.get("action_requests", []):
        name = req.get("subagent_type", req.get("name"))
        print(f"\nPending review: {name}")
        while True:
            c = input("  Decision (approve/edit/reject) [Enter=approve]: ").strip().lower() or "approve"
            if c in ("approve", "edit", "reject"):
                break
        if c == "approve":
            decisions.append({"type": "approve"})
        elif c == "edit":
            decisions.append({"type": "edit", "edited_result": input("  Corrected result: ").strip()})
        else:
            decisions.append({"type": "reject", "message": input("  Rejection note: ").strip() or f"User rejected {name}"})
    return decisions


# ==================================================================
# Review front-end 2: file queue
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
    """File-based review queue. No timeout by default — waits for a human decision."""

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
        print(f"\n[HITL] Pending request: {req_file}")
        print(f"[HITL] Write decisions to {dec_file}, e.g. {{'decisions': [{{'type': 'approve'}}]}}\n")

        deadline = time.time() + self.timeout if self.timeout else None
        while not os.path.exists(dec_file):
            if deadline and time.time() > deadline:
                n = len(hitl_request.get("action_requests", []))
                if self.default_decision == "approve":
                    decisions = [{"type": "approve"} for _ in range(n)]
                    print(f"[HITL] Timeout {self.timeout}s — auto-approve (demo)")
                else:
                    decisions = [{"type": "reject", "message": "Timed out, not reviewed"} for _ in range(n)]
                    print(f"[HITL] Timeout {self.timeout}s — auto-reject")
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
            raise RuntimeError("Failed to parse decision JSON")

        decisions = response.get("decisions", [])
        print(f"[HITL] Read {len(decisions)} decision(s), resuming...")
        return decisions


# ==================================================================
# Generate a step-specific review question + options (LLM)
# ==================================================================
def _generate_review_options(subagent_type: str, summary: str) -> Dict[str, Any]:
    """Use an LLM to generate a step-specific review question and options for the dialog."""
    model = build_model()
    prompt = (
        "You are helping a human operator review an intermediate result of an "
        "aeronautical-chart-generation multi-agent pipeline.\n\n"
        f"Stage: {subagent_type}\n"
        f"Result summary:\n{str(summary)[:2000]}\n\n"
        "Generate a focused review question for the operator and 2-4 concrete options "
        "they can choose from. Output only a JSON object of the form:\n"
        '{"question":"...", "options":[{"label":"...","action":"approve"}, '
        '{"label":"...","action":"reject"}, {"label":"...","action":"edit"}]}\n'
        'Allowed actions are exactly one of: "approve", "reject", "edit". '
        '"approve" = accept and continue; "reject" = redo this step; '
        '"edit" = the operator will provide a manual correction. '
        "Output only the JSON object, no explanation or code fences."
    )
    opts: Dict[str, Any] = {}
    try:
        resp = model.invoke(prompt)
        raw = str(resp.content).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        opts = json.loads(raw)
    except Exception:
        opts = {}
    if not isinstance(opts, dict) or not opts.get("options"):
        opts = {
            "question": f"Review the {subagent_type} result and decide how to proceed.",
            "options": [
                {"label": "Accept and continue", "action": "approve"},
                {"label": "Reject and redo this step", "action": "reject"},
                {"label": "Manually correct the result", "action": "edit"},
            ],
        }
    return opts


# ==================================================================
# Review front-end 3: QGIS Qt dialog (option-based)
# ==================================================================
class QtDialogReviewer:
    """QGIS Qt dialog review front-end (option-based)."""

    def __call__(self, hitl_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        from qgis.PyQt.QtWidgets import (
            QDialog, QVBoxLayout, QLabel, QRadioButton, QButtonGroup,
            QTextEdit, QPushButton, QInputDialog,
        )
        decisions = []
        for req in hitl_request.get("action_requests", []):
            subagent_type = req.get("subagent_type", req.get("name"))
            summary = req.get("description", "")   # English concise summary; avoid Chinese in the LLM prompt
            opts = _generate_review_options(subagent_type, summary)

            dialog = QDialog()
            dialog.setWindowTitle("Human-in-the-loop Review")
            dialog.resize(720, 560)
            layout = QVBoxLayout(dialog)

            # review question (bold heading)
            question = QLabel(opts.get("question", f"Review the {subagent_type} result"))
            question.setWordWrap(True)
            question.setStyleSheet("font-weight: bold;")
            layout.addWidget(question)

            # result summary (read-only)
            summary_box = QTextEdit()
            summary_box.setPlainText(req.get("description", ""))
            summary_box.setReadOnly(True)
            summary_box.setMaximumHeight(200)
            layout.addWidget(summary_box)

            # options (radio buttons)
            group = QButtonGroup(dialog)
            radios = []
            for i, opt in enumerate(opts.get("options", [])):
                rb = QRadioButton(opt.get("label", f"Option {i + 1}"))
                group.addButton(rb)
                layout.addWidget(rb)
                radios.append((rb, opt))
                if i == 0:
                    rb.setChecked(True)

            confirm_btn = QPushButton("Confirm")
            layout.addWidget(confirm_btn)

            holder: Dict[str, Any] = {"decision": None}

            def _confirm():
                for rb, opt in radios:
                    if rb.isChecked():
                        action = opt.get("action", "approve")
                        if action == "edit":
                            txt, ok = QInputDialog.getMultiLineText(
                                dialog, "Manual correction",
                                "Enter your feedback / correction:", "")
                            holder["decision"] = ({"type": "edit", "nl_instruction": txt}
                                                  if ok and txt.strip() else {"type": "approve"})
                        elif action == "reject":
                            holder["decision"] = {"type": "reject", "message": opt.get("label", "Rejected")}
                        else:
                            holder["decision"] = {"type": "approve"}
                        break
                dialog.accept()

            confirm_btn.clicked.connect(_confirm)
            dialog.exec_()

            decision = holder["decision"] or {"type": "reject", "message": "User closed the review window"}
            decisions.append(decision)

        return decisions


def _default_reviewer():
    try:
        import qgis  # noqa: F401
        return QtDialogReviewer()
    except ImportError:
        return FileReviewQueue().review


if __name__ == "__main__":
    run_pipeline_with_hitl(
        user_request="Parse the given document and generate a California airway connectivity map",
        review_callback=FileReviewQueue().review,
    )
