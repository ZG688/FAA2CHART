# Human-in-the-Loop (HITL) Interface

A human-in-the-loop review mechanism for the FAA2CHART multi-agent framework, added to address reviewer concern R5-3: the framework is autonomous end-to-end, but in safety-critical contexts (aviation) users must be able to verify or correct intermediate agent outputs (e.g., an incorrect waypoint geolocation).

## Key idea

Review the **result of the previous step**, not just approve the next action. After each sub-agent completes, its output is surfaced to the user, who can **approve**, **reject**, or **edit** before the pipeline proceeds.

## How it works

- Built on LangGraph's native `interrupt_after` + `checkpointer` + `get_state()`.
- The supervisor is compiled with a checkpointer and streamed with `interrupt_after=["tools"]`, so execution pauses *after* the tools node runs (on the main thread), without re-running the node on resume.
- Review granularity is per sub-agent via the `subagent_type` argument of the `task` tool (`event_extract` / `geocode` / `layer_loader` / `analysis` / `map_publish`), filtered by `review_subagents`.
- On `edit`, the corrected data is written back to the original JSON path (so downstream mapping reads the corrected data automatically) and a `*_reviewed_*.json` copy is archived.

## Requirements

- `domap`, `deepagents`, `langgraph`, `langchain-core` (already used by the framework)
- Optional: `langgraph-checkpoint-sqlite` for persistent checkpoints (otherwise `InMemorySaver`)

## Usage

```python
from domap.connectivity.hitl import run_pipeline_with_hitl

# Review only the geocoding stage (QGIS will use a Qt pop-up by default)
run_pipeline_with_hitl(
    user_request="解析航报并生成加州航路连通图",
    review_subagents={"geocode"},
    db_path="checkpoints.sqlite",   # optional: persistent SqliteSaver
)
```

Three review front-ends are provided:

| Front-end | Notes |
|-----------|-------|
| `QtDialogReviewer` | QGIS pop-up with **批准 / 反馈输入 / 驳回** buttons (default inside QGIS) |
| `FileReviewQueue` | File-based queue (`pending_request.json` → `decision.json`), suitable for scripts or an external MCP client |
| `console_reviewer` | Interactive `input()` (real terminal only) |

Decision types: `approve` (proceed), `reject` (ask the model to redo), `edit` (apply a natural-language correction, e.g. "把 ABERDEEN 改回 GRAND ISLAND，坐标改成 40.96, -98.32").

## Validation cases

`evaluation/03_ablation/architecture/hitl_validation.py` provides two reproducible cases.

### Case A — explicit fault (R5-4)

Injects a first-call crash into the `geocode` sub-agent and reports whether the fault is exposed, whether the coordinator re-delegates, and whether downstream tools are invoked prematurely.

```python
from hitl_validation import run_caseA
run_caseA()
```

Observed behavior: the fault is exposed, but the exception propagates through the `task` tool / ToolNode and terminates the pipeline (fail-fast); no automatic retry, and downstream mapping is not invoked prematurely.

### Case B — silent error + HITL (R5-3 / R5-4)

Tampers a waypoint identifier (e.g. `GRAND ISLAND` → `ABERDEEN`, both real NASR waypoints), then lets the user correct it at the HITL review point and verifies the corrected JSON is written.

```python
from hitl_validation import run_caseB
run_caseB("V6", "GRAND ISLAND", "ABERDEEN")
```

Observed behavior: the tampered identifier is silently accepted by geocoding, the user corrects it at the review dialog, and the corrected coordinates are written back to the JSON consumed downstream.

## Notes

- `InMemorySaver` only supports in-process resume. Use `db_path` with `langgraph-checkpoint-sqlite` for persistence across restarts.
- The pipeline enforces `recursion_limit=5000` and a `max_iterations` cap to bound runaway re-delegation loops.
