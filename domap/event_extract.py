# -*- coding: utf-8 -*-
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests

from langchain.tools import tool


# =============================
# Configuration
# =============================
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"  # <-- Replace with your key
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

INPUT_FILE = "PATH_TO_CLEAN_JSON"  # <-- Replace with actual path
OUTPUT_FILE = "PATH_TO_OUTPUT_JSON"  # <-- Replace with actual path


MAX_PROMPT_CHARS = 12000
MAX_SINGLE_ITEM = 9000
MAX_RETRY = 2   # auto-retry count

MINERU_TOKEN = "YOUR_MINERU_TOKEN"  # <-- Replace with your token
USER_TOKEN = "YOUR_USER_TOKEN"  # <-- Replace with your token
SAVE_DIR = "PATH_TO_SAVE_DIR"  # <-- Replace with actual directory


# =============================
# Domain Configuration (default FAA airways)
# =============================
from domap.domain_profile import (
    DomainProfile,
    FAA_AIRWAY_PROFILE,
    USCG_LNM_PROFILE,
    PROFILES,
)

DEFAULT_PROFILE = FAA_AIRWAY_PROFILE

# Backward compatibility: external code referencing these names still works
PROMPT_TEMPLATE = DEFAULT_PROFILE.prompt_template


def _resolve_profile(profile: Optional[Any]) -> DomainProfile:
    """Accept a DomainProfile instance, profile name string, or None (use default)."""
    if profile is None:
        return DEFAULT_PROFILE
    if isinstance(profile, DomainProfile):
        return profile
    if isinstance(profile, str):
        if profile not in PROFILES:
            raise ValueError(
                f"Unknown domain profile {profile!r}, options: {sorted(PROFILES)}"
            )
        return PROFILES[profile]
    raise TypeError(f"profile must be DomainProfile / str / None, got {type(profile)}")


# =============================
# state helpers
# =============================
def merge_state(state: Optional[dict] = None, **kwargs) -> dict:
    """
    Merge shared state.
    None values will not overwrite existing fields.
    """
    new_state = dict(state or {})
    for k, v in kwargs.items():
        if v is not None:
            new_state[k] = v
    return new_state

# =============================
# 1. MinerU headers
# =============================
def mineru_headers():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_TOKEN}",
    }
    if USER_TOKEN:
        headers["token"] = USER_TOKEN
    return headers


# =============================
# 2. Create task
# =============================
def create_task(file_url: str, model_version: str = "vlm") -> dict:
    """
    Create a MinerU parsing task, return structured result.
    """
    url = "https://mineru.net/api/v4/extract/task"
    data = {
        "url": file_url,
        "model_version": model_version,
    }

    res = requests.post(url, headers=mineru_headers(), json=data, timeout=30)
    print("create_task:", res.status_code, res.text[:500])
    res.raise_for_status()

    result = res.json()
    if result.get("code") != 0:
        raise Exception(f"Failed to create task: {result}")

    task_id = result["data"]["task_id"]
    return {
        "file_url": file_url,
        "model_version": model_version,
        "task_id": task_id,
    }


# =============================
# 2.1 Local file upload (get parseable URL)
# =============================

# serveo.net tunnel base URL (requires manual SSH tunnel + HTTP server)
SERVEO_BASE = None  # e.g. "https://xxxx.serveousercontent.com"

def upload_local_file(local_path: str) -> str:
    """
    Expose a local file as a publicly accessible URL for MinerU parsing.

    Prefer serveo tunnel; fall back to MinerU built-in batch upload if unavailable
    (the fallback may fail because the pre-signed URL only supports PUT).
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file does not exist: {local_path}")

    name = local_path.name

    # Priority: serveo tunnel
    if SERVEO_BASE:
        import shutil
        http_dir = Path("PATH_TO_HTTP_DIR")  # <-- Replace with actual directory
        dest = http_dir / name
        if not dest.exists() or dest.stat().st_size != local_path.stat().st_size:
            shutil.copy2(local_path, dest)
        url = SERVEO_BASE.rstrip("/") + "/" + name
        print(f"upload_local_file: serveo -> {url}")
        return url

    # Fallback: MinerU built-in batch upload (pre-signed URL is PUT-only, may not work)
    url = "https://mineru.net/api/v4/file-urls/batch"
    payload = {"files": [{"name": name}]}

    res = requests.post(url, headers=mineru_headers(), json=payload, timeout=30)
    print("upload_local_file: request upload URL", res.status_code, res.text[:300])
    res.raise_for_status()

    result = res.json()
    if result.get("code") != 0:
        raise Exception(f"Failed to request upload URL: {result}")

    batch_id = result["data"]["batch_id"]
    put_url = result["data"]["file_urls"][0]

    with open(local_path, "rb") as f:
        data = f.read()

    put_res = requests.put(put_url, data=data, timeout=300)
    print("upload_local_file: PUT upload", put_res.status_code)
    put_res.raise_for_status()

    print(f"upload_local_file: {name} -> upload successful (batch_id={batch_id})")
    return put_url

# =============================
# 3. Wait for task completion and download results
# =============================
def wait_and_download(
    task_id: str,
    save_dir: str = SAVE_DIR,
    interval: int = 5,
    timeout: int = 600,
) -> dict:
    query_url = f"https://mineru.net/api/v4/extract/task/{task_id}"

    start = time.time()
    full_zip_url = None

    while time.time() - start < timeout:
        res = requests.get(query_url, headers=mineru_headers(), timeout=30)
        print("query_task:", res.status_code, res.text[:500])
        res.raise_for_status()

        result = res.json()
        if result.get("code") != 0:
            raise Exception(f"Failed to query task: {result}")

        data = result.get("data", {})
        print(data)

        if data.get("state") == "failed":
            raise Exception(f"Task failed: {data.get('err_msg', '')}")

        full_zip_url = data.get("full_zip_url")
        if full_zip_url:
            break

        time.sleep(interval)

    if not full_zip_url:
        raise TimeoutError("Wait timeout, did not get full_zip_url")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    zip_path = save_dir / f"{task_id}.zip"
    extract_dir = save_dir / task_id

    with requests.get(full_zip_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    model_json = extract_dir / f"{task_id}_model.json"
    if model_json.exists():
        model_json_path = str(model_json)
    else:
        files = list(extract_dir.rglob("*_model.json"))
        if not files:
            raise FileNotFoundError("No *_model.json found after extraction")
        model_json_path = str(files[0])

    return {
        "task_id": task_id,
        "full_zip_url": full_zip_url,
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir),
        "model_json_path": model_json_path,
    }


# =============================
# 4. Domain content filtering and cleaning
# =============================
# Relevance rules are provided by DomainProfile.is_relevant; keep backward-compat alias.
from domap.domain_profile import is_route_content  # noqa: E402  (backward compat)


def merge_consecutive_text(items: List[dict]) -> List[dict]:
    merged = []
    buffer = []

    def flush_buffer():
        nonlocal buffer
        if buffer:
            merged.append({"type": "text", "content": "\n".join(buffer).strip()})
            buffer = []

    for it in items:
        if not isinstance(it, dict):
            continue

        t = it.get("type")
        c = (it.get("content") or "").strip()

        if t == "text":
            if c:
                buffer.append(c)
            else:
                flush_buffer()
            continue

        flush_buffer()
        merged.append(it)

    flush_buffer()
    return merged


def extract_route_text_and_tables_with_adjacent_title(
    json_file_path: str,
    profile: Optional[Any] = None,
) -> dict:
    """
    Extract domain-relevant text/table from model_json and return structured results.

    profile determines what content counts as relevant; default FAA airways.
    """
    prof = _resolve_profile(profile)
    results: Dict[str, str] = {}

    def make_unique_key(base: str) -> str:
        if base not in results:
            return base
        idx = 1
        while f"{base}_{idx}" in results:
            idx += 1
        return f"{base}_{idx}"

    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    outer = json_data if isinstance(json_data, list) else [json_data]

    for inner in outer:
        if isinstance(inner, dict):
            items = [inner]
        elif isinstance(inner, list):
            items = inner
        else:
            continue

        merged_items = merge_consecutive_text(items)

        for i, item in enumerate(merged_items):
            if not isinstance(item, dict):
                continue

            t = item.get("type")
            if t not in ("text", "table"):
                continue

            content = (item.get("content") or "").strip()
            if not content:
                continue

            if not prof.is_relevant(content):
                continue

            title_content = None
            if i - 1 >= 0:
                prev_item = merged_items[i - 1]
                if isinstance(prev_item, dict) and prev_item.get("type") == "title":
                    title_content = (prev_item.get("content") or "").strip() or None

            if title_content:
                base_title = title_content
            else:
                base_title = "none_text" if t == "text" else "none_table"

            key = make_unique_key(base_title)
            results[key] = content

    print(f"Successfully extracted {len(results)} {prof.record_word} entries")

    return {
        "json_file_path": json_file_path,
        "clean_data": results,
        "clean_count": len(results),
    }


def save_json(data: Any, output_path: str, result_key: str = "output_path") -> dict:
    """
    Save JSON and return structured result.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    saved_count = None
    if isinstance(data, (dict, list)):
        saved_count = len(data)

    return {
        result_key: str(output_path),
        "saved_count": saved_count,
    }


# =============================
# 5. DeepSeek call
# =============================
def call_deepseek(prompt: str, model_name=MODEL_NAME) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    resp = requests.post(DEEPSEEK_BASE_URL, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_json_array(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\[.*\]", text, re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def parse_partial_json_array(text):
    text = text.strip()

    start = text.find("[")
    if start == -1:
        return []

    sub = text[start:]
    decoder = json.JSONDecoder()
    idx = 1
    results = []

    while idx < len(sub):
        while idx < len(sub) and sub[idx] in " \n\r\t,":
            idx += 1
        if idx >= len(sub) or sub[idx] == "]":
            break

        try:
            obj, end = decoder.raw_decode(sub, idx)
            results.append(obj)
            idx = end
        except Exception:
            break

    return results


def split_html_table(html, max_len=MAX_SINGLE_ITEM):
    rows = html.split("</tr>")
    chunks, buf = [], "<table>"

    for r in rows:
        if not r.strip():
            continue
        candidate = buf + r + "</tr>"
        if len(candidate) > max_len:
            chunks.append(buf + "</table>")
            buf = "<table>" + r + "</tr>"
        else:
            buf = candidate

    buf += "</table>"
    chunks.append(buf)
    return chunks


def build_batches(data_dict):
    batches = []
    current_batch = {}
    current_len = 0

    for k, v in data_dict.items():
        item_text = json.dumps({k: v}, ensure_ascii=False)
        if current_len + len(item_text) > MAX_PROMPT_CHARS:
            if current_batch:
                batches.append(current_batch)
            current_batch = {k: v}
            current_len = len(item_text)
        else:
            current_batch[k] = v
            current_len += len(item_text)

    if current_batch:
        batches.append(current_batch)

    return batches


# =============================
# 6. Parse clean_json -> DeepSeek -> structured result
# =============================
def parse_clean_json_with_deepseek(
    clean_json_path: str,
    output_path: Optional[str] = None,
    profile: Optional[Any] = None,
) -> dict:
    prof = _resolve_profile(profile)

    with open(clean_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = {}
    for k, v in raw_data.items():
        if isinstance(v, str) and len(v) > MAX_SINGLE_ITEM:
            parts = split_html_table(v)
            for i, p in enumerate(parts):
                data[f"{k}_PART{i+1}"] = p
        else:
            data[k] = v

    batches = build_batches(data)
    all_results = []
    start_time = time.time()

    clean_json_path = Path(clean_json_path)
    if output_path is None:
        output_path = clean_json_path.with_name(prof.final_file_name)
    else:
        output_path = Path(output_path)

    fail_dir = output_path.parent / "failed_batches"
    fail_dir.mkdir(parents=True, exist_ok=True)

    for idx, batch in enumerate(batches):
        print(f"Processing batch {idx + 1}/{len(batches)} ...")
        prompt = prof.prompt_template.format(
            batch_text=json.dumps(batch, ensure_ascii=False)
        )

        success = False
        result_text = ""

        for attempt in range(MAX_RETRY + 1):
            try:
                result_text = call_deepseek(prompt)

                result_json = extract_json_array(result_text)

                if result_json is None:
                    fix_prompt = (
                        "The following output is not valid JSON. Please fix it into a valid JSON array, "
                        "no extra explanations:\n"
                        f"{result_text}"
                    )
                    result_text2 = call_deepseek(fix_prompt)
                    result_json = extract_json_array(result_text2)

                if result_json is None:
                    partial = parse_partial_json_array(result_text)
                    if partial:
                        print(f"Batch {idx + 1}: recovered {len(partial)} partial routes")
                        all_results.extend(partial)
                        success = True
                        break
                    else:
                        print(f"Batch {idx + 1}: parse failed (attempt {attempt + 1})")
                else:
                    all_results.extend(result_json)
                    success = True
                    break

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(0.1)

        if not success:
            fail_path = fail_dir / f"failed_batch_{idx + 1}.txt"
            with open(fail_path, "w", encoding="utf-8") as f:
                f.write(result_text)
            print(f"Batch {idx + 1} failed: JSON parse error -> {fail_path}")

    end_time = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("✅ Extraction finished.")
    print(f"✅ Total execution time: {end_time - start_time:.2f} seconds")
    print(f"✅ Total airway extracted: {len(all_results)}")
    print(f"✅ Saved to {output_path}")

    return {
        "clean_json_path": str(clean_json_path),
        "parsed_routes_path": str(output_path),
        "route_count": len(all_results),
        "failed_dir": str(fail_dir),
    }


# =============================
# 7. Top-level wrapper
# =============================
def parse_doc(
    file_url,
    save_dir=SAVE_DIR,
    model_version="vlm",
    clean_file_name=None,
    final_file_name=None,
    profile=None,
):
    """
    Full pipeline:
    1) MinerU parsing
    2) Download and extract, get model_json
    3) Clean into intermediate JSON according to profile
    4) DeepSeek extracts structured events according to profile
    5) Return complete state

    profile is the domain configuration (DomainProfile instance or name string), default FAA airways.

    file_url supports:
    - Online PDF/file URL
    - Local PDF absolute path
    - Virtual path (starting with /), auto-converted to host path
    - Local model.json (already parsed), skip MinerU step
    """
    from domap.connectivity.virtual_paths import to_host_path

    # Virtual path -> host path
    if isinstance(file_url, str) and file_url.startswith("/"):
        try:
            file_url = to_host_path(file_url)
        except ValueError:
            pass  # not under HOST_ROOT, keep original path

    prof = _resolve_profile(profile)
    clean_file_name = clean_file_name or prof.clean_file_name
    final_file_name = final_file_name or prof.final_file_name

    file_url_str = str(file_url)
    file_path = Path(file_url_str)

    # Guard: reject invalid placeholders (consistent with parse_doc_tool)
    _invalid = {"tbd", "to_be_confirmed", "unknown", "null", "none", "placeholder", ""}
    if (not file_url_str) or (file_url_str.strip().lower() in _invalid):
        raise ValueError(
            f"parse_doc received invalid file_url: '{file_url_str}'. "
            "Please provide a valid online PDF URL or local file path."
        )

    # Detect if it's an already-parsed model.json -> skip MinerU, go straight to extraction
    if file_path.suffix.lower() == ".json" and file_path.exists():
        model_json_path = file_path
        extract_dir = model_json_path.parent
        if not extract_dir or extract_dir == Path("."):
            extract_dir = Path(save_dir) / model_json_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"parse_doc: detected already-parsed model.json, skipping MinerU parsing")
        print(f"  model_json_path: {model_json_path}")
        print(f"  extract_dir: {extract_dir}")

        # Extract directly from model.json
        state = {}
        state = merge_state(
            state,
            **extract_route_text_and_tables_with_adjacent_title(
                str(model_json_path),
                profile=prof,
            ),
        )
        state = merge_state(
            state,
            **save_json(
                data=state["clean_data"],
                output_path=extract_dir / clean_file_name,
                result_key="clean_json_path",
            ),
        )
        state = merge_state(
            state,
            **parse_clean_json_with_deepseek(
                clean_json_path=state["clean_json_path"],
                output_path=extract_dir / final_file_name,
                profile=prof,
            ),
        )
        # Add task_id and path info (compatible with downstream code)
        state["task_id"] = model_json_path.stem
        state["model_json_path"] = str(model_json_path)
        state["extract_dir"] = str(extract_dir)

        print("task_id:", state["task_id"])
        print("model_json_path:", state["model_json_path"])
        print("clean_json_path:", state["clean_json_path"])
        print("parsed_routes_path:", state["parsed_routes_path"])
        print("route_count:", state["route_count"])
        return state

    # === Normal PDF pipeline ===
    state = {}

    # Local file (Windows path) upload first to get public URL
    if "://" not in file_url_str:
        file_url = upload_local_file(file_url_str)
        file_url_str = str(file_url)

    state = merge_state(state, **create_task(file_url_str, model_version=model_version))
    state = merge_state(
        state,
        **wait_and_download(
            task_id=state["task_id"],
            save_dir=save_dir,
        ),
    )
    state = merge_state(
        state,
        **extract_route_text_and_tables_with_adjacent_title(
            state["model_json_path"],
            profile=prof,
        ),
    )
    state = merge_state(
        state,
        **save_json(
            data=state["clean_data"],
            output_path=Path(state["extract_dir"]) / clean_file_name,
            result_key="clean_json_path",
        ),
    )
    state = merge_state(
        state,
        **parse_clean_json_with_deepseek(
            clean_json_path=state["clean_json_path"],
            output_path=Path(state["extract_dir"]) / final_file_name,
            profile=prof,
        ),
    )

    print("task_id:", state["task_id"])
    print("model_json_path:", state["model_json_path"])
    print("clean_json_path:", state["clean_json_path"])
    print("parsed_routes_path:", state["parsed_routes_path"])

    return state


@tool(parse_docstring=True)
def create_task_tool(
    file_url: str,
    model_version: str = "vlm",
    state: Optional[dict] = None,
) -> dict:
    """
    Create a MinerU document parsing task and write the result to shared state.

    Args:
        file_url: Online document URL.
        model_version: MinerU model version, typically "vlm".
        state: Shared state dict, can be empty.

    Returns:
        Updated shared state dict, containing at least:
        - file_url: input file URL
        - model_version: model version used
        - task_id: task ID returned by MinerU
    """
    result = create_task(file_url=file_url, model_version=model_version)
    return merge_state(state, **result)


@tool(parse_docstring=True)
def wait_and_download_tool(
    task_id: str = "",
    save_dir: str = SAVE_DIR,
    interval: int = 5,
    timeout: int = 600,
    state: Optional[dict] = None,
) -> dict:
    """
    Poll MinerU task, download and extract results upon completion, write to shared state.

    Args:
        task_id: MinerU task ID. If empty, try to read from state.
        save_dir: Result save directory.
        interval: Polling interval in seconds.
        timeout: Maximum wait time in seconds.
        state: Shared state dict, can be empty.

    Returns:
        Updated shared state dict with new fields including:
        - task_id: current task ID
        - full_zip_url: result zip download URL
        - zip_path: local zip path
        - extract_dir: extraction directory
        - model_json_path: extracted *_model.json file path
    """
    real_task_id = task_id or (state or {}).get("task_id")
    if not real_task_id:
        raise ValueError("wait_and_download_tool missing task_id, and state also has no task_id")

    result = wait_and_download(
        task_id=real_task_id,
        save_dir=save_dir,
        interval=interval,
        timeout=timeout,
    )
    return merge_state(state, **result)


@tool(parse_docstring=True)
def extract_and_save_route_tool(
    json_file_path: str = "",
    output_path: str = "",
    state: Optional[dict] = None,
) -> dict:
    """
    Extract airway-related text/table content from model_json and write to shared state.

    Args:
        json_file_path: Path to the *_model.json file output by MinerU. If empty, read from state.
        output_path: Output .json file path after data cleaning. If empty, read from state.
        state: Shared state dict, can be empty.

    Returns:
        Updated shared state dict with new fields including:
        - json_file_path: actual model_json path processed
        - clean_data: extracted cleaning result dict
        - clean_count: number of airway content entries extracted
    """
    real_json_file_path = json_file_path or (state or {}).get("model_json_path")
    if not real_json_file_path:
        raise ValueError("Missing json_file_path")

    if not output_path:
        base_dir = Path((state or {}).get("extract_dir", Path(real_json_file_path).parent))
        output_path = str(base_dir / "table_with_title-new3.json")

    result = extract_route_text_and_tables_with_adjacent_title(real_json_file_path)
    save_result = save_json(
        data=result["clean_data"],
        output_path=output_path,
        result_key="clean_json_path",
    )

    return merge_state(
        state,
        json_file_path=result["json_file_path"],
        clean_count=result["clean_count"],
        clean_json_path=save_result["clean_json_path"],
        saved_count=save_result["saved_count"],
    )



# @tool(parse_docstring=True)
# def save_json_tool(
#     data: Optional[dict] = None,
#     output_path: str = "",
#     result_key: str = "output_path",
#     state: Optional[dict] = None,
# ) -> dict:
#     """
#     Save data as JSON file and write path to shared state.
#
#     Args:
#         data: JSON data to save. If empty, try to read from state["clean_data"].
#         output_path: Output file path.
#         result_key: Path field name after saving, e.g. "clean_json_path".
#         state: Shared state dict, can be empty.
#
#     Returns:
#         Updated shared state dict with new fields including:
#         - path field specified by result_key
#         - saved_count: number of elements saved (if countable)
#     """
#     real_data = data if data is not None else (state or {}).get("clean_data")
#     if real_data is None:
#         raise ValueError("save_json_tool missing data, and state also has no clean_data")
#     if not output_path:
#         raise ValueError("save_json_tool missing output_path")
#
#     result = save_json(
#         data=real_data,
#         output_path=output_path,
#         result_key=result_key,
#     )
#     return merge_state(state, **result)


@tool(parse_docstring=True)
def parse_clean_json_tool(
    clean_json_path: str = "",
    output_path: Optional[str] = None,
    state: Optional[dict] = None,
    profile: str = "faa_airway",
) -> dict:
    """
    Send cleaned JSON to DeepSeek for structured extraction and write results to shared state.

    Args:
        clean_json_path: Cleaned JSON file path. If empty, read from state.
        output_path: Parsed result output path.
        state: Shared state dict, can be empty.
        profile: Domain profile name, "faa_airway" (FAA airways) or "uscg_lnm" (USCG LNM).

    Returns:
        Updated shared state dict with new fields including:
        - clean_json_path: input cleaned file path
        - parsed_routes_path: DeepSeek parse result path
        - route_count: number of airways extracted
        - failed_dir: save directory for failed batches
    """
    real_clean_json_path = clean_json_path or (state or {}).get("clean_json_path")
    if not real_clean_json_path:
        raise ValueError("parse_clean_json_tool missing clean_json_path, and state also has no clean_json_path")

    result = parse_clean_json_with_deepseek(
        clean_json_path=real_clean_json_path,
        output_path=output_path,
        profile=profile,
    )
    return merge_state(state, **result)

@tool(parse_docstring=True)
def parse_doc_tool(
    file_url: str,
    save_dir: str = SAVE_DIR,
    model_version: str = "vlm",
    profile: str = "faa_airway",
) -> dict:
    """
    Parse a long PDF document, extract structured events, output final result file path and statistics.

    Args:
        file_url: Online PDF URL or local/virtual path
        save_dir: Output directory
        model_version: MinerU model version
        profile: Domain profile name, "faa_airway" (FAA airways) or "uscg_lnm" (USCG LNM)

    Returns:
        Simplified result dict containing only key paths and counts
    """
    # Guard: if LLM passes invalid placeholder, raise an error immediately
    invalid_placeholders = {"tbd", "to_be_confirmed", "unknown", "null", "none", "placeholder", ""}
    if (not file_url) or (file_url.strip().lower() in invalid_placeholders):
        raise ValueError(
            f"parse_doc_tool received invalid file_url: '{file_url}'. "
            "Please get the source document path from the supervisor task's [Source Document to Parse] "
            "and pass it to the file_url parameter exactly as-is. "
            "Do NOT fabricate or use placeholders."
        )
    state = parse_doc(
        file_url=file_url,
        save_dir=save_dir,
        model_version=model_version,
        profile=profile,
    )

    return {
        "task_id": state["task_id"],
        "model_json_path": state["model_json_path"],
        "clean_json_path": state["clean_json_path"],
        "parsed_routes_path": state["parsed_routes_path"],
        "route_count": state["route_count"],
        "failed_dir": state["failed_dir"],
    }

# =============================
# 9. Usage example
# =============================
if __name__ == "__main__":
    state = parse_doc(
        "https://nfdc.faa.gov/webContent/Part95/Part_95_Consolidation_February_2025.pdf/Part_95_Consolidation_February_2025.pdf"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))