# -*- coding: utf-8 -*-
"""
Strong Baseline 2: Retrieval-Augmented Extraction (RAG)
==================================================================
Represents the "RAG-style extraction" baseline: instead of structuring the entire
document, it splits into chunks + keyword retrieval to find chunks most relevant
to "airway tables", then feeds them to the LLM for single-pass extraction.

Fairness: consumes the same table_with_title-new1.json, produces the same schema.
Differences from the multi-agent approach are confined to the "retrieval" stage:
RAG only feeds the top-k relevant chunks to the LLM.

★ Safety: API key is read from environment variable only, never hardcoded. Before running:
    set ANTHROPIC_API_KEY=your_key
Usage:
    python baseline_rag.py                # default top_k=40
    python baseline_rag.py --top-k 60
"""
from __future__ import annotations

import os
import re
import json
import time
import math
import argparse
from collections import Counter
from typing import Any, Dict, List

import requests

from baselines_common import (
    load_clean_input, code_from_title, save_pred,
)

CLAUDE_URL = "https://api.anthropic.com/v1/messages"
MODEL_NAME = "claude-opus-4-5-20250929"

# Retrieval query: typical vocabulary of airway tables
RAG_QUERY = "airway federal route VOR NDB FIX waypoint MEA MAA segment altitude"

# Single extraction prompt (aligned with main experiment schema)
EXTRACT_PROMPT = r"""Task: Extract structured JSON from the following single airway HTML table/text. Output only one JSON object, no explanation.
Airway code: {code}
Content:
{content}

Strict output format:
{{"airway_code":"{code}","MEA":"number or none","MAA":"number or none",
"airway_point":[{{"name":"point name","region":"two-letter region code or none","type":"VOR/NDB/FIX/WP/etc","position":["none","none"]}}]}}

Rules: MEA=3rd column/first altitude, MAA=4th column; fill "none" for missing; keep waypoints in order, remove consecutive duplicates only."""


def _require_key():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY not found. Set ANTHROPIC_API_KEY=your_key first")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]{2,}", (text or "").upper())


def build_retriever(data: Dict[str, str]):
    """Build a minimal TF-IDF retriever (pure Python, simulating RAG retrieval stage)."""
    docs = list(data.items())
    df = Counter()
    doc_tokens = []
    for _title, content in docs:
        toks = set(_tokenize(content))
        doc_tokens.append(toks)
        for t in toks:
            df[t] += 1
    n = len(docs)
    idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
    return docs, doc_tokens, idf


def retrieve_top_k(query: str, docs, doc_tokens, idf, top_k: int) -> List[int]:
    q = set(_tokenize(query))
    scores = []
    for i, toks in enumerate(doc_tokens):
        s = sum(idf.get(t, 0.0) for t in (q & toks))
        scores.append((s, i))
    scores.sort(reverse=True)
    return [i for _s, i in scores[:top_k]]


def call_llm(prompt: str) -> str:
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
               "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    body = {"model": MODEL_NAME, "temperature": 0, "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(CLAUDE_URL, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def _extract_json_obj(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def run_rag(top_k: int = 40) -> str:
    _require_key()
    t0 = time.time()
    data = load_clean_input()
    docs, doc_tokens, idf = build_retriever(data)
    hit_idx = set(retrieve_top_k(RAG_QUERY, docs, doc_tokens, idf, top_k))

    records: List[Dict[str, Any]] = []
    for i, (title, content) in enumerate(docs):
        if i not in hit_idx:
            continue  # RAG only processes retrieved chunks (intentionally preserving missed-detection failure mode)
        code, _cont = code_from_title(title)
        prompt = EXTRACT_PROMPT.format(code=code, content=content[:8000])
        try:
            obj = _extract_json_obj(call_llm(prompt))
            if obj and obj.get("airway_point"):
                records.append(obj)
        except Exception as e:
            print(f"  [rag] {code} extraction failed: {e}")
    path = save_pred(records, "rag")
    print(f"[rag] top_k={top_k} hit {len(hit_idx)}/{len(docs)} chunks, "
          f"extracted {len(records)} records, elapsed {time.time()-t0:.1f}s")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()
    run_rag(top_k=args.top_k)