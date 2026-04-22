"""Query a given model on LocalBench and save raw predictions.

Produces one JSONL file per model at:

    <output_dir>/predictions/<model_id>.jsonl

Each line contains the QA metadata, prompt, and model response. Runs
support resumption: rows whose id is already present in the JSONL file
are skipped.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List

from tqdm.auto import tqdm

from loader import QAItem
from prompt import build_prompt, as_messages


def _prediction_path(output_dir: str, model_id: str) -> str:
    return os.path.join(output_dir, "predictions", f"{model_id}.jsonl")


def _load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def _run_one(model, item: QAItem, temperature: float, max_tokens: int, include_context: bool) -> dict:
    prompt = build_prompt(item, include_context=include_context)
    messages = as_messages(prompt)
    t0 = time.time()
    try:
        response = model.generate(messages, temperature=temperature, max_tokens=max_tokens)
        error = None
    except Exception as e:  # pragma: no cover - logged per-row
        response = ""
        error = f"{type(e).__name__}: {e}"
    return {
        "id": item.id,
        "dataset": item.dataset,
        "county": item.county,
        "state": item.state,
        "rucc_group": item.rucc_group,
        "dimension": item.dimension,
        "question_type": item.question_type,
        "question": item.question,
        "reference": item.answer,
        "context": item.context,
        "prompt_system": prompt["system"],
        "prompt_user": prompt["user"],
        "response": response,
        "latency_s": round(time.time() - t0, 3),
        "error": error,
        "model": model.public_id,
    }


def run_model(
    model,
    items: Iterable[QAItem],
    output_dir: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    include_context: bool = False,
    concurrency: int = 4,
    rate_limit_seconds: float = 0.0,
    resume: bool = True,
) -> str:
    """Run `model` on `items` and append predictions to a JSONL file."""
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    out_path = _prediction_path(output_dir, model.public_id)
    done = _load_done_ids(out_path) if resume else set()
    items = [it for it in items if it.id not in done]
    if not items:
        print(f"[{model.public_id}] nothing to do ({len(done)} already done).")
        return out_path

    # Serial path when concurrency <= 1
    mode = "a" if os.path.exists(out_path) else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        if concurrency <= 1:
            for it in tqdm(items, desc=f"{model.public_id}"):
                rec = _run_one(model, it, temperature, max_tokens, include_context)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if rate_limit_seconds:
                    time.sleep(rate_limit_seconds)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(_run_one, model, it, temperature, max_tokens, include_context): it
                    for it in items
                }
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{model.public_id}"):
                    rec = fut.result()
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
    print(f"[{model.public_id}] wrote {len(items)} predictions to {out_path}")
    return out_path
