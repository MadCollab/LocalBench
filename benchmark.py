"""End-to-end LocalBench runner.

Reads `config.json` (or the path passed with --config), queries each of
the configured models on the configured datasets, and evaluates the
resulting predictions.

Usage:

    python benchmark.py                  # full pipeline (query + evaluate)
    python benchmark.py --skip-query     # re-score existing predictions only
    python benchmark.py --skip-evaluate  # run queries but do not score
    python benchmark.py --models gpt-4o  # override the list of models
    python benchmark.py --sample 50      # small-scale smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from loader import load_all
from models import available_models, get_model
from query import run_model
from evaluate import evaluate_all


def parse_args():
    p = argparse.ArgumentParser(description="Run LocalBench end-to-end.")
    p.add_argument("--config", default="config.json")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Override model list. Available: {available_models()}")
    p.add_argument("--datasets", nargs="+", default=None,
                   choices=["census", "reddit", "news"])
    p.add_argument("--sample", type=int, default=None,
                   help="Sample N items per dataset (useful for smoke tests)")
    p.add_argument("--skip-query", action="store_true")
    p.add_argument("--skip-evaluate", action="store_true")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    data_dir = args.data_dir or cfg.get("data_dir", "data")
    output_dir = args.output_dir or cfg.get("output_dir", "outputs")
    datasets = args.datasets or cfg.get("datasets", ["census", "reddit", "news"])
    model_ids = args.models or cfg.get("models", [])
    sample = args.sample if args.sample is not None else cfg.get("sample_size_per_dataset")

    qcfg = cfg.get("query", {})

    print(f"LocalBench: datasets={datasets}  models={model_ids}  sample={sample}")
    items = load_all(
        data_dir=data_dir,
        datasets=datasets,
        sample_size_per_dataset=sample,
        random_seed=cfg.get("random_seed", 42),
    )
    print(f"Loaded {len(items)} QA items")

    if not args.skip_query:
        if not model_ids:
            print("No models configured. Set `models` in config.json or pass --models.",
                  file=sys.stderr)
            sys.exit(1)
        for mid in model_ids:
            try:
                model = get_model(mid)
            except Exception as e:
                print(f"[skip] {mid}: {e}", file=sys.stderr)
                continue
            run_model(
                model,
                items,
                output_dir=output_dir,
                temperature=qcfg.get("temperature", 0.0),
                max_tokens=qcfg.get("max_tokens", 256),
                include_context=qcfg.get("include_context", False),
                concurrency=qcfg.get("concurrency", 4),
                rate_limit_seconds=qcfg.get("rate_limit_seconds", 0.0),
                resume=qcfg.get("resume", True),
            )

    if not args.skip_evaluate:
        cfg_for_eval = dict(cfg)
        evaluate_all(output_dir=output_dir, config=cfg_for_eval)


if __name__ == "__main__":
    main()
