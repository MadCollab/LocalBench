"""Reddit QA generation (Step #1-3 of the pipeline).

Input: a parquet with one row per Reddit thread and the columns

    title, selftext, comments, county, state, created_time,
    rucc, rucc_group, fips

Output: a parquet with one QA pair per (accepted) thread, in the same
schema as `data/reddit_QA.parquet`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, Optional

import backoff
import pandas as pd
from tqdm.auto import tqdm

from dotenv import load_dotenv

from quality_analyzer import QualityAnalyzer


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "reddit_qa_prompt.txt")


def load_prompt_template() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(row: pd.Series, template: str) -> str:
    return template.format(
        county=row["county"],
        state=row["state"],
        created_time=row.get("created_time", ""),
        title=row.get("title", ""),
        selftext=row.get("selftext", ""),
        comments=row.get("comments", ""),
    )


def parse_pair(raw: str) -> Optional[Dict[str, str]]:
    pair: Dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        for key, field in [
            ("Question:", "question"),
            ("Context:", "context"),
            ("Answer:", "answer"),
            ("Selected Comments:", "selected_comments"),
            ("Pair_type:", "pair_type"),
        ]:
            if line.startswith(key):
                pair[field] = line[len(key):].strip()
    return pair if {"question", "answer", "context"} <= pair.keys() else None


class RedditQAGenerator:
    def __init__(self, model: str = "o3", temperature: float = 0.7, max_tokens: int = 200):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required.")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(self, prompt: str, refinement: str = "") -> str:
        messages = [
            {"role": "system", "content":
                "You generate high-quality, locally grounded question-context-answer pairs."},
            {"role": "user", "content": prompt},
        ]
        if refinement:
            messages.append({"role": "user", "content": refinement})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


def build_refinement(failed: list, county: str, state: str) -> str:
    if not failed:
        return ""
    lines = [
        "The previous QA pair failed quality assessment on these requirements:",
        *[f"- {name}" for name in failed],
        f"Please generate a new QA pair for {county}, {state} that fixes every failure.",
    ]
    return "\n".join(lines)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate Reddit QA pairs.")
    parser.add_argument("--input", required=True, help="Raw reddit threads parquet")
    parser.add_argument("--output", required=True, help="Destination QA parquet")
    parser.add_argument("--generator", default="o3")
    parser.add_argument("--filter", default="gpt-4o-mini")
    parser.add_argument("--max-regenerations", type=int, default=3)
    args = parser.parse_args()

    template = load_prompt_template()
    generator = RedditQAGenerator(model=args.generator)
    analyser = QualityAnalyzer(model=args.filter)

    df = pd.read_parquet(args.input)
    accepted = []
    checkpoint_path = args.output + ".checkpoint.json"
    done_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            done_ids = set(json.load(f).get("processed", []))

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="reddit"):
        key = str(row.get("fips", "")) + "_" + str(idx)
        if key in done_ids:
            continue
        prompt = render_prompt(row, template)
        refinement = ""
        pair = None
        for attempt in range(args.max_regenerations + 1):
            raw = generator.generate(prompt, refinement)
            pair = parse_pair(raw)
            if pair is None:
                refinement = "Use the exact OUTPUT FORMAT with [PAIR1], Question:, Context:, Answer: labels."
                continue
            result = analyser.evaluate(
                question=pair["question"], context=pair["context"], answer=pair["answer"],
                county=row["county"], state=row["state"],
            )
            if result.passed:
                break
            refinement = build_refinement(result.failed, row["county"], row["state"])
        if pair and result.passed:
            accepted.append({
                "state": row["state"],
                "county": row["county"],
                "created_time": row.get("created_time"),
                "rucc": row.get("rucc"),
                "rucc_group": row.get("rucc_group"),
                "fips": row.get("fips"),
                "question": pair["question"],
                "context": pair["context"],
                "answer": pair["answer"],
                "chosen_dimension": None,  # filled by Step #4 classifier
            })
        done_ids.add(key)
        if idx % 20 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"processed": list(done_ids)}, f)

    out = pd.DataFrame(accepted)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out)} QA pairs to {args.output}")


if __name__ == "__main__":
    main()
