"""News QA generation.

Input: a parquet with one row per article and the columns

    title, factual_content, date, county, state, source,
    rucc, rucc_group, fips

`factual_content` must be a JSON-serialised list of strings (one per
factual sentence).

Output: same schema as `data/news_QA.parquet`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import backoff
import pandas as pd
from tqdm.auto import tqdm

from dotenv import load_dotenv

from quality_analyzer import QualityAnalyzer


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "news_qa_prompt.txt")


def load_prompt_template() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(row: pd.Series, template: str) -> str:
    sents = row["factual_content"]
    if isinstance(sents, str):
        try:
            sents = json.loads(sents)
        except Exception:
            sents = [sents]
    joined = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(sents))
    return template.format(
        title=row.get("title", ""),
        date=row.get("date", ""),
        county=row.get("county", ""),
        state=row.get("state", ""),
        source=row.get("source", ""),
        factual_sentences=joined,
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
            ("Selected Sentences:", "selected_sentences"),
            ("Pair_type:", "pair_type"),
        ]:
            if line.startswith(key):
                pair[field] = line[len(key):].strip()
    return pair if {"question", "answer", "context"} <= pair.keys() else None


class NewsQAGenerator:
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
                "You generate high-quality question-context-answer pairs from local news articles."},
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
    return "\n".join([
        "The previous QA pair failed quality assessment on these requirements:",
        *[f"- {name}" for name in failed],
        f"Please regenerate for {county}, {state} so that every failure is fixed.",
    ])


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate news QA pairs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generator", default="o3")
    parser.add_argument("--filter", default="gpt-4o-mini")
    parser.add_argument("--max-regenerations", type=int, default=3)
    args = parser.parse_args()

    template = load_prompt_template()
    generator = NewsQAGenerator(model=args.generator)
    analyser = QualityAnalyzer(model=args.filter)

    df = pd.read_parquet(args.input)
    accepted = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="news"):
        prompt = render_prompt(row, template)
        refinement = ""
        pair, result = None, None
        for _ in range(args.max_regenerations + 1):
            raw = generator.generate(prompt, refinement)
            pair = parse_pair(raw)
            if pair is None:
                refinement = "Use the exact OUTPUT FORMAT with [PAIR], Question:, Context:, Answer: labels."
                continue
            result = analyser.evaluate(
                question=pair["question"], context=pair["context"], answer=pair["answer"],
                county=row["county"], state=row["state"],
            )
            if result.passed:
                break
            refinement = build_refinement(result.failed, row["county"], row["state"])
        if pair and result and result.passed:
            accepted.append({
                "state": row["state"],
                "county": row["county"],
                "date": row.get("date"),
                "RUCC": row.get("rucc"),
                "rucc_group": row.get("rucc_group"),
                "fips": row.get("fips"),
                "question": pair["question"],
                "context": pair["context"],
                "answer": pair["answer"],
                "chosen_dimension": None,  # filled by Step #4 classifier
            })

    out = pd.DataFrame(accepted)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out)} QA pairs to {args.output}")


if __name__ == "__main__":
    main()
