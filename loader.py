"""Dataset loading for LocalBench.

This module loads the three LocalBench QA sources (Census, Reddit, News)
and normalises them into a single schema used throughout the benchmark:

    id              : unique string identifier
    dataset         : one of {"census", "reddit", "news"}
    county          : county name (title-cased)
    state           : state name (title-cased)
    fips            : 5-digit FIPS code as string (may be None)
    rucc            : raw RUCC code (1-9) if available
    rucc_group      : {"Urban", "Suburban", "Rural"}
    dimension       : localness dimension label
    question_type   : {"numerical", "non_numerical"}
    question        : the question string presented to the model
    context         : optional context passage (may be empty)
    answer          : ground-truth answer as a string

See the paper (Benchmark Construction) for details on each source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd


@dataclass
class QAItem:
    """A single normalised QA record."""
    id: str
    dataset: str
    county: str
    state: str
    fips: Optional[str]
    rucc: Optional[int]
    rucc_group: Optional[str]
    dimension: Optional[str]
    question_type: str
    question: str
    context: str
    answer: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "county": self.county,
            "state": self.state,
            "fips": self.fips,
            "rucc": self.rucc,
            "rucc_group": self.rucc_group,
            "dimension": self.dimension,
            "question_type": self.question_type,
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
        }


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------

def _normalise_name(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip().title()


def _normalise_fips(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return str(int(float(x))).zfill(5)
    except (TypeError, ValueError):
        return str(x)


def load_census(path: str) -> List[QAItem]:
    df = pd.read_csv(path)
    items: List[QAItem] = []
    for i, row in df.iterrows():
        q_type = "numerical" if str(row.get("question_type", "")).lower() == "numerical" else "non_numerical"
        items.append(
            QAItem(
                id=f"census_{i}",
                dataset="census",
                county=_normalise_name(row.get("COUNTY_NAME")),
                state=_normalise_name(row.get("STATE_NAME")),
                fips=_normalise_fips(row.get("fips")),
                rucc=int(row["RUCC"]) if pd.notna(row.get("RUCC")) else None,
                rucc_group=row.get("RUCC_group"),
                dimension=row.get("Dimension"),
                question_type=q_type,
                question=str(row.get("question", "")),
                context="",
                answer=str(row.get("answer", "")),
            )
        )
    return items


def load_reddit(path: str) -> List[QAItem]:
    df = pd.read_parquet(path)
    items: List[QAItem] = []
    for i, row in df.iterrows():
        items.append(
            QAItem(
                id=f"reddit_{i}",
                dataset="reddit",
                county=_normalise_name(row.get("county")),
                state=_normalise_name(row.get("state")),
                fips=_normalise_fips(row.get("fips")),
                rucc=int(row["rucc"]) if pd.notna(row.get("rucc")) else None,
                rucc_group=row.get("rucc_group"),
                dimension=row.get("chosen_dimension"),
                question_type="non_numerical",
                question=str(row.get("question", "")),
                context=str(row.get("context", "") or ""),
                answer=str(row.get("answer", "")),
            )
        )
    return items


def load_news(path: str) -> List[QAItem]:
    df = pd.read_parquet(path)
    items: List[QAItem] = []
    for i, row in df.iterrows():
        items.append(
            QAItem(
                id=f"news_{i}",
                dataset="news",
                county=_normalise_name(row.get("county")),
                state=_normalise_name(row.get("state")),
                fips=_normalise_fips(row.get("fips")),
                rucc=int(row["RUCC"]) if pd.notna(row.get("RUCC")) else None,
                rucc_group=row.get("rucc_group"),
                dimension=row.get("chosen_dimension"),
                question_type="non_numerical",
                question=str(row.get("question", "")),
                context=str(row.get("context", "") or ""),
                answer=str(row.get("answer", "")),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADERS = {
    "census": ("census_QA.csv", load_census),
    "reddit": ("reddit_QA.parquet", load_reddit),
    "news":   ("news_QA.parquet", load_news),
}


def load_dataset(name: str, data_dir: str = "data") -> List[QAItem]:
    """Load a single LocalBench sub-dataset by name."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {list(_LOADERS)}")
    filename, fn = _LOADERS[name]
    return fn(os.path.join(data_dir, filename))


def load_all(
    data_dir: str = "data",
    datasets: Optional[Iterable[str]] = None,
    sample_size_per_dataset: Optional[int] = None,
    random_seed: int = 42,
) -> List[QAItem]:
    """Load and concatenate the requested sub-datasets."""
    datasets = list(datasets) if datasets else list(_LOADERS)
    all_items: List[QAItem] = []
    for name in datasets:
        items = load_dataset(name, data_dir)
        if sample_size_per_dataset is not None and sample_size_per_dataset < len(items):
            import random
            rng = random.Random(random_seed)
            items = rng.sample(items, sample_size_per_dataset)
        all_items.extend(items)
    return all_items


def as_dataframe(items: Iterable[QAItem]) -> pd.DataFrame:
    return pd.DataFrame([it.to_dict() for it in items])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect LocalBench datasets.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default=None, choices=list(_LOADERS) + [None])
    args = parser.parse_args()

    if args.dataset:
        items = load_dataset(args.dataset, args.data_dir)
    else:
        items = load_all(args.data_dir)

    df = as_dataframe(items)
    print(f"Loaded {len(df)} QA pairs")
    print(df.groupby(["dataset", "question_type"]).size())
