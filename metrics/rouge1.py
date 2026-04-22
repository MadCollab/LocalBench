"""ROUGE-1 F1 score.

Uses the `rouge_score` package when available; otherwise falls back to a
small pure-Python implementation so the metric still works in minimal
environments.
"""

from __future__ import annotations

import re
from collections import Counter

from .exact_match import normalise


_WORD = re.compile(r"\w+")


def _tokens(text: str):
    return _WORD.findall(normalise(text))


def _pure_python_rouge1_f1(pred: str, ref: str) -> float:
    pt, rt = _tokens(pred), _tokens(ref)
    if not pt or not rt:
        return 0.0
    common = Counter(pt) & Counter(rt)
    match = sum(common.values())
    if match == 0:
        return 0.0
    p = match / len(pt)
    r = match / len(rt)
    return 2 * p * r / (p + r)


try:
    from rouge_score import rouge_scorer  # type: ignore

    _scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)

    def rouge1_f1(prediction: str, reference: str) -> float:
        if not prediction or not reference:
            return 0.0
        return float(_scorer.score(reference, prediction)["rouge1"].fmeasure)

except ImportError:  # pragma: no cover
    def rouge1_f1(prediction: str, reference: str) -> float:
        return _pure_python_rouge1_f1(prediction, reference)
