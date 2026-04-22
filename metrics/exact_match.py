"""Exact-match metric with light normalisation.

Following the paper, we apply Unicode/whitespace/punctuation normalisation
and case-folding before comparing the generated answer to the reference.
"""

from __future__ import annotations

import re
import string
import unicodedata


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lower-cased, punctuation- and article-stripped comparison form."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalise(prediction) == normalise(reference) and normalise(prediction) else 0.0
