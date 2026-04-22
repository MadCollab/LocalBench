"""Semantic similarity via OpenAI embeddings.

Following the paper we use `text-embedding-3-small` by default, but any
OpenAI embedding model can be swapped in. A lightweight in-memory cache
avoids recomputing embeddings for repeated reference answers.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List


class SemanticMatcher:
    """Cosine similarity between embeddings of prediction and reference."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._client = None
        self._cache: Dict[str, List[float]] = {}

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError("Install `openai` to use SemanticMatcher.") from e
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is required for SemanticMatcher.")
            self._client = OpenAI(api_key=key)
        return self._client

    def _embed(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []
        if text in self._cache:
            return self._cache[text]
        resp = self.client.embeddings.create(model=self.model, input=text)
        emb = resp.data[0].embedding
        self._cache[text] = emb
        return emb

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def score(self, prediction: str, reference: str) -> float:
        return max(0.0, self._cosine(self._embed(prediction), self._embed(reference)))
