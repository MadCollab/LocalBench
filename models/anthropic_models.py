"""Anthropic Claude models."""

from __future__ import annotations

import os
from typing import List, Dict

import backoff

from .base import BaseModel


def _get_client():
    try:
        from anthropic import Anthropic
    except ImportError as e:  # pragma: no cover
        raise ImportError("anthropic package is required. Run `pip install anthropic`.") from e
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    return Anthropic(api_key=key)


class AnthropicModel(BaseModel):
    def __init__(self, model: str, public_id: str):
        super().__init__(model, public_id)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _get_client()
        return self._client

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] != "system"
        ]
        resp = self.client.messages.create(
            model=self.model,
            system=system or None,
            messages=user_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Concatenate any text blocks
        parts = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
