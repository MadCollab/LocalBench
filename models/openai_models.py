"""OpenAI-backed models (GPT-4o, GPT-4.1) and a Web-search augmented variant."""

from __future__ import annotations

import os
from typing import List, Dict

import backoff

from .base import BaseModel


def _get_client():
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError("openai package is required. Run `pip install openai`.") from e
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)


class OpenAIModel(BaseModel):
    """Standard chat-completions call against an OpenAI model."""

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
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class OpenAIWebModel(OpenAIModel):
    """GPT-4.1 with the Responses API + web_search_preview tool."""

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        # Flatten system+user for the Responses API
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        resp = self.client.responses.create(
            model=self.model,
            instructions=system or None,
            input=user,
            tools=[{"type": "web_search_preview"}],
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        # The SDK exposes a convenience `output_text` property
        return (getattr(resp, "output_text", "") or "").strip()
