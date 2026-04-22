"""Google Gemini models (vanilla and with Search Grounding)."""

from __future__ import annotations

import os
from typing import List, Dict

import backoff

from .base import BaseModel


def _get_client():
    try:
        from google import genai
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "google-genai package is required. Run `pip install google-genai`."
        ) from e
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return genai.Client(api_key=key)


def _messages_to_contents(messages: List[Dict[str, str]]):
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    prompt = f"{system}\n\n{user}".strip() if system else user
    return prompt


class GeminiModel(BaseModel):
    use_grounding = False

    def __init__(self, model: str, public_id: str):
        super().__init__(model, public_id)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _get_client()
        return self._client

    def _config(self, temperature: float, max_tokens: int):
        from google.genai import types
        cfg = dict(temperature=temperature, max_output_tokens=max_tokens)
        if self.use_grounding:
            cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        return types.GenerateContentConfig(**cfg)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        contents = _messages_to_contents(messages)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._config(temperature, max_tokens),
        )
        return (getattr(resp, "text", "") or "").strip()


class GeminiGroundingModel(GeminiModel):
    """Gemini + Google Search grounding (the paper's `+Grounding` variant)."""
    use_grounding = True
