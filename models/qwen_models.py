"""Qwen3 models.

Two modes are supported:

1. OpenAI-compatible HTTP endpoint (recommended for the large MoE models).
   Set QWEN_API_BASE and QWEN_API_KEY in the environment. Any vLLM,
   Together, DashScope or Ollama endpoint that speaks the OpenAI
   chat-completions schema will work.

2. Local Hugging Face transformers (for the smaller dense models such as
   Qwen3-8B / 14B / 32B). Requires `transformers` and `torch`.
"""

from __future__ import annotations

import os
from typing import List, Dict

import backoff

from .base import BaseModel


class _HTTPClient:
    def __init__(self):
        from openai import OpenAI  # OpenAI-compatible endpoints
        base = os.environ["QWEN_API_BASE"]
        key = os.environ.get("QWEN_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=base, api_key=key)


class QwenModel(BaseModel):
    def __init__(self, model: str, public_id: str):
        super().__init__(model, public_id)
        self._backend = None
        self._hf_model = None
        self._hf_tokenizer = None

    def _init_backend(self):
        if self._backend is not None:
            return
        if os.environ.get("QWEN_API_BASE"):
            self._backend = "http"
            self._http = _HTTPClient()
        else:
            self._backend = "hf"
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch  # noqa: F401
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Local Qwen inference needs `transformers` and `torch`. "
                    "Alternatively, set QWEN_API_BASE to an OpenAI-compatible endpoint."
                ) from e
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model, torch_dtype="auto", device_map="auto"
            )

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        self._init_backend()
        if self._backend == "http":
            resp = self._http.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        # Local Hugging Face path
        import torch
        text = self._hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._hf_tokenizer([text], return_tensors="pt").to(self._hf_model.device)
        with torch.no_grad():
            output_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-5),
                do_sample=temperature > 0,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._hf_tokenizer.decode(generated, skip_special_tokens=True).strip()
