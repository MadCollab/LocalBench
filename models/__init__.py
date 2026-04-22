"""Model provider dispatch for LocalBench.

Each supported model has an `id` (used in config.json) and is implemented
as a callable class with the signature

    model = MyModel(model_id)
    answer = model.generate(messages, temperature=0.0, max_tokens=256)

where `messages` is an OpenAI-style list of {"role", "content"} dicts and
the return value is the raw text produced by the model.

`get_model` resolves an id string like "gpt-4o" or "qwen3-8b" to the
correct backend class.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseModel
from .openai_models import OpenAIModel, OpenAIWebModel
from .anthropic_models import AnthropicModel
from .gemini_models import GeminiModel, GeminiGroundingModel

try:
    from .qwen_models import QwenModel  # optional (vLLM/HF)
except ImportError:  # pragma: no cover - only needed if Qwen is used
    QwenModel = None  # type: ignore


# id -> (BackendClass, backend_model_id)
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # OpenAI
    "gpt-4o":          {"cls": OpenAIModel,    "model": "gpt-4o-2024-08-06"},
    "gpt-4.1":         {"cls": OpenAIModel,    "model": "gpt-4.1-2025-04-14"},
    "gpt-4.1-web":     {"cls": OpenAIWebModel, "model": "gpt-4.1-2025-04-14"},
    # Anthropic
    "claude-sonnet-4":   {"cls": AnthropicModel, "model": "claude-sonnet-4-20250514"},
    "claude-sonnet-3.7": {"cls": AnthropicModel, "model": "claude-3-7-sonnet-20250219"},
    # Google Gemini
    "gemini-2.5-pro":            {"cls": GeminiModel,          "model": "gemini-2.5-pro"},
    "gemini-2.5-flash":          {"cls": GeminiModel,          "model": "gemini-2.5-flash"},
    "gemini-2.5-pro-grounding":  {"cls": GeminiGroundingModel, "model": "gemini-2.5-pro"},
    # Qwen family (via OpenAI-compatible endpoint or local HF)
    "qwen3-235b-a22b": {"cls": QwenModel, "model": "Qwen/Qwen3-235B-A22B"},
    "qwen3-30b-a3b":   {"cls": QwenModel, "model": "Qwen/Qwen3-30B-A3B"},
    "qwen3-32b":       {"cls": QwenModel, "model": "Qwen/Qwen3-32B"},
    "qwen3-14b":       {"cls": QwenModel, "model": "Qwen/Qwen3-14B"},
    "qwen3-8b":        {"cls": QwenModel, "model": "Qwen/Qwen3-8B"},
}


def available_models() -> list:
    return sorted(_MODEL_REGISTRY.keys())


def get_model(model_id: str) -> BaseModel:
    if model_id not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_id}'. Available: {available_models()}"
        )
    entry = _MODEL_REGISTRY[model_id]
    cls = entry["cls"]
    if cls is None:
        raise ImportError(
            f"Backend for '{model_id}' is not installed. "
            f"For Qwen models, install `transformers` and `torch`, "
            f"or set QWEN_API_BASE / QWEN_API_KEY for a hosted endpoint."
        )
    return cls(entry["model"], public_id=model_id)


__all__ = ["get_model", "available_models", "BaseModel"]
