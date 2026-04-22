"""Base class for LocalBench model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseModel(ABC):
    """Abstract backend. Implementations must produce plain-text answers."""

    def __init__(self, model: str, public_id: str):
        self.model = model        # provider-specific identifier
        self.public_id = public_id  # the id used in config.json

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Return the raw text produced by the model."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.public_id})"
