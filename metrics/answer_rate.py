"""Answer rate (willingness to respond).

A response is counted as a refusal when it is empty or matches common
"I don't know" patterns. `answer_rate` returns the fraction of
substantive responses over a list of predictions.
"""

from __future__ import annotations

import re
from typing import Iterable

_REFUSAL_PATTERNS = [
    r"^i\s*(don'?t|do not)\s*know\b",
    r"^i\s*(cannot|can'?t)\s*(answer|help|provide|determine|find)\b",
    r"^i\s*(don'?t|do not)\s*have\s*(enough|sufficient|access)",
    r"^i\s*am\s*not\s*(sure|able|certain)\b",
    r"^(sorry|unfortunately)[,.]?\s*(i|but)\b.*\b(don'?t|do not|cannot|can'?t)\b",
    r"^no\s+(information|data|knowledge)\s+(available|known)\b",
    r"^unknown\b",
    r"^n/?a\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(text: str) -> bool:
    if text is None:
        return True
    stripped = str(text).strip()
    if not stripped:
        return True
    return bool(_REFUSAL_RE.search(stripped))


def answer_rate(predictions: Iterable[str]) -> float:
    preds = list(predictions)
    if not preds:
        return 0.0
    substantive = sum(1 for p in preds if not is_refusal(p))
    return substantive / len(preds)
