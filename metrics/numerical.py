"""Numerical accuracy metric.

A prediction is counted as correct if its relative error from the
reference is below `tolerance` (default 2%, following the paper). When
the reference is exactly zero, the prediction must also be exactly zero.
"""

from __future__ import annotations

import re
from typing import Optional

# Matches 1,234 / 12.5 / -3 / 4.2e6 etc.
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_number(text: str) -> Optional[float]:
    """Extract the first number from `text`. Returns None if none found."""
    if text is None:
        return None
    text = str(text).replace("\u2212", "-")  # unicode minus
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def is_numerical_answer(reference: str) -> bool:
    return parse_number(reference) is not None


def numerical_accuracy(prediction: str, reference: str, tolerance: float = 0.02) -> float:
    """1.0 if prediction matches reference within `tolerance` relative error."""
    pred = parse_number(prediction)
    ref = parse_number(reference)
    if ref is None or pred is None:
        return 0.0
    if ref == 0:
        return 1.0 if pred == 0 else 0.0
    return 1.0 if abs(pred - ref) / abs(ref) <= tolerance else 0.0
