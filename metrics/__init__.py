"""Evaluation metrics used by LocalBench.

All metrics expose the signature

    score(prediction: str, reference: str, **kwargs) -> float

and return a value in [0, 1]. A few helpers handle numerical answers,
refusal detection, and GPT-based judging.
"""

from .exact_match import exact_match, normalise
from .rouge1 import rouge1_f1
from .numerical import numerical_accuracy, is_numerical_answer, parse_number
from .semantic import SemanticMatcher
from .gpt_judge import GPTJudge
from .answer_rate import is_refusal, answer_rate

__all__ = [
    "exact_match",
    "normalise",
    "rouge1_f1",
    "numerical_accuracy",
    "is_numerical_answer",
    "parse_number",
    "SemanticMatcher",
    "GPTJudge",
    "is_refusal",
    "answer_rate",
]
