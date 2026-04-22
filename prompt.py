"""Prompt construction for LocalBench evaluation.

Each LLM receives a standardised instruction that asks it to answer
county-level local questions concisely. Separate prompts are used for
numerical and non-numerical questions. Context is never provided at
evaluation time (closed-book and web-augmented settings), following
the paper.
"""

from __future__ import annotations

from typing import Optional

from loader import QAItem


SYSTEM_PROMPT = (
    "You are a helpful assistant with knowledge of U.S. county-level local "
    "information, including Census statistics, local news, and community "
    "discussions. Answer each question as concisely as possible. If you "
    "genuinely do not know, reply exactly with 'I don't know.'"
)


NUMERICAL_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Answer with a single number only (no units, no commas, no explanation). "
    "If you do not know, reply exactly with 'I don't know.'"
)


NONNUMERICAL_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Answer in one short phrase or sentence. Do not add extra commentary. "
    "If the question is a True/False comparison, answer exactly 'True' or 'False'."
)


CONTEXT_USER_TEMPLATE = (
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer in one short phrase or sentence using the context if helpful."
)


def build_prompt(item: QAItem, include_context: bool = False) -> dict:
    """Return a dict with `system` and `user` prompt strings."""
    if include_context and item.context:
        user = CONTEXT_USER_TEMPLATE.format(
            question=item.question.strip(), context=item.context.strip()
        )
    elif item.question_type == "numerical":
        user = NUMERICAL_USER_TEMPLATE.format(question=item.question.strip())
    else:
        user = NONNUMERICAL_USER_TEMPLATE.format(question=item.question.strip())
    return {"system": SYSTEM_PROMPT, "user": user}


def as_messages(prompt: dict) -> list:
    """Convert a prompt dict into an OpenAI-style chat messages list."""
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
