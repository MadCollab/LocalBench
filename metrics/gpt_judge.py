"""GPT Judge metric (Appendix 6 of the paper).

Uses GPT-4o-mini to issue a binary correctness verdict given the question,
gold answer, and generated answer. A reference context can be passed in
when available (e.g. for Reddit/news QA).
"""

from __future__ import annotations

import os
import re

import backoff

JUDGE_PROMPT = (
    "Evaluate if the AI-generated answer is correct based on the question and golden answer.\n\n"
    "- Question: {question}\n"
    "- Context: {context}\n"
    "- Golden Answer: {gold_answer}\n"
    "- AI Answer: {pred_answer}\n\n"
    "## Instructions:\n"
    "- Answer \"Yes\" if the AI answer is factually correct and addresses the question\n"
    "- Answer \"No\"  if the AI answer is factually incorrect or doesn't address the question\n"
    "- Consider partial credit for answers that are mostly correct\n"
    "- Ignore minor wording differences if the core meaning is correct\n\n"
    "Answer (Yes/No):"
)


class GPTJudge:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError("Install `openai` to use GPTJudge.") from e
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is required for GPTJudge.")
            self._client = OpenAI(api_key=key)
        return self._client

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def score(self, question: str, reference: str, prediction: str, context: str = "") -> float:
        prompt = JUDGE_PROMPT.format(
            question=question.strip(),
            context=(context or "").strip() or "(none)",
            gold_answer=str(reference).strip(),
            pred_answer=str(prediction).strip(),
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        )
        text = (resp.choices[0].message.content or "").strip().lower()
        # Accept 'yes', 'yes.', 'y', etc.
        return 1.0 if re.match(r"^y(es)?\b", text) else 0.0
