"""QA Quality Analyzer (Appendix 3 of the paper).

Implements the nine quality criteria applied to Reddit- and news-derived
QA pairs. Uses a fine-tuned `gpt-4o-mini` model by default, but any
OpenAI-compatible model id works. Each criterion returns a yes/no
verdict and the full set of verdicts is combined into a pass/fail
decision.

The criteria are:

    1. fact_based_single_answer
    2. no_subjectivity
    3. no_meta_reference            (Reddit) / no_meta_content (news)
    4. context_location_time
    5. no_answer_leakage
    6. privacy_safety
    7. local_grounding
    8. difficulty_assessment
    9. question_clarity
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import backoff


REQUIREMENT_PROMPTS: Dict[str, str] = {
    "fact_based_single_answer": (
        "Determine whether the question has exactly one correct, fact-based answer.\n\n"
        "Question: {question}\nContext: {context}\nAnswer: {answer}\n\n"
        "A question meets this requirement if:\n"
        "- It has one factual, verifiable answer\n"
        "- It is not subjective, opinion-based, or open-ended\n"
        "- It does not depend on user preferences, multiple viewpoints, or recommendations\n\n"
        "Respond in JSON: {{\"requirement\": \"fact_based_single_answer\", \"meetornot\": \"yes/no\"}}"
    ),
    "no_subjectivity": (
        "Does this question avoid subjectivity (feelings, recommendations, opinions)?\n\n"
        "Question: {question}\n\n"
        "It meets the requirement if it does NOT ask about:\n"
        "- How users felt\n- What users recommended\n- Subjective opinions or preferences\n- Vague 'why' questions\n\n"
        "Respond in JSON: {{\"requirement\": \"no_subjectivity\", \"meetornot\": \"yes/no\"}}"
    ),
    "no_meta_reference": (
        "Determine whether the QA avoids referencing a Reddit post or news article as a source.\n\n"
        "Question: {question}\nContext: {context}\nAnswer: {answer}\n\n"
        "It must NOT refer to 'the thread/post/article', 'commenters', 'what users said', etc.\n"
        "The QA pair should stand alone without relying on such sources.\n\n"
        "Respond in JSON: {{\"requirement\": \"no_meta_reference\", \"meetornot\": \"yes/no\"}}"
    ),
    "context_location_time": (
        "Determine whether the combined question and context provide a specific date, "
        "a county name, and a state name.\n\n"
        "Question: {question}\nContext: {context}\n\n"
        "Respond in JSON: {{\"requirement\": \"context_location_time\", \"meetornot\": \"yes/no\"}}"
    ),
    "no_answer_leakage": (
        "Determine whether the context avoids directly revealing the answer.\n\n"
        "Question: {question}\nContext: {context}\nAnswer: {answer}\n\n"
        "Respond in JSON: {{\"requirement\": \"no_answer_leakage\", \"meetornot\": \"yes/no\"}}"
    ),
    "privacy_safety": (
        "Does this QA pair respect privacy and safety guidelines? No phone numbers, "
        "addresses, personally-identifying info, or doxxing.\n\n"
        "Question: {question}\nContext: {context}\nAnswer: {answer}\n\n"
        "Respond in JSON: {{\"requirement\": \"privacy_safety\", \"meetornot\": \"yes/no\"}}"
    ),
    "local_grounding": (
        "Is this QA pair about local information of {county}, {state}?\n\n"
        "Question: {question}\nContext: {context}\nAnswer: {answer}\n\n"
        "Respond in JSON: {{\"requirement\": \"local_grounding\", \"meetornot\": \"yes/no\"}}"
    ),
    "difficulty_assessment": (
        "Is the question neither trivial (answerable with basic general knowledge) "
        "nor impossibly difficult (requiring highly specialized expertise)?\n\n"
        "Question: {question}\nContext: {context}\nAnswer: {answer}\n\n"
        "Respond in JSON: {{\"requirement\": \"difficulty_assessment\", \"meetornot\": \"yes/no\"}}"
    ),
    "question_clarity": (
        "Is the question clearly formulated, unambiguous, and understandable with "
        "only the county specified?\n\n"
        "Question: {question}\nContext: {context}\n\n"
        "Respond in JSON: {{\"requirement\": \"question_clarity\", \"meetornot\": \"yes/no\"}}"
    ),
}


@dataclass
class QualityResult:
    passed: bool
    verdicts: Dict[str, str] = field(default_factory=dict)
    failed: List[str] = field(default_factory=list)


class QualityAnalyzer:
    """GPT-backed QA quality analyser with DPO-compatible prompts."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        openai_api_key: str | None = None,
    ):
        self.model = model
        self._api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY is required for QualityAnalyzer.")
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _query(self, prompt: str) -> Dict[str, str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        text = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            verdict = "yes" if "yes" in text.lower() else "no"
            return {"meetornot": verdict}

    def evaluate(
        self,
        *,
        question: str,
        context: str,
        answer: str,
        county: str,
        state: str,
        requirements: List[str] | None = None,
    ) -> QualityResult:
        """Check a QA pair against the requested subset of requirements."""
        requirements = requirements or list(REQUIREMENT_PROMPTS)
        verdicts: Dict[str, str] = {}
        failed: List[str] = []
        for req in requirements:
            tmpl = REQUIREMENT_PROMPTS[req]
            prompt = tmpl.format(
                question=question, context=context, answer=answer,
                county=county, state=state,
            )
            result = self._query(prompt)
            verdict = str(result.get("meetornot", "no")).lower()
            verdicts[req] = verdict
            if verdict != "yes":
                failed.append(req)
        return QualityResult(passed=not failed, verdicts=verdicts, failed=failed)
