"""Minimal end-to-end smoke test (no API keys required).

Registers a deterministic stub model, runs the full query + evaluate
pipeline on a 10-sample slice, and prints the resulting summary.
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loader import load_all  # noqa: E402
from query import run_model  # noqa: E402
from evaluate import evaluate_all  # noqa: E402
from metrics import exact_match, normalise  # noqa: E402


class EchoModel:
    """Answers "True" for comparisons and echoes the reference for census numbers."""
    public_id = "echo-stub"

    def generate(self, messages, temperature=0.0, max_tokens=256):
        # Very naive deterministic behaviour so that scoring exercises all
        # metrics without any network access.
        user = next(m["content"] for m in messages if m["role"] == "user").lower()
        if "true" in user or "false" in user or "higher than" in user:
            return "True"
        return "I don't know"


def main():
    os.environ.setdefault("OPENAI_API_KEY", "")
    items = load_all(data_dir=os.path.join(ROOT, "data"),
                     datasets=["census", "reddit", "news"],
                     sample_size_per_dataset=3, random_seed=0)
    print(f"Loaded {len(items)} items")

    with tempfile.TemporaryDirectory() as tmp:
        model = EchoModel()
        run_model(model, items, output_dir=tmp, concurrency=1, resume=False)

        summary = evaluate_all(
            output_dir=tmp,
            config={
                "metrics": {
                    "exact_match": True,
                    "rouge1": True,
                    "semantic_match": False,   # requires OpenAI
                    "numerical_accuracy": True,
                    "gpt_judge": False,        # requires OpenAI
                },
                "evaluate": {"numerical_tolerance": 0.02},
            },
        )
        print("\nSMOKE TEST PASSED")
        print(summary.to_string(index=False))

    # Unit sanity checks
    assert exact_match("True", "True") == 1.0
    assert normalise("The Answer! ") == "answer"
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
