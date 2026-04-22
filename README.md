# LocalBench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2511.10459-b31b1b.svg)](https://arxiv.org/abs/2511.10459)
[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](#citation)

> Benchmarking LLMs on **county-level local knowledge and reasoning** across the United States.

LocalBench is the first benchmark designed to systematically evaluate LLMs on hyper-local, county-level knowledge. It contains **14,782 validated question-answer pairs across 526 U.S. counties in 49 states**, integrating Census statistics, local subreddit discourse, and regional news. It spans the physical, cognitive, and relational dimensions of locality defined by the Localness Conceptual Framework.

This repository ships the dataset, the full evaluation pipeline (query + metrics + aggregation), and a reference implementation of the QA generation pipeline used to build LocalBench.

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/MadCollab/LocalBench.git
cd LocalBench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys (only for the providers you plan to use)
cp .env.example .env
# edit .env -> OPENAI_API_KEY=..., ANTHROPIC_API_KEY=..., GOOGLE_API_KEY=...

# 3. Inspect the dataset
python loader.py            # prints shape + counts by dataset/question_type

# 4. Run the benchmark (querying + evaluation) on a 50-sample smoke test
python benchmark.py --models gpt-4o --sample 50

# 5. Run the full benchmark with everything configured in config.json
python benchmark.py
```

Results land in `outputs/`:

```
outputs/
├── predictions/
│   └── <model_id>.jsonl        # raw model answers
└── reports/
    ├── <model_id>.csv          # per-row metric scores
    ├── <model_id>.jsonl
    └── summary.csv             # aggregated table (Table 2 style)
```

---

## Dataset

| Source         | Rows  | File |
|----------------|------:|------|
| U.S. Census    | 6,120 | `data/census_QA.csv` |
| Local Reddit   | 4,000 | `data/reddit_QA.parquet` |
| NELA-Local     | 4,662 | `data/news_QA.parquet` |

Loaded via `loader.py`, every row exposes the fields:

```python
id, dataset, county, state, fips, rucc, rucc_group, dimension,question_type, question, context, answer
```

See `data/README.md` for the per-file schema, and Appendix 14 of the paper for the full list of census metrics and their provenance.

Distribution by Localness dimension (Table 1):

| Dimension | Count |
|-----------|------:|
| Place Interaction          | 1,330 |
| Temporal Presence          | 2,907 |
| Cultural Understanding     | 1,435 |
| Environmental Cognition    | 1,739 |
| Local Knowledge            | 3,855 |
| Emotional Connection       |   838 |
| Social / Community Engagement | 2,678 |
| **Total**                  | **14,782** |

---

## Benchmarking your own model

### Option A — use one of the pre-wired backends

Set the model id in `config.json` (or pass `--models`) and run `benchmark.py`. Pre-wired ids:

```
gpt-4o                gpt-4.1                gpt-4.1-web
claude-sonnet-3.7     claude-sonnet-4
gemini-2.5-pro        gemini-2.5-flash       gemini-2.5-pro-grounding
qwen3-8b  qwen3-14b  qwen3-32b  qwen3-30b-a3b  qwen3-235b-a22b
```

### Option B — plug in a new backend

Create a subclass of `models.base.BaseModel` and register it in `models/__init__.py`. Every backend just needs to implement:

```python
def generate(self, messages, temperature=0.0, max_tokens=256) -> str: ...
```

`messages` follows the OpenAI chat schema
(`[{"role": "system"|"user"|"assistant", "content": "..."}]`).

### Option C — plain Hugging Face model

Qwen backends already include a local HF path (`models/qwen_models.py`). Set `QWEN_API_BASE` to an OpenAI-compatible endpoint (vLLM, Together, DashScope, Ollama…) for large MoE models, or install `transformers` + `torch` to run the smaller dense models locally.

---

## Metrics

Implemented in `metrics/` and enabled via `config.json → metrics`:

| Metric               | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `exact_match`        | Strict string equality after normalisation                              |
| `rouge1`             | ROUGE-1 F1 (paraphrase-tolerant lexical overlap)                        |
| `semantic_match`     | Cosine similarity over `text-embedding-3-small` embeddings              |
| `numerical_accuracy` | ≤ 2% relative error (exact for zero-valued answers)                     |
| `gpt_judge`          | GPT-4o-mini binary correctness verdict (Appendix 6)                     |
| `answer_rate`        | Fraction of substantive (non-refusal) responses                         |

All metrics run on the standardised loader output, so you can score
arbitrary prediction files:

```python
from evaluate import score_predictions
df = score_predictions("outputs/predictions/gpt-4o.jsonl", metrics_cfg={
    "exact_match": True, "rouge1": True, "semantic_match": False,
    "numerical_accuracy": True, "gpt_judge": False,
})
```

---

## Reproducing the published headline numbers

```bash
# Closed-book proprietary + open-source models (Table 2)
python benchmark.py --models gpt-4o gpt-4.1 gemini-2.5-pro \
                             gemini-2.5-flash claude-sonnet-4 claude-sonnet-3.7 \
                             qwen3-235b-a22b qwen3-32b qwen3-14b qwen3-8b

# Web-augmented configurations
python benchmark.py --models gpt-4.1-web gemini-2.5-pro-grounding
```

All models are called with `temperature=0.0` and `max_tokens=256`, matching the evaluation protocol in the paper. Three independent runs with different seeds can be obtained by repeating the command (the JSONL output is append-safe, and `evaluate.py` averages over everything it finds).

---

## Regenerating the dataset

If you want to extend LocalBench (new counties, new sources, new languages) use the scripts in `qa_generation/`:

```bash
python qa_generation/reddit_qa.py --input raw_reddit.parquet    --output data/reddit_QA.parquet
python qa_generation/news_qa.py   --input raw_news.parquet      --output data/news_QA.parquet
python qa_generation/census_qa.py --counties balanced_rucc_counties.csv \
                                  --mapping  localness_metrics_mapping.csv \
                                  --output   data/census_QA.csv
```

See `qa_generation/README.md` for the input schemas and the full three-stage pipeline (raw generation → multi-rule filter → feedback-driven refinement → localness classification).

---

## Repository layout

```
LocalBench/
├── README.md
├── LICENSE
├── requirements.txt
├── config.json
├── .env.example
├── benchmark.py           # end-to-end runner
├── loader.py              # unified dataset API
├── prompt.py              # prompt construction for evaluation
├── query.py               # per-model inference + JSONL writer
├── evaluate.py            # scoring + aggregation
├── data/                  # the released QA pairs
│   ├── census_QA.csv
│   ├── news_QA.parquet
│   ├── reddit_QA.parquet
│   └── README.md
├── models/                # pluggable LLM backends
│   ├── base.py
│   ├── openai_models.py
│   ├── anthropic_models.py
│   ├── gemini_models.py
│   └── qwen_models.py
├── metrics/               # EM / ROUGE / semantic / numerical / judge / answer-rate
├── prompts/               # prompt templates used in qa_generation/
├── qa_generation/         # reference QA construction pipeline
│   ├── reddit_qa.py
│   ├── news_qa.py
│   ├── census_qa.py
│   ├── quality_analyzer.py
│   └── qa_generation_reference.ipynb
└── outputs/               # predictions + reports (gitignored contents)
```

---

## Citation

If you use LocalBench, please cite:

```bibtex
@article{gao2026localbench, 
  title={LocalBench: Benchmarking LLMs on County-Level Local Knowledge and Reasoning}, 
  author={Gao, Zihan and Xu, Yifei and Thebault-Spieker, Jacob}, 
  volume={40}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/41190}, 
  DOI={10.1609/aaai.v40i45.41190}, 
  number={45}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  year={2026}, 
  month={Mar.}, 
  pages={38487-38495} 
}
```

---

## Ethics & responsible use

LocalBench includes discourse from local subreddits and content from community news outlets. Please:

- Do not use the dataset to re-identify individuals or communities.
- Treat local-knowledge holders as partners, not data sources.
- Follow the content policies of the upstream providers (Reddit, NELA, Census, etc.) when redistributing or building on this work.

See the **Ethical Statement** section of the paper for a fuller discussion of community data sovereignty and participatory development.

---

## Acknowledgements

LocalBench is a project from the University of Wisconsin–Madison and UCLA. We thank the many local subreddit moderators, community news outlets, and public-data curators whose work makes this benchmark possible.
