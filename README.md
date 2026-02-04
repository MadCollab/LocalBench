# LocalBench: County-Level Local Knowledge and Reasoning Benchmark

**LocalBench** is the first large-scale evaluation suite designed to systematically assess the "local/place intelligence" of LLMs at the U.S. county level. While LLMs excel at macro-scale geographic tasks, they often struggle with the hyper-local nuances—administrative, cultural, and relational—that define community life.

## 📍 Overview

LocalBench comprises 14,782 validated question-answer (QA) pairs covering 526 U.S. counties across 49 states. The benchmark is grounded in the Localness Conceptual Framework, evaluating models across three core domains: Physical, Cognitive, and Relational.


## 🏗️ Dataset Construction

The benchmark integrates three complementary data sources to capture both structured facts and unstructured community discourse:

* U.S. Census & Structured Data: 34 indicators (e.g., USDA, CDC, Census) covering 180 stratified urban, suburban, and rural counties.
* Hyper-Local Forums (Reddit): 4,000 QA pairs derived from community discussions in local subreddits.
* Regional News (NELA-Local): 4,662 QA pairs from over 300 local news outlets, focusing on governance and civic events.

### QA Generation Pipeline

All QA pairs are generated through a rigorous four-step pipeline:

1. Raw Generation: Reasoning models generate candidates from source documents.
2. Multi-Rule Filtering: A DPO-tuned analyzer evaluates pairs based on nine criteria, including geographic grounding and privacy.
3. Feedback-Driven Refinement: Iterative regeneration loops for failing pairs.
4. Expert Validation: Human-in-the-loop verification ensuring 94.2% precision.

## 📊 Key Findings

Our evaluation of 13 state-of-the-art LLMs (including GPT-4, Gemini 2.5, and Claude 3.7) reveals critical limitations:

* The Numerical Gap: Even the best models struggle with numerical reasoning, with accuracy typically falling below **15.5%**.
* The Scaling Paradox: Increased model size and Mixture-of-Experts (MoE) architectures do not consistently improve local reasoning.
* Retrieval Variance: Web augmentation improves some models (Gemini +13.6%) while degrading others (GPT-series -11.4%).

## 🚀 Getting Started

The repository is currently being prepared for full public release.

### Access the Dataset

If you are interested in early access for research purposes, please contact:
**Zihan Gao** University of Wisconsin-Madison

📩 [zihan.gao@wisc.edu](mailto:zihan.gao@wisc.edu)

## 📎 Citation

If you use LocalBench in your research, please cite our AAAI 2026 paper:

```bibtex
@inproceedings{gao2026localbench,
  title={LocalBench: Benchmarking LLMs on County-Level Local Knowledge and Reasoning},
  author={Gao, Zihan and Xu, Yifei and Thebault-Spieker, Jacob},
  booktitle={Proceedings of the Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2026}
}

```
