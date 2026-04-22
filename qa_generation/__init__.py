"""Reference implementation of the LocalBench QA generation pipeline.

See README.md in this directory for usage. The `quality_analyzer`
module contains the nine DPO-style quality checks described in
Appendix 3 of the paper; `reddit_qa`, `news_qa`, and `census_qa`
implement the three data-source-specific generators.
"""
