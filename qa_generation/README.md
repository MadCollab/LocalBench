# QA Generation Pipeline

This directory contains a reference implementation of the pipeline used
to construct LocalBench (Figure 1 in the paper):

1. **Raw Generation** — an `o3` reasoning model proposes candidate QA
   pairs from each source document.
2. **Multi-Rule Filter** — a fine-tuned `gpt-4o-mini` "quality
   analyzer" scores each candidate on the nine criteria in
   Appendix 3.
3. **Feedback-Driven Refinement** — failed pairs are regenerated with
   targeted feedback for up to three rounds.
4. **Localness Attribute Classification** — each surviving pair is
   tagged with localness dimensions and subcomponents.

> The pre-built LocalBench dataset ships in `../data/`. You only need
> the code in this directory if you want to regenerate or extend it.

## Files

| File | Purpose |
|------|---------|
| `quality_analyzer.py`          | The nine DPO-style quality checks |
| `reddit_qa.py`                 | Reddit thread → QA candidates |
| `news_qa.py`                   | NELA-Local article → QA candidates |
| `census_qa.py`                 | Census numeric + comparison QA generator |
| `qa_generation_reference.ipynb`| Original research notebook (unedited) |

## Usage

```bash
# 1. Configure API keys
cp ../.env.example ../.env   # fill in OPENAI_API_KEY etc.

# 2. Regenerate Reddit QA
python reddit_qa.py \
    --input raw_reddit.parquet \
    --output ../data/reddit_QA.parquet

# 3. Regenerate news QA
python news_qa.py \
    --input raw_news.parquet \
    --output ../data/news_QA.parquet

# 4. Regenerate census QA (needs the two auxiliary CSVs described below)
python census_qa.py \
    --counties balanced_rucc_counties.csv \
    --mapping  localness_metrics_mapping.csv \
    --output   ../data/census_QA.csv
```

## External data required for regeneration

The three input files below are **not shipped** with the benchmark
because of licensing and size. Download them from the public sources
listed in the paper and place them anywhere; just point the scripts
at them.

- Reddit: collected via PRAW from the "Global List of Local Reddits";
  one row per thread with columns `title, selftext, comments,
  county, state, created_time, rucc, rucc_group, fips`.
- News: a filtered view of the NELA-Local corpus
  (Horne et al., 2022) with columns `title, factual_content,
  county, state, date, source, rucc, rucc_group, fips`.
- Census: the 681 counties with full coverage of 34 localness
  indicators (`balanced_rucc_counties.csv`) and their text
  descriptions (`localness_metrics_mapping.csv`). See Appendix 14
  of the paper for the full source list.

## Notes

- All API calls use exponential-backoff retry via `backoff`.
- Checkpointing: the Reddit/news scripts write a `checkpoint.json`
  that records which rows have been processed; rerunning the
  script resumes where it left off.
- Human-validated DPO training data for the quality analyzer
  (473 annotated pairs) is available on request from the authors.
