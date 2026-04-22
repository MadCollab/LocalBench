# LocalBench Data

This directory contains the 14,782 released QA pairs.

| File | Rows | Source |
|------|-----:|--------|
| `census_QA.csv`        | 6,120 | U.S. Census + USDA + NRHP + IMLS (numerical + True/False) |
| `reddit_QA.parquet`    | 4,000 | Local subreddits (Jan 2024 – Mar 2025), top-50 comments |
| `news_QA.parquet`      | 4,662 | NELA-Local (Horne et al., 2022) county-tagged articles |

All files share the same logical schema after being passed through
`loader.py`:

```
id              unique id (e.g. "reddit_17")
dataset         {"census", "reddit", "news"}
county          title-cased county name
state           title-cased state name
fips            5-digit FIPS code (string, may be None)
rucc            raw RUCC code 1-9
rucc_group      {"Urban", "Suburban", "Rural"}
dimension       localness dimension label (Appendix 8)
question_type   {"numerical", "non_numerical"}
question        the prompt to give the model
context         passage (reddit + news only, empty for census)
answer          ground-truth answer as a string
```

See the paper (Benchmark Construction) and `../qa_generation/README.md`
for how each file was constructed.
