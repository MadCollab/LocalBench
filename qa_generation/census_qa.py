"""Census QA generation.

Builds the 34 x 3 QA variants per county (numerical fill-in-the-blank
+ two True/False comparisons) and flattens them into the
`data/census_QA.csv` schema.

Inputs:
  - counties CSV: one row per county containing all 34 metric columns
    plus `STATE_NAME`, `COUNTY_NAME`, `fips`, `RUCC`, `RUCC_group`,
    `POP_COU`.
  - mapping CSV: two columns `Column_name`, `Metrics` mapping each
    metric column to a natural-language description (with the phrase
    "this county" to substitute).

Output: `census_QA.csv` in the LocalBench release schema.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ID_COLUMNS = ["STATE_NAME", "COUNTY_NAME", "fips", "RUCC", "RUCC_group", "POP_COU", "POPPCT_RUR"]


def _build_variants(
    counties: pd.DataFrame,
    metric_descriptions: Dict[str, str],
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    metric_columns = [c for c in counties.columns if c not in ID_COLUMNS and c in metric_descriptions]
    df = counties.copy()
    for metric in metric_columns:
        df[f"{metric}_q1"] = ""
        df[f"{metric}_a1"] = np.nan
        df[f"{metric}_q2"] = ""
        df[f"{metric}_a2"] = False
        df[f"{metric}_q3"] = ""
        df[f"{metric}_a3"] = False

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="counties"):
        county_name = row["COUNTY_NAME"]
        state_name = row["STATE_NAME"]
        for metric in metric_columns:
            value = row[metric]
            if pd.isna(value):
                continue
            description = metric_descriptions[metric]
            modified = description.replace("this county", f"{county_name}, {state_name}")
            if modified.endswith("."):
                modified = modified[:-1]
            df.at[idx, f"{metric}_q1"] = f"{modified} is []"
            df.at[idx, f"{metric}_a1"] = value

            others = df[df.index != idx]
            valid = others[others[metric].notna()]
            higher = valid[valid[metric] > value]
            lower = valid[valid[metric] < value]

            compare_base = description.replace("this county", f"{county_name} County, {state_name}")
            if compare_base.endswith("."):
                compare_base = compare_base[:-1]

            if len(higher) > 0:
                h = higher.sample(n=1, random_state=seed).iloc[0]
                df.at[idx, f"{metric}_q2"] = (
                    f"{compare_base} is higher than that in {h['COUNTY_NAME']} County, {h['STATE_NAME']}"
                )
                df.at[idx, f"{metric}_a2"] = False
            elif len(lower) > 0:
                low = lower.sample(n=1, random_state=seed).iloc[0]
                df.at[idx, f"{metric}_q2"] = (
                    f"{compare_base} is higher than that in {low['COUNTY_NAME']} County, {low['STATE_NAME']}"
                )
                df.at[idx, f"{metric}_a2"] = True

            if len(lower) > 0:
                low = lower.sample(n=1, random_state=seed).iloc[0]
                df.at[idx, f"{metric}_q3"] = (
                    f"{compare_base} is higher than that in {low['COUNTY_NAME']}, {low['STATE_NAME']}"
                )
                df.at[idx, f"{metric}_a3"] = True
            elif len(higher) > 0:
                h = higher.sample(n=1, random_state=seed).iloc[0]
                df.at[idx, f"{metric}_q3"] = (
                    f"{compare_base} is higher than that in {h['COUNTY_NAME']}, {h['STATE_NAME']}"
                )
                df.at[idx, f"{metric}_a3"] = False
    return df


def _transform(df: pd.DataFrame, dimension_map: Dict[str, str] | None = None, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)

    metric_names = set()
    for c in df.columns:
        if c.endswith("_q1") or c.endswith("_q2") or c.endswith("_q3"):
            metric_names.add(c.rsplit("_q", 1)[0])
    rows: List[dict] = []
    for idx, row in df.iterrows():
        base = {
            "STATE_NAME": row["STATE_NAME"],
            "COUNTY_NAME": row["COUNTY_NAME"],
            "fips": row.get("fips"),
            "POP_COU": row.get("POP_COU"),
            "RUCC": row.get("RUCC"),
            "RUCC_group": row.get("RUCC_group"),
        }
        for metric in metric_names:
            cols = [f"{metric}_q1", f"{metric}_q2", f"{metric}_q3",
                    f"{metric}_a1", f"{metric}_a2", f"{metric}_a3"]
            if not all(c in df.columns for c in cols):
                continue
            variant = np.random.choice(["q1", "q2", "q3"], p=[0.5, 0.25, 0.25])
            if variant == "q1":
                q, a, qt = row[f"{metric}_q1"], row[f"{metric}_a1"], "numerical"
            elif variant == "q2":
                q, a, qt = row[f"{metric}_q2"], row[f"{metric}_a2"], "text"
            else:
                q, a, qt = row[f"{metric}_q3"], row[f"{metric}_a3"], "text"
            if not q:
                continue
            record = dict(base, metric=metric, question_type=qt, question=q,
                          answer=a, selected_variant=variant)
            if dimension_map and metric in dimension_map:
                record["Dimension"] = dimension_map[metric]
            rows.append(record)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate census QA pairs.")
    parser.add_argument("--counties", required=True,
                        help="CSV of balanced counties with metric columns")
    parser.add_argument("--mapping", required=True,
                        help="CSV with Column_name -> Metrics description (and optional Dimension)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counties = pd.read_csv(args.counties, encoding="cp1252")
    mapping = pd.read_csv(args.mapping)
    descriptions = dict(zip(mapping["Column_name"], mapping["Metrics"]))
    dimension_map = (
        dict(zip(mapping["Column_name"], mapping["Dimension"]))
        if "Dimension" in mapping.columns else None
    )

    variants = _build_variants(counties, descriptions, seed=args.seed)
    flat = _transform(variants, dimension_map=dimension_map, seed=args.seed)
    flat.to_csv(args.output, index=False)
    print(f"Wrote {len(flat)} QA rows to {args.output}")


if __name__ == "__main__":
    main()
