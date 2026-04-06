#!/usr/bin/env python3
"""
Export survey data from the ai-prevalence repo for the website.

Run from the ai-prevalence repo:

    python export_survey_data.py \
        --data-dir human_study/data \
        --output-dir output

Reads the three Prolific response files used by analyze_human_study.py:

    human_study/data/prolific-part1-responses.csv   (H1, H2, H3)
    human_study/data/prolific-part2-responses.csv   (H4, H6)
    human_study/data/prolific-part3-responses.csv   (H5)

…and writes pre-aggregated CSVs the website (assets/scripts/plots.js)
can load directly:

    output/survey_overall.csv   — overall AI-usage and AI-view distributions
                                  (drives the two figures in
                                  "What is the public's perception of AI's
                                  impact on the internet?")

    output/survey_hypotheses.csv — per-hypothesis Likert counts, broken down
                                  overall, by AI-usage bucket, and by
                                  AI-view bucket. Drives the three small
                                  charts under each Hypothesis section.

Likert short codes match plots.js: SD, D, SoD, N, SoA, A, SA.
AI-usage buckets:  Never | Monthly | Weekly | Daily.
AI-view buckets:   Negative | Neutral | Positive
                   (negative-leaning, neutral, positive-leaning).
"""

import argparse
import csv
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------
# Column names in the Prolific exports (taken verbatim from
# analyze_human_study.py — must match the source CSVs exactly).
# ----------------------------------------------------------------------

USAGE_COL = "How often do you use AI tools (e.g., ChatGPT, Claude, Gemini)?"
VIEW_COL  = "In general, how do you view the impact of AI on society as a whole?"

H1_COL = ("As AI content becomes more common on the internet, the range of "
          "unique ideas and diverse viewpoints seems to be shrinking.")
H2_COL = ("As AI content becomes more common on the internet, I am encountering "
          "factually incorrect information and hallucinations more frequently.")
H3_COL = ("As AI content becomes more common on the internet, online writing "
          "feels increasingly sanitized and artificially cheerful.")
H4_COL = ("As AI content becomes more common on the internet, articles are "
          "increasingly providing answers without including links to external "
          "sources.")
# Note: this column name has a stray trailing double-quote in the raw export.
H6_COL = ('As AI content becomes more common on the internet, distinct '
          'individual writing styles are disappearing in favor of a generic, '
          'uniform voice."')
H5_COL = ("As AI content becomes more common on the internet, content is "
          "becoming significantly longer in word count while containing less "
          "actual meaning.")


# ----------------------------------------------------------------------
# Canonical orderings / mappings (match plots.js)
# ----------------------------------------------------------------------

LIKERT_ORDER = ["SD", "D", "SoD", "N", "SoA", "A", "SA"]

LIKERT_MAP = {
    "Strongly Disagree":           "SD",
    "Disagree":                    "D",
    "Somewhat Disagree":           "SoD",
    "Neither Agree nor Disagree":  "N",
    "Somewhat Agree":              "SoA",
    "Agree":                       "A",
    "Strongly Agree":              "SA",
}

USAGE_ORDER = ["Never", "Monthly", "Weekly", "Daily"]

# Raw responses → 4-bucket usage. The Prolific question uses these exact labels.
USAGE_MAP = {
    "Never":   "Never",
    "Monthly": "Monthly",
    "Weekly":  "Weekly",
    "Daily":   "Daily",
}

VIEW_ORDER = ["Negative", "Neutral", "Positive"]

# 7-point sentiment → 3 buckets. Anything containing "Negative" is Negative,
# anything containing "Positive" is Positive, "Neither..." is Neutral.
VIEW_MAP = {
    "Very Negative":                 "Negative",
    "Negative":                      "Negative",
    "Somewhat Negative":             "Negative",
    "Neither Positive nor Negative": "Neutral",
    "Somewhat Positive":             "Positive",
    "Positive":                      "Positive",
    "Very Positive":                 "Positive",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load_part(data_dir: Path, part: int, hyp_cols: dict) -> pd.DataFrame:
    """Load one Prolific part and return a tidy frame.

    `hyp_cols` maps {short_name: source_column}, e.g. {"h1": H1_COL}.
    """
    path = data_dir / f"prolific-part{part}-responses.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    out = pd.DataFrame({
        "participant_id": df["Participant id"],
        "part":           part,
        "ai_usage_raw":   df[USAGE_COL],
        "ai_view_raw":    df[VIEW_COL],
    })
    for short, src in hyp_cols.items():
        out[short] = df[src]
    return out


def _bucket(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping)


def _likert_counts(series: pd.Series) -> dict:
    """Map full Likert text → short codes and return counts in canonical order."""
    short = series.map(LIKERT_MAP).dropna()
    counts = short.value_counts().to_dict()
    return {code: int(counts.get(code, 0)) for code in LIKERT_ORDER}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", default="human_study/data",
                    help="Directory containing prolific-part{1,2,3}-responses.csv")
    ap.add_argument("--output-dir", default="output",
                    help="Where to write survey_overall.csv and survey_hypotheses.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all three parts
    # ------------------------------------------------------------------
    p1 = _load_part(data_dir, 1, {"h1": H1_COL, "h2": H2_COL, "h3": H3_COL})
    p2 = _load_part(data_dir, 2, {"h4": H4_COL, "h6": H6_COL})
    p3 = _load_part(data_dir, 3, {"h5": H5_COL})

    print(f"Part 1: {len(p1)} responses (H1, H2, H3)")
    print(f"Part 2: {len(p2)} responses (H4, H6)")
    print(f"Part 3: {len(p3)} responses (H5)")

    # ------------------------------------------------------------------
    # OVERALL distributions for the perception section.
    #
    # The website's "AI Usage Frequency" and "View of AI Impact" figures
    # are described as "We surveyed 303 US adults" — that's the Part 1
    # sample, where H1/H2/H3 are asked. Use Part 1 to drive these two
    # overall figures so the headline n matches.
    # ------------------------------------------------------------------
    overall = p1.copy()
    overall["ai_usage"] = _bucket(overall["ai_usage_raw"], USAGE_MAP)
    overall["ai_view"]  = _bucket(overall["ai_view_raw"],  VIEW_MAP)

    n_overall = int(overall["ai_usage"].notna().sum())
    usage_counts = (
        overall["ai_usage"].value_counts()
        .reindex(USAGE_ORDER, fill_value=0)
        .astype(int)
    )
    view_counts = (
        overall["ai_view"].value_counts()
        .reindex(VIEW_ORDER, fill_value=0)
        .astype(int)
    )

    overall_path = out_dir / "survey_overall.csv"
    with open(overall_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "bucket", "count", "pct"])
        for bucket in USAGE_ORDER:
            c = int(usage_counts[bucket])
            w.writerow(["ai_usage", bucket, c,
                        round(c / n_overall * 100, 2) if n_overall else 0])
        for bucket in VIEW_ORDER:
            c = int(view_counts[bucket])
            w.writerow(["ai_view", bucket, c,
                        round(c / n_overall * 100, 2) if n_overall else 0])
    print(f"Wrote {overall_path}  (n={n_overall})")

    # ------------------------------------------------------------------
    # PER-HYPOTHESIS Likert counts (overall + by usage + by view).
    #
    # Each hypothesis pulls from its own part, so the n per hypothesis
    # varies (matching the paper's reported 299–303).
    # ------------------------------------------------------------------
    hyp_sources = [
        ("h1", p1, "h1"),
        ("h2", p1, "h2"),
        ("h3", p1, "h3"),
        ("h4", p2, "h4"),
        ("h5", p3, "h5"),
        ("h6", p2, "h6"),
    ]

    hyp_path = out_dir / "survey_hypotheses.csv"
    with open(hyp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hypothesis", "split", "group", "n"] + LIKERT_ORDER)

        for hyp_key, part_df, col in hyp_sources:
            df = part_df.copy()
            df["ai_usage"] = _bucket(df["ai_usage_raw"], USAGE_MAP)
            df["ai_view"]  = _bucket(df["ai_view_raw"],  VIEW_MAP)

            # Overall row
            counts = _likert_counts(df[col])
            n = sum(counts.values())
            w.writerow([hyp_key, "overall", "all", n] + [counts[c] for c in LIKERT_ORDER])

            # By AI usage
            for bucket in USAGE_ORDER:
                sub = df[df["ai_usage"] == bucket]
                counts = _likert_counts(sub[col])
                n = sum(counts.values())
                w.writerow([hyp_key, "by_usage", bucket, n]
                           + [counts[c] for c in LIKERT_ORDER])

            # By AI view
            for bucket in VIEW_ORDER:
                sub = df[df["ai_view"] == bucket]
                counts = _likert_counts(sub[col])
                n = sum(counts.values())
                w.writerow([hyp_key, "by_view", bucket, n]
                           + [counts[c] for c in LIKERT_ORDER])

    print(f"Wrote {hyp_path}  (6 hypotheses × {1 + len(USAGE_ORDER) + len(VIEW_ORDER)} rows each)")

    print()
    print("Next: copy these into the website repo's data/ directory:")
    print(f"  {overall_path}  ->  data/survey_overall.csv")
    print(f"  {hyp_path}  ->  data/survey_hypotheses.csv")


if __name__ == "__main__":
    main()
