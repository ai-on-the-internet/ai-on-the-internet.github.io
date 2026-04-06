#!/usr/bin/env python3
"""
Export per-participant survey data from the ai-prevalence repo for the website.

Run this from the ai-prevalence repo (alongside export_website_data.py),
pointed at the raw participant export (Qualtrics / Prolific CSV):

    python export_survey_data.py --input data/survey/participants.csv \
                                 --output output/survey.csv

Produces a single CSV with one row per participant and these columns:

    participant_id, ai_usage, ai_view, h1, h2, h3, h4, h5, h6

Where:
    ai_usage  ∈ {Never, Monthly, Weekly, Daily}
    ai_view   ∈ {Negative, Neutral, Positive}
    h1..h6    ∈ {SD, D, SoD, N, SoA, A, SA}   (7-point Likert short codes)

The hypothesis order matches assets/scripts/plots.js SURVEY_DATA:
    h1 = Semantic Contraction
    h2 = Truth Decay
    h3 = Positivity Shift
    h4 = Epistemic Islands
    h5 = Entropy Dilution
    h6 = Stylistic Monoculture

The website (plots.js) consumes this CSV and computes the `overall`,
`byUsage`, and `byView` count tables itself, replacing the previous
hardcoded SURVEY_DATA constants.

----------------------------------------------------------------------
CONFIGURATION
----------------------------------------------------------------------
Edit COLUMN_MAP below if your raw CSV uses different column names.
The script also accepts free-form Likert text and normalizes it, so
small wording differences in the source data are fine.
"""

import argparse
import csv
import re
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# CONFIG: source column names in the raw participant CSV.
# Adjust these to match your Qualtrics / Prolific export.
# ----------------------------------------------------------------------
COLUMN_MAP = {
    "participant_id": "ResponseId",   # or "PROLIFIC_PID"
    "ai_usage":       "Q_AI_USAGE",   # frequency-of-AI-use question
    "ai_view":        "Q_AI_VIEW",    # general view of AI's impact
    "h1":             "Q_H1",         # Semantic Contraction
    "h2":             "Q_H2",         # Truth Decay
    "h3":             "Q_H3",         # Positivity Shift
    "h4":             "Q_H4",         # Epistemic Islands
    "h5":             "Q_H5",         # Entropy Dilution
    "h6":             "Q_H6",         # Stylistic Monoculture
}

# Rows to skip at the top of a Qualtrics export (the 2 metadata header rows
# that follow the column-name row). Set to 0 if your CSV has no extras.
QUALTRICS_METADATA_ROWS = 2


# ----------------------------------------------------------------------
# Normalization tables
# ----------------------------------------------------------------------

# 7-point Likert → canonical short code
LIKERT_MAP = {
    "strongly disagree":         "SD",
    "disagree":                  "D",
    "somewhat disagree":         "SoD",
    "slightly disagree":         "SoD",
    "neither agree nor disagree": "N",
    "neutral":                   "N",
    "neither":                   "N",
    "somewhat agree":            "SoA",
    "slightly agree":            "SoA",
    "agree":                     "A",
    "strongly agree":            "SA",
    # numeric (1=SD ... 7=SA)
    "1": "SD", "2": "D", "3": "SoD", "4": "N", "5": "SoA", "6": "A", "7": "SA",
}

# AI-usage frequency → 4 buckets used by the website
USAGE_MAP = {
    "never":                "Never",
    "rarely":               "Never",
    "less than once a month": "Never",
    "monthly":              "Monthly",
    "once a month":         "Monthly",
    "a few times a month":  "Monthly",
    "2-3 times a month":    "Monthly",
    "weekly":               "Weekly",
    "once a week":          "Weekly",
    "a few times a week":   "Weekly",
    "2-3 times a week":     "Weekly",
    "daily":                "Daily",
    "every day":            "Daily",
    "multiple times a day": "Daily",
    "several times a day":  "Daily",
}

# General view of AI's impact → 3 buckets
VIEW_MAP = {
    "very negative":      "Negative",
    "negative":           "Negative",
    "somewhat negative":  "Negative",
    "slightly negative":  "Negative",
    "mostly negative":    "Negative",
    "neutral":            "Neutral",
    "neither":            "Neutral",
    "mixed":              "Neutral",
    "no opinion":         "Neutral",
    "very positive":      "Positive",
    "positive":           "Positive",
    "somewhat positive":  "Positive",
    "slightly positive":  "Positive",
    "mostly positive":    "Positive",
    # numeric 1..5 fallback (1=very neg, 5=very pos)
    "1": "Negative", "2": "Negative", "3": "Neutral", "4": "Positive", "5": "Positive",
}


def _norm(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".!?")
    return s


def _lookup(value, table, field, pid, strict):
    key = _norm(value)
    if key in table:
        return table[key]
    # try splitting "5 - Strongly agree" style
    parts = re.split(r"\s*[-–:]\s*", key, maxsplit=1)
    if len(parts) == 2:
        for p in parts:
            if p in table:
                return table[p]
    msg = f"  WARN: participant {pid}: unrecognized {field} value: {value!r}"
    if strict:
        raise ValueError(msg)
    print(msg, file=sys.stderr)
    return ""


def _open_rows(path):
    """Yield dict rows, skipping Qualtrics metadata rows if present."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Detect Qualtrics: row 2 is JSON-ish or labels, row 3 is import IDs.
        # Peek at the next row to decide.
        peek = next(reader, None)
        skip = 0
        if peek and (peek[0].startswith("{") or peek[0].startswith('"{')
                     or any('ImportId' in c for c in peek)):
            skip = QUALTRICS_METADATA_ROWS - 1  # we already consumed one
            for _ in range(skip):
                next(reader, None)
        elif peek is not None:
            # Not a Qualtrics metadata row — yield it as data.
            yield dict(zip(header, peek))
        for row in reader:
            yield dict(zip(header, row))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True,
                    help="Path to raw participant CSV (Qualtrics/Prolific export)")
    ap.add_argument("--output", "-o", default="output/survey.csv",
                    help="Output CSV path (default: output/survey.csv)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail on any unrecognized response value instead of warning")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        sys.exit(f"ERROR: input file not found: {in_path}")

    out_fields = ["participant_id", "ai_usage", "ai_view",
                  "h1", "h2", "h3", "h4", "h5", "h6"]

    n_in = 0
    n_out = 0
    n_dropped = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for row in _open_rows(in_path):
            n_in += 1
            try:
                pid = row.get(COLUMN_MAP["participant_id"], "").strip()
                if not pid:
                    pid = f"p{n_in}"

                usage = _lookup(row.get(COLUMN_MAP["ai_usage"], ""),
                                USAGE_MAP, "ai_usage", pid, args.strict)
                view  = _lookup(row.get(COLUMN_MAP["ai_view"], ""),
                                VIEW_MAP,  "ai_view",  pid, args.strict)

                hyps = {}
                for h in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    hyps[h] = _lookup(row.get(COLUMN_MAP[h], ""),
                                      LIKERT_MAP, h, pid, args.strict)

                # Drop participants missing required fields (matches the
                # paper's per-hypothesis n of 299–303 — small dropouts).
                if not usage or not view:
                    n_dropped += 1
                    continue

                writer.writerow({
                    "participant_id": pid,
                    "ai_usage": usage,
                    "ai_view":  view,
                    **hyps,
                })
                n_out += 1
            except ValueError as e:
                print(e, file=sys.stderr)
                n_dropped += 1

    print(f"Read     {n_in} rows from {in_path}")
    print(f"Wrote    {n_out} participants → {out_path}")
    if n_dropped:
        print(f"Dropped  {n_dropped} rows (missing usage/view or strict failure)")
    print()
    print("Next: copy this file into the website repo at  data/survey.csv")
    print("      and update assets/scripts/plots.js to load + aggregate it.")


if __name__ == "__main__":
    main()
