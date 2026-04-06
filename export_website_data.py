#!/usr/bin/env python3
"""
Export website data from run_all_hypotheses.py results.

Run this AFTER running:
    python run_all_hypotheses.py --sources internet --format png

Usage:
    python export_website_data.py [--plots-dir plots/] [--output-dir output/]

Reads the hypothesis_N_results.json files produced by run_all_hypotheses.py
and the main internet.csv to generate two CSV files for the website:
    output/prevalence.csv   - Monthly AI prevalence (for the main figure)
    output/hypotheses.csv   - Monthly hypothesis signals (for scatter/timeseries plots)
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict


def load_prevalence_from_csv(csv_path):
    """Compute monthly prevalence from internet.csv using v3 classifications."""
    import pandas as pd

    print(f"Loading {csv_path} for prevalence data...")
    df = pd.read_csv(csv_path, dtype=str)
    print(f"  {len(df)} rows loaded")

    # Parse dates (wayback timestamp format: YYYYMMDDHHmmss)
    df['year_month'] = df['date'].str[:4] + '-' + df['date'].str[4:6]

    results = []
    for ym in sorted(df['year_month'].unique()):
        subset = df[df['year_month'] == ym]
        scored = subset[subset['score_pangram_v3_prediction_short'].notna() &
                        (subset['score_pangram_v3_prediction_short'] != '')]

        if len(scored) == 0:
            results.append({'month': ym, 'ai_generated_pct': 0, 'ai_combined_pct': 0})
            continue

        ai_count = (scored['score_pangram_v3_prediction_short'] == 'AI').sum()
        mixed_count = (scored['score_pangram_v3_prediction_short'] == 'Mixed').sum()
        total = len(scored)

        results.append({
            'month': ym,
            'ai_generated_pct': round(ai_count / total * 100, 2),
            'ai_combined_pct': round((ai_count + mixed_count) / total * 100, 2),
        })

    return results


def load_hypothesis_results(plots_dir, source='internet'):
    """Load monthly results from all hypothesis JSON files."""
    # Map hypothesis number to the signal key in the JSON
    signal_keys = {
        1: 'avg_semantic_similarity',
        2: 'false_ratio',       # H2 uses false_ratio
        3: 'positive_rate',
        4: 'avg_link_density',
        5: 'avg_compression_ratio',
        6: 'avg_style_similarity',
    }

    csv_columns = {
        1: 'h1_cosine_similarity',
        2: 'h2_error_rate',
        3: 'h3_positive_rate',
        4: 'h4_link_density',
        5: 'h5_compression_ratio',
        6: 'h6_jaccard_similarity',
    }

    # Collect all monthly data, keyed by month
    monthly = defaultdict(lambda: {'ai_likelihood': None})

    for h in range(1, 7):
        json_path = plots_dir / f'hypothesis_{h}' / f'hypothesis_{h}_results.json'
        if not json_path.exists():
            print(f"  WARNING: {json_path} not found, skipping H{h}")
            continue

        with open(json_path) as f:
            data = json.load(f)

        if source not in data:
            print(f"  WARNING: source '{source}' not in H{h} results, skipping")
            continue

        monthly_data = data[source].get('monthly', [])
        signal_key = signal_keys[h]
        col = csv_columns[h]

        for entry in monthly_data:
            month = entry['month']
            monthly[month][col] = entry.get(signal_key)

            # Use ai_likelihood from whichever hypothesis has it
            if monthly[month]['ai_likelihood'] is None:
                ai_val = entry.get('avg_ai_likelihood')
                if ai_val is not None:
                    monthly[month]['ai_likelihood'] = ai_val

    return monthly


def main():
    parser = argparse.ArgumentParser(description='Export website data from hypothesis results')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Directory containing hypothesis results (default: plots/)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for CSV files (default: output/)')
    parser.add_argument('--source', type=str, default='internet',
                        help='Data source to export (default: internet)')
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    csv_path = Path('data') / f'{args.source}.csv'

    # ---- Prevalence CSV ----
    print("\n=== Prevalence Data ===")
    if csv_path.exists():
        # Check for LFS pointer
        with open(csv_path, 'r') as f:
            first = f.readline()
        if 'git-lfs' in first:
            print(f"ERROR: {csv_path} is a Git LFS pointer. Run 'git lfs pull' first.")
            sys.exit(1)

        prevalence = load_prevalence_from_csv(csv_path)
        prev_path = output_dir / 'prevalence.csv'
        with open(prev_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['month', 'ai_generated_pct', 'ai_combined_pct'])
            writer.writeheader()
            writer.writerows(prevalence)
        print(f"  Wrote {prev_path} ({len(prevalence)} months)")
    else:
        print(f"  WARNING: {csv_path} not found, skipping prevalence export")

    # ---- Hypotheses CSV ----
    print("\n=== Hypothesis Data ===")
    if not plots_dir.exists():
        print(f"ERROR: {plots_dir} not found. Run run_all_hypotheses.py first.")
        sys.exit(1)

    monthly = load_hypothesis_results(plots_dir, args.source)

    if not monthly:
        print("  ERROR: No hypothesis results found.")
        sys.exit(1)

    fields = ['month', 'ai_likelihood', 'h1_cosine_similarity', 'h2_error_rate',
              'h3_positive_rate', 'h4_link_density', 'h5_compression_ratio', 'h6_jaccard_similarity']

    hyp_path = output_dir / 'hypotheses.csv'
    with open(hyp_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for month in sorted(monthly.keys()):
            row = [month]
            for field in fields[1:]:
                val = monthly[month].get(field)
                if val is None:
                    row.append('')
                elif isinstance(val, float):
                    row.append(f"{val:.6f}")
                else:
                    row.append(str(val))
            writer.writerow(row)
    print(f"  Wrote {hyp_path} ({len(monthly)} months)")

    print(f"\n{'='*60}")
    print(f"Done! Copy these files to your website repo's data/ directory:")
    print(f"  {prev_path}")
    print(f"  {hyp_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
