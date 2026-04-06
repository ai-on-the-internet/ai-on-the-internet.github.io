#!/usr/bin/env python3
"""
Preprocessing script for the ai-on-the-internet.github.io website.

Run this inside the ai-prevalence repo directory:
    python preprocess_data.py

It will generate two CSV files in an output/ directory:
    output/prevalence.csv   - Monthly AI prevalence data (for the main figure)
    output/hypotheses.csv   - Monthly hypothesis signals (for all 6 hypothesis figures)

Requirements (install via pip):
    pip install sentence-transformers transformers torch pandas beautifulsoup4

The script expects the following data files (from the ai-prevalence repo):
    data/internet.csv                - Main internet dataset
    data/html_internet/              - HTML files for link density
    factuality_analysis/annotations.csv - Fact-checking annotations
"""

import csv
import json
import gzip
import re
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_internet_csv(filepath):
    """Load the main internet.csv dataset."""
    print(f"Loading {filepath}...")
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows)} rows")
    return rows


def load_annotations(filepath):
    """Load fact-checking annotations."""
    print(f"Loading {filepath}...")
    annotations = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(row)
    print(f"  Loaded {len(annotations)} annotations")
    return annotations


def group_by_month(rows):
    """Group rows by year-month, parsing wayback timestamps."""
    monthly = defaultdict(list)
    for i, row in enumerate(rows):
        date_str = row.get('date', '')
        try:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            monthly[f"{year}-{month:02d}"].append((i, row))
        except (ValueError, IndexError):
            continue
    return monthly


def compute_prevalence(monthly_rows):
    """Compute monthly AI prevalence using Pangram v3 classifications."""
    print("\nComputing prevalence data...")
    results = []
    for month in sorted(monthly_rows.keys()):
        items = monthly_rows[month]
        v3_n = 0
        v3_ai = 0
        v3_mixed = 0
        for _, row in items:
            pred = row.get('score_pangram_v3_prediction_short', '')
            if pred:
                v3_n += 1
                if pred == 'AI':
                    v3_ai += 1
                elif pred == 'Mixed':
                    v3_mixed += 1

        ai_pct = round(v3_ai / v3_n * 100, 2) if v3_n > 0 else 0
        combined_pct = round((v3_ai + v3_mixed) / v3_n * 100, 2) if v3_n > 0 else 0
        results.append({
            'month': month,
            'ai_generated_pct': ai_pct,
            'ai_combined_pct': combined_pct,
        })
        print(f"  {month}: AI={ai_pct:.1f}%, Combined={combined_pct:.1f}% (n={v3_n})")

    return results


def compute_hypothesis_signals(monthly_rows, annotations, html_dir):
    """Compute all 6 hypothesis signals per month."""

    # Lazy-load ML models
    print("\nLoading ML models...")
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentiment_pipe = pipeline(
        'sentiment-analysis',
        model='cardiffnlp/twitter-roberta-base-sentiment-latest',
        device='cpu',
        truncation=True,
        max_length=512,
    )
    print("  Models loaded!")

    # Precompute H2 error rates from annotations
    monthly_errors = defaultdict(lambda: {'total': 0, 'refuted': 0})
    for ann in annotations:
        ym = ann.get('year_month', '')
        if not ym:
            continue
        monthly_errors[ym]['total'] += 1
        if ann.get('verdict', '').strip() == 'refuted':
            monthly_errors[ym]['refuted'] += 1

    # HTML parsing patterns
    link_pattern = re.compile(r'<a\s[^>]*href\s*=', re.IGNORECASE)
    tag_pattern = re.compile(r'<[^>]+>')

    def char_trigrams(text):
        return set(text[i:i+3] for i in range(len(text) - 2)) if len(text) >= 3 else set()

    def jaccard(s1, s2):
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    results = []
    sorted_months = sorted(monthly_rows.keys())

    for mi, month in enumerate(sorted_months):
        print(f"  Processing {month} ({mi+1}/{len(sorted_months)})...")
        items = monthly_rows[month]
        indices = [i for i, _ in items]
        rows_m = [r for _, r in items]

        # AI likelihood (v2 pangram_avg mean)
        pangram_vals = []
        for r in rows_m:
            v = r.get('score_pangram_avg', '')
            if v and v.strip():
                try:
                    pangram_vals.append(float(v))
                except ValueError:
                    pass
        ai_likelihood = sum(pangram_vals) / len(pangram_vals) if pangram_vals else 0

        # Get texts for signal computation
        texts = [r['text'] for r in rows_m if r.get('text', '').strip() and len(r['text']) > 50]

        # ---- H1: Avg pairwise cosine similarity of semantic embeddings ----
        h1 = None
        sample = texts[:100]
        if len(sample) >= 2:
            embs = embed_model.encode(sample)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normed = embs / norms
            sim = normed @ normed.T
            n = len(sample)
            total = sum(float(sim[i][j]) for i in range(n) for j in range(i + 1, n))
            count = n * (n - 1) / 2
            h1 = total / count

        # ---- H2: Factual error rate ----
        h2 = None
        if month in monthly_errors and monthly_errors[month]['total'] > 0:
            h2 = monthly_errors[month]['refuted'] / monthly_errors[month]['total']

        # ---- H3: Positive sentiment rate ----
        h3 = None
        sent_sample = texts[:200]
        if sent_sample:
            try:
                sents = sentiment_pipe(sent_sample, batch_size=32)
                h3 = sum(1 for s in sents if s['label'].lower() == 'positive') / len(sents)
            except Exception as e:
                print(f"    Sentiment error: {e}")

        # ---- H4: Outbound link density (per 1k words) ----
        h4 = None
        if html_dir and os.path.isdir(html_dir):
            total_links = 0
            total_words = 0
            for idx in indices:
                fpath = os.path.join(html_dir, f"{idx}.html")
                if os.path.exists(fpath):
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            html = f.read()
                        total_links += len(link_pattern.findall(html))
                        total_words += len(tag_pattern.sub(' ', html).split())
                    except Exception:
                        pass
            if total_words > 0:
                h4 = (total_links / total_words) * 1000

        # ---- H5: Gzip compression ratio ----
        h5 = None
        comps = []
        for t in texts:
            raw = t.encode('utf-8')
            if len(raw) > 0:
                comps.append(len(gzip.compress(raw)) / len(raw))
        if comps:
            h5 = sum(comps) / len(comps)

        # ---- H6: Avg pairwise char 3-gram Jaccard similarity ----
        h6 = None
        tsets = [char_trigrams(t) for t in texts[:30] if len(t) >= 10]
        if len(tsets) >= 2:
            n = min(len(tsets), 30)
            total_j = sum(jaccard(tsets[i], tsets[j]) for i in range(n) for j in range(i + 1, n))
            count = n * (n - 1) / 2
            h6 = total_j / count

        results.append({
            'month': month,
            'ai_likelihood': ai_likelihood,
            'h1_cosine_similarity': h1,
            'h2_error_rate': h2,
            'h3_positive_rate': h3,
            'h4_link_density': h4,
            'h5_compression_ratio': h5,
            'h6_jaccard_similarity': h6,
        })

    return results


def write_prevalence_csv(results, output_path):
    """Write prevalence data to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['month', 'ai_generated_pct', 'ai_combined_pct'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {output_path}")


def write_hypotheses_csv(results, output_path):
    """Write hypothesis signals to CSV."""
    fields = ['month', 'ai_likelihood', 'h1_cosine_similarity', 'h2_error_rate',
              'h3_positive_rate', 'h4_link_density', 'h5_compression_ratio', 'h6_jaccard_similarity']
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for r in results:
            row = []
            for field in fields:
                val = r[field]
                if val is None:
                    row.append('')
                elif isinstance(val, float):
                    row.append(f"{val:.6f}")
                else:
                    row.append(str(val))
            writer.writerow(row)
    print(f"Wrote {output_path}")


def main():
    repo_root = Path('.')

    # Check required files
    internet_csv = repo_root / 'data' / 'internet.csv'
    annotations_csv = repo_root / 'factuality_analysis' / 'annotations.csv'
    html_dir = repo_root / 'data' / 'html_internet'

    if not internet_csv.exists():
        print(f"ERROR: {internet_csv} not found. Run this script from the ai-prevalence repo root.")
        sys.exit(1)

    # Check if it's an LFS pointer
    with open(internet_csv, 'r') as f:
        first_line = f.readline()
    if 'git-lfs' in first_line:
        print("ERROR: internet.csv is a Git LFS pointer. Run 'git lfs pull' first.")
        sys.exit(1)

    output_dir = repo_root / 'output'
    output_dir.mkdir(exist_ok=True)

    # Load data
    rows = load_internet_csv(internet_csv)
    monthly_rows = group_by_month(rows)

    annotations = []
    if annotations_csv.exists():
        annotations = load_annotations(annotations_csv)
    else:
        print(f"WARNING: {annotations_csv} not found. H2 error rates will be empty.")

    # Compute prevalence
    prevalence = compute_prevalence(monthly_rows)
    write_prevalence_csv(prevalence, output_dir / 'prevalence.csv')

    # Compute hypothesis signals
    html_path = str(html_dir) if html_dir.exists() else None
    if not html_path:
        print(f"WARNING: {html_dir} not found. H4 link density will be empty.")

    hypothesis_data = compute_hypothesis_signals(monthly_rows, annotations, html_path)
    write_hypotheses_csv(hypothesis_data, output_dir / 'hypotheses.csv')

    print(f"\n{'='*60}")
    print(f"Done! Output files:")
    print(f"  {output_dir / 'prevalence.csv'}")
    print(f"  {output_dir / 'hypotheses.csv'}")
    print(f"\nCopy these to your website repo's data/ directory.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
