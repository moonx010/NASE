#!/usr/bin/env python3
"""Per-degradation breakdown analysis.

Usage:
    python scripts/per_degradation_breakdown.py \
        --results_csv logs/eval_dereverb/multi_deg/_results.csv \
        --labels_csv /path/to/test_multi/labels.csv \
        [--guidance_csv logs/eval_ep159/multi_deg/_guidance_log.csv]

Computes PESQ/ESTOI/SI-SDR grouped by degradation type.
"""
import argparse
import pandas as pd
import sys


def classify_degradation(row):
    has_noise = row['noise_type'] != 'none' and row['snr'] > 0
    has_reverb = row['reverb_t60'] > 0
    has_distort = row['distort_intensity'] > 0

    parts = []
    if has_noise:
        parts.append('noise')
    if has_reverb:
        parts.append('reverb')
    if has_distort:
        parts.append('distort')

    if not parts:
        return 'clean'
    return '+'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_csv', required=True, help='Path to _results.csv from calc_metrics.py')
    parser.add_argument('--labels_csv', required=True, help='Path to labels.csv with degradation labels')
    parser.add_argument('--guidance_csv', default=None, help='Optional _guidance_log.csv for adaptive weights')
    args = parser.parse_args()

    # Load data
    results = pd.read_csv(args.results_csv)
    labels = pd.read_csv(args.labels_csv)

    # Classify degradation type
    labels['deg_type'] = labels.apply(classify_degradation, axis=1)

    # Merge on filename
    merged = results.merge(labels, on='filename', how='inner')
    if len(merged) == 0:
        print("ERROR: No matching filenames between results and labels!")
        sys.exit(1)

    print(f"Matched {len(merged)}/{len(results)} files\n")

    # Optional: merge guidance weights
    if args.guidance_csv:
        guidance = pd.read_csv(args.guidance_csv)
        merged = merged.merge(guidance, on='filename', how='left')

    # Per-degradation breakdown
    groups = merged.groupby('deg_type')

    print(f"{'Type':<25} {'N':>5} {'PESQ':>7} {'ESTOI':>7} {'SI-SDR':>8}", end='')
    if args.guidance_csv and 'noise_w' in merged.columns:
        print(f" {'noise_w':>8} {'reverb_w':>9} {'distort_w':>10}", end='')
    print()
    print('-' * 90)

    for name, group in sorted(groups, key=lambda x: -len(x[1])):
        n = len(group)
        pesq_mean = group['pesq'].mean()
        estoi_mean = group['estoi'].mean()
        sisdr_mean = group['si_sdr'].mean()

        print(f"{name:<25} {n:>5} {pesq_mean:>7.2f} {estoi_mean:>7.3f} {sisdr_mean:>8.1f}", end='')

        if args.guidance_csv and 'noise_w' in merged.columns:
            nw = group['noise_w'].mean()
            rw = group['reverb_w'].mean()
            dw = group['distort_w'].mean()
            print(f" {nw:>8.3f} {rw:>9.3f} {dw:>10.3f}", end='')
        print()

    # Overall
    print('-' * 90)
    n = len(merged)
    print(f"{'OVERALL':<25} {n:>5} {merged['pesq'].mean():>7.2f} {merged['estoi'].mean():>7.3f} {merged['si_sdr'].mean():>8.1f}")

    # Also save as CSV
    summary_rows = []
    for name, group in sorted(groups, key=lambda x: -len(x[1])):
        row = {
            'deg_type': name,
            'n': len(group),
            'pesq': round(group['pesq'].mean(), 2),
            'estoi': round(group['estoi'].mean(), 3),
            'si_sdr': round(group['si_sdr'].mean(), 1),
        }
        if args.guidance_csv and 'noise_w' in merged.columns:
            row['noise_w'] = round(group['noise_w'].mean(), 3)
            row['reverb_w'] = round(group['reverb_w'].mean(), 3)
            row['distort_w'] = round(group['distort_w'].mean(), 3)
        summary_rows.append(row)

    out_csv = args.results_csv.replace('_results.csv', '_per_degradation.csv')
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")


if __name__ == '__main__':
    main()
