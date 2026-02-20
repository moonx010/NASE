#!/usr/bin/env python3
"""Analyze adaptive weights (noise_w, reverb_w, distort_w) by degradation type.

Usage:
    python scripts/analyze_noise_w.py \
        --guidance_csv logs/eval_ep159/multi_deg/_guidance_log.csv \
        --labels_csv /path/to/test_multi/labels.csv
"""
import argparse
import pandas as pd


def classify(row):
    parts = []
    if row['noise_type'] != 'none' and row['snr'] > 0:
        parts.append('noise')
    if row['reverb_t60'] > 0:
        parts.append('reverb')
    if row['distort_intensity'] > 0:
        parts.append('distort')
    return '+'.join(parts) if parts else 'clean'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--guidance_csv', required=True)
    parser.add_argument('--labels_csv', required=True)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels_csv)
    guidance = pd.read_csv(args.guidance_csv)
    merged = labels.merge(guidance, on='filename', how='inner')
    merged['deg_type'] = merged.apply(classify, axis=1)

    print("=== Adaptive weights by degradation type ===")
    header = "{:<25} {:>5} {:>10} {:>10} {:>11}".format("Type", "N", "noise_w", "reverb_w", "distort_w")
    print(header)
    print("-" * 65)

    for name, group in sorted(merged.groupby('deg_type'), key=lambda x: -len(x[1])):
        line = "{:<25} {:>5} {:>10.4f} {:>10.4f} {:>11.4f}".format(
            name, len(group),
            group['noise_w'].mean(), group['reverb_w'].mean(), group['distort_w'].mean())
        print(line)

    print()
    print("=== noise_type=none (no noise) files ===")
    no_noise = merged[merged['noise_type'] == 'none']
    print("Total files with noise_type=none: {}".format(len(no_noise)))
    print("  noise_w:   mean={:.4f}, min={:.4f}, max={:.4f}".format(
        no_noise['noise_w'].mean(), no_noise['noise_w'].min(), no_noise['noise_w'].max()))
    print("  reverb_w:  mean={:.4f}, min={:.4f}, max={:.4f}".format(
        no_noise['reverb_w'].mean(), no_noise['reverb_w'].min(), no_noise['reverb_w'].max()))
    print("  distort_w: mean={:.4f}, min={:.4f}, max={:.4f}".format(
        no_noise['distort_w'].mean(), no_noise['distort_w'].min(), no_noise['distort_w'].max()))

    print()
    print("=== Breakdown of noise_type=none by deg_type ===")
    for name, group in sorted(no_noise.groupby('deg_type'), key=lambda x: -len(x[1])):
        print("  {:<20} N={:>4}, noise_w={:.4f}, reverb_w={:.4f}, distort_w={:.4f}".format(
            name, len(group), group['noise_w'].mean(), group['reverb_w'].mean(), group['distort_w'].mean()))

    print()
    print("=== Noise head discrimination ===")
    has_noise = merged[merged['noise_type'] != 'none']
    print("Noisy files avg noise_w:     {:.4f} (N={})".format(has_noise['noise_w'].mean(), len(has_noise)))
    print("Non-noisy files avg noise_w: {:.4f} (N={})".format(no_noise['noise_w'].mean(), len(no_noise)))
    print("Gap: {:.4f}".format(has_noise['noise_w'].mean() - no_noise['noise_w'].mean()))

    print()
    print("=== Reverb head discrimination ===")
    has_reverb = merged[merged['reverb_t60'] > 0]
    no_reverb = merged[merged['reverb_t60'] == 0]
    print("Reverb files avg reverb_w:     {:.4f} (N={})".format(has_reverb['reverb_w'].mean(), len(has_reverb)))
    print("Non-reverb files avg reverb_w: {:.4f} (N={})".format(no_reverb['reverb_w'].mean(), len(no_reverb)))
    print("Gap: {:.4f}".format(has_reverb['reverb_w'].mean() - no_reverb['reverb_w'].mean()))

    print()
    print("=== Distort head discrimination ===")
    has_distort = merged[merged['distort_intensity'] > 0]
    no_distort = merged[merged['distort_intensity'] == 0]
    print("Distort files avg distort_w:     {:.4f} (N={})".format(has_distort['distort_w'].mean(), len(has_distort)))
    print("Non-distort files avg distort_w: {:.4f} (N={})".format(no_distort['distort_w'].mean(), len(no_distort)))
    print("Gap: {:.4f}".format(has_distort['distort_w'].mean() - no_distort['distort_w'].mean()))

    # Sample noise_type=none files
    print()
    print("=== Sample noise_type=none files (first 20) ===")
    cols = ['filename', 'noise_type', 'reverb_t60', 'distort_intensity', 'noise_w', 'reverb_w', 'distort_w']
    print(no_noise[cols].head(20).to_string(index=False))


if __name__ == '__main__':
    main()
