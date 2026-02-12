"""EXP-3: Distance Distribution Analysis — Figure 1 for paper.

Reads guidance_log.csv from kNN adaptive runs and generates 4-panel figure:
(a) Histogram: in-dist vs OOD embedding distance
(b) Histogram: in-dist vs OOD adaptive w
(c) Box plot: per-noise-category distance
(d) Scatter: distance vs PESQ improvement (adaptive - baseline)

Usage:
    python scripts/plot_distance_analysis.py \
        --knn_in_dist enhanced/e3-dist-knn/in_dist \
        --knn_ood enhanced/e3-dist-knn/ood_esc50_all \
        --baseline_in_dist enhanced/e1-full-baseline/in_dist \
        --baseline_ood enhanced/e1-full-baseline/ood_esc50_all \
        --output figures/distance_analysis.pdf
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np

# Use non-interactive backend for server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Known noise categories
INDIST_NOISES = ['babble', 'cafeteria', 'car', 'kitchen', 'meeting',
                 'metro', 'restaurant', 'ssn', 'station', 'traffic']


def parse_noise_type(filename):
    """Extract noise type from VBD or ESC-50 filename."""
    # VBD format: p232_001_babble_snr5.wav or similar
    # ESC-50 OOD format: <esc_class>_<clean_file>.wav
    name = os.path.splitext(filename)[0]

    # Try to match VBD in-dist noise types
    for noise in INDIST_NOISES:
        if noise in name.lower():
            return noise

    # ESC-50: first part before underscore is typically the noise class
    # But filenames may vary. Try first segment
    parts = name.split('_')
    if len(parts) >= 2:
        # ESC-50 format: <noise_type>_<digits>_<clean_info>.wav
        return parts[0]

    return 'unknown'


def load_guidance_log(enhanced_dir):
    """Load _guidance_log.csv from enhanced directory."""
    path = os.path.join(enhanced_dir, '_guidance_log.csv')
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    return pd.read_csv(path)


def load_results(enhanced_dir):
    """Load _results.csv (per-file metrics) from enhanced directory."""
    path = os.path.join(enhanced_dir, '_results.csv')
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn_in_dist", type=str, required=True)
    parser.add_argument("--knn_ood", type=str, required=True)
    parser.add_argument("--baseline_in_dist", type=str, default=None,
                        help="Baseline enhanced dir (for PESQ delta). Optional.")
    parser.add_argument("--baseline_ood", type=str, default=None,
                        help="Baseline OOD enhanced dir (for PESQ delta). Optional.")
    parser.add_argument("--output", type=str, default="figures/distance_analysis.pdf")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load data
    log_in = load_guidance_log(args.knn_in_dist)
    log_ood = load_guidance_log(args.knn_ood)

    if log_in is None or log_ood is None:
        print("ERROR: Cannot find guidance_log.csv files. Run kNN adaptive eval first.")
        sys.exit(1)

    # ---- Figure ----
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # (a) Distance histogram
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, max(log_in['distance'].max(), log_ood['distance'].max()) * 1.1, 40)
    ax1.hist(log_in['distance'], bins=bins, alpha=0.6, label='In-dist', color='#2196F3', density=True)
    ax1.hist(log_ood['distance'], bins=bins, alpha=0.6, label='OOD (ESC-50)', color='#F44336', density=True)
    ax1.set_xlabel('k-NN Embedding Distance')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Distance Distribution')
    ax1.legend()
    # Add mean lines
    ax1.axvline(log_in['distance'].mean(), color='#1565C0', linestyle='--', linewidth=1.5)
    ax1.axvline(log_ood['distance'].mean(), color='#C62828', linestyle='--', linewidth=1.5)

    # (b) w histogram
    ax2 = fig.add_subplot(gs[0, 1])
    bins_w = np.linspace(0, 1.05, 30)
    ax2.hist(log_in['w'], bins=bins_w, alpha=0.6, label='In-dist', color='#2196F3', density=True)
    ax2.hist(log_ood['w'], bins=bins_w, alpha=0.6, label='OOD (ESC-50)', color='#F44336', density=True)
    ax2.set_xlabel('Adaptive Guidance Scale w')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) Guidance Scale Distribution')
    ax2.legend()
    ax2.axvline(log_in['w'].mean(), color='#1565C0', linestyle='--', linewidth=1.5)
    ax2.axvline(log_ood['w'].mean(), color='#C62828', linestyle='--', linewidth=1.5)

    # (c) Box plot per noise category
    ax3 = fig.add_subplot(gs[1, 0])
    log_in_copy = log_in.copy()
    log_ood_copy = log_ood.copy()
    log_in_copy['noise_type'] = log_in_copy['filename'].apply(parse_noise_type)
    log_ood_copy['noise_type'] = log_ood_copy['filename'].apply(parse_noise_type)
    log_in_copy['source'] = 'in-dist'
    log_ood_copy['source'] = 'OOD'

    combined = pd.concat([log_in_copy, log_ood_copy], ignore_index=True)
    # Group by noise type, sort by mean distance
    type_means = combined.groupby('noise_type')['distance'].mean().sort_values()
    ordered_types = type_means.index.tolist()

    # Only show top N types if too many
    if len(ordered_types) > 20:
        # Keep all in-dist + top 10 OOD
        indist_types = [t for t in ordered_types if t in INDIST_NOISES]
        ood_types = [t for t in ordered_types if t not in INDIST_NOISES]
        ordered_types = indist_types + ood_types[-10:]

    box_data = [combined[combined['noise_type'] == t]['distance'].values for t in ordered_types]
    colors = ['#2196F3' if t in INDIST_NOISES else '#F44336' for t in ordered_types]

    bp = ax3.boxplot(box_data, labels=ordered_types, patch_artist=True, vert=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_xlabel('Noise Type')
    ax3.set_ylabel('k-NN Distance')
    ax3.set_title('(c) Distance by Noise Category')
    ax3.tick_params(axis='x', rotation=45)
    # Custom legend
    from matplotlib.patches import Patch
    ax3.legend(handles=[Patch(facecolor='#2196F3', alpha=0.6, label='In-dist'),
                        Patch(facecolor='#F44336', alpha=0.6, label='OOD')],
               loc='upper left')

    # (d) Distance vs PESQ improvement
    ax4 = fig.add_subplot(gs[1, 1])
    has_baseline = False

    res_adaptive_in = load_results(args.knn_in_dist)
    res_adaptive_ood = load_results(args.knn_ood)

    if args.baseline_in_dist and args.baseline_ood:
        res_base_in = load_results(args.baseline_in_dist)
        res_base_ood = load_results(args.baseline_ood)
        if all(r is not None for r in [res_adaptive_in, res_adaptive_ood, res_base_in, res_base_ood]):
            has_baseline = True

    if has_baseline:
        # Merge on filename
        merged_in = pd.merge(log_in, res_adaptive_in[['filename', 'pesq']], on='filename', suffixes=('', '_adapt'))
        merged_in = pd.merge(merged_in, res_base_in[['filename', 'pesq']], on='filename', suffixes=('', '_base'))
        merged_in['pesq_delta'] = merged_in['pesq'] - merged_in['pesq_base'] if 'pesq_base' in merged_in.columns else merged_in['pesq_adapt'] - merged_in['pesq']

        merged_ood = pd.merge(log_ood, res_adaptive_ood[['filename', 'pesq']], on='filename', suffixes=('', '_adapt'))
        merged_ood = pd.merge(merged_ood, res_base_ood[['filename', 'pesq']], on='filename', suffixes=('', '_base'))
        merged_ood['pesq_delta'] = merged_ood['pesq'] - merged_ood['pesq_base'] if 'pesq_base' in merged_ood.columns else merged_ood['pesq_adapt'] - merged_ood['pesq']

        ax4.scatter(merged_in['distance'], merged_in['pesq_delta'], alpha=0.4, s=15, c='#2196F3', label='In-dist')
        ax4.scatter(merged_ood['distance'], merged_ood['pesq_delta'], alpha=0.4, s=15, c='#F44336', label='OOD')
        ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax4.set_xlabel('k-NN Embedding Distance')
        ax4.set_ylabel('PESQ (Adaptive - Baseline)')
        ax4.set_title('(d) Distance vs PESQ Improvement')
        ax4.legend()
    else:
        # Fallback: just show distance vs adaptive PESQ
        if res_adaptive_in is not None and res_adaptive_ood is not None:
            merged_in = pd.merge(log_in, res_adaptive_in[['filename', 'pesq']], on='filename')
            merged_ood = pd.merge(log_ood, res_adaptive_ood[['filename', 'pesq']], on='filename')
            ax4.scatter(merged_in['distance'], merged_in['pesq'], alpha=0.4, s=15, c='#2196F3', label='In-dist')
            ax4.scatter(merged_ood['distance'], merged_ood['pesq'], alpha=0.4, s=15, c='#F44336', label='OOD')
            ax4.set_xlabel('k-NN Embedding Distance')
            ax4.set_ylabel('PESQ')
            ax4.set_title('(d) Distance vs PESQ')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No _results.csv available\nRun calc_metrics.py first',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('(d) Distance vs PESQ Improvement')

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {args.output}")

    # Print stats summary
    print(f"\n=== Distance Stats ===")
    print(f"In-dist: mean={log_in['distance'].mean():.3f}, std={log_in['distance'].std():.3f}, "
          f"min={log_in['distance'].min():.3f}, max={log_in['distance'].max():.3f}")
    print(f"OOD:     mean={log_ood['distance'].mean():.3f}, std={log_ood['distance'].std():.3f}, "
          f"min={log_ood['distance'].min():.3f}, max={log_ood['distance'].max():.3f}")
    print(f"\n=== Guidance Scale (w) Stats ===")
    print(f"In-dist: mean={log_in['w'].mean():.3f}, std={log_in['w'].std():.3f}")
    print(f"OOD:     mean={log_ood['w'].mean():.3f}, std={log_ood['w'].std():.3f}")
    print(f"Separation (mean_w_in - mean_w_ood): {log_in['w'].mean() - log_ood['w'].mean():.3f}")


if __name__ == "__main__":
    main()
