#!/usr/bin/env python3
"""Evaluate noise/reverb/distort head accuracy on test set.

Reports:
  - Noise head: top-1 accuracy, per-class accuracy, none vs non-none accuracy
  - Reverb head: MAE, correlation, detection accuracy (pred>0.1 vs T60>0)
  - Distort head: MAE, correlation, detection accuracy (pred>0.1 vs intensity>0)

Usage:
    python scripts/eval_head_accuracy.py \
        --ckpt logs/multi-deg-wavlm-v2/last.ckpt \
        --pretrain_class_model /path/to/WavLM-Large.pt \
        --test_dir /path/to/test_multi/noisy \
        --labels_csv /path/to/test_multi/labels.csv
"""
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchaudio import load
from collections import defaultdict

from sgmse.model import ScoreModel
from sgmse.data_module import noise2id_multi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--pretrain_class_model', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--labels_csv', required=True)
    args = parser.parse_args()

    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    labels_dict = {r['filename']: r for _, r in labels_df.iterrows()}

    # Reverse map: id -> noise_type name
    id2noise = {v: k for k, v in noise2id_multi.items()}

    # Load model
    model = ScoreModel.load_from_checkpoint(
        args.ckpt, base_dir='', batch_size=16, num_workers=0,
        pretrain_class_model=args.pretrain_class_model,
        encoder_type='wavlm', multi_degradation=True,
        inject_method='temb', kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()

    files = sorted(glob.glob(args.test_dir + '/*.wav'))
    print("Evaluating {} files...".format(len(files)))

    # Collect predictions
    results = []
    for f in files:
        fname = f.split('/')[-1]
        if fname not in labels_dict:
            continue

        label = labels_dict[fname]
        y, _ = load(f)

        with torch.no_grad():
            _, noise_logits, _, reverb_pred, distort_pred = model.noise_encoder(y.cuda())

            # Noise: convert sigmoid logits -> raw -> softmax -> argmax
            raw = torch.logit(noise_logits.clamp(1e-6, 1 - 1e-6))
            probs = F.softmax(raw, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            p_none = probs[0, 10].item()
            max_prob = probs[0, pred_class].item()

            # Reverb/distort: sigmoid outputs
            reverb_val = reverb_pred.squeeze(-1).item()
            distort_val = distort_pred.squeeze(-1).item()

        gt_noise_type = label['noise_type']
        gt_noise_id = noise2id_multi.get(gt_noise_type, 10)
        gt_reverb = float(label['reverb_t60'])
        gt_distort = float(label['distort_intensity'])

        results.append({
            'filename': fname,
            'gt_noise_type': gt_noise_type,
            'gt_noise_id': gt_noise_id,
            'pred_noise_id': pred_class,
            'pred_noise_type': id2noise.get(pred_class, 'unknown'),
            'noise_confidence': max_prob,
            'p_none': p_none,
            'gt_reverb': gt_reverb,
            'pred_reverb': reverb_val,
            'gt_distort': gt_distort,
            'pred_distort': distort_val,
        })

    df = pd.DataFrame(results)
    print("Matched {} files\n".format(len(df)))

    # ==================== NOISE HEAD ====================
    print("=" * 60)
    print("NOISE HEAD (11-class classification)")
    print("=" * 60)

    # Overall accuracy
    correct = (df['gt_noise_id'] == df['pred_noise_id']).sum()
    total = len(df)
    print("Overall accuracy: {}/{} = {:.1f}%".format(correct, total, 100 * correct / total))

    # None vs non-none binary accuracy
    df['gt_has_noise'] = df['gt_noise_type'] != 'none'
    df['pred_has_noise'] = df['pred_noise_id'] != 10  # 10 = none
    binary_correct = (df['gt_has_noise'] == df['pred_has_noise']).sum()
    print("Binary (noise/no-noise) accuracy: {}/{} = {:.1f}%".format(
        binary_correct, total, 100 * binary_correct / total))

    # Using p_none threshold for binary
    df['pred_has_noise_soft'] = df['p_none'] < 0.5
    soft_correct = (df['gt_has_noise'] == df['pred_has_noise_soft']).sum()
    print("Binary (p_none<0.5) accuracy: {}/{} = {:.1f}%".format(
        soft_correct, total, 100 * soft_correct / total))

    # Per-class accuracy
    print("\nPer-class accuracy:")
    print("{:<15} {:>6} {:>6} {:>8}".format("Class", "Total", "Correct", "Acc"))
    print("-" * 40)
    for noise_type in sorted(df['gt_noise_type'].unique()):
        subset = df[df['gt_noise_type'] == noise_type]
        gt_id = noise2id_multi.get(noise_type, 10)
        sub_correct = (subset['pred_noise_id'] == gt_id).sum()
        print("{:<15} {:>6} {:>6} {:>7.1f}%".format(
            noise_type, len(subset), sub_correct, 100 * sub_correct / len(subset)))

    # Confusion: what does the model predict for 'none' class?
    none_files = df[df['gt_noise_type'] == 'none']
    if len(none_files) > 0:
        print("\nNone-class prediction distribution:")
        pred_dist = none_files['pred_noise_type'].value_counts()
        for name, count in pred_dist.items():
            print("  predicted '{}': {} ({:.1f}%)".format(name, count, 100 * count / len(none_files)))

    # ==================== REVERB HEAD ====================
    print("\n" + "=" * 60)
    print("REVERB HEAD (T60 regression, sigmoid output)")
    print("=" * 60)

    mae_reverb = (df['gt_reverb'] - df['pred_reverb']).abs().mean()
    corr_reverb = df[['gt_reverb', 'pred_reverb']].corr().iloc[0, 1]
    print("MAE: {:.4f}".format(mae_reverb))
    print("Correlation: {:.4f}".format(corr_reverb))

    # Binary detection: reverb present or not
    df['gt_has_reverb'] = df['gt_reverb'] > 0
    for thresh in [0.05, 0.1, 0.2, 0.3]:
        df['pred_has_reverb'] = df['pred_reverb'] > thresh
        det_correct = (df['gt_has_reverb'] == df['pred_has_reverb']).sum()
        tp = ((df['gt_has_reverb']) & (df['pred_has_reverb'])).sum()
        fn = ((df['gt_has_reverb']) & (~df['pred_has_reverb'])).sum()
        fp = ((~df['gt_has_reverb']) & (df['pred_has_reverb'])).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print("Detection (thresh={:.2f}): acc={:.1f}%, precision={:.1f}%, recall={:.1f}%".format(
            thresh, 100 * det_correct / total, 100 * precision, 100 * recall))

    # ==================== DISTORT HEAD ====================
    print("\n" + "=" * 60)
    print("DISTORT HEAD (intensity regression, sigmoid output)")
    print("=" * 60)

    mae_distort = (df['gt_distort'] - df['pred_distort']).abs().mean()
    corr_distort = df[['gt_distort', 'pred_distort']].corr().iloc[0, 1]
    print("MAE: {:.4f}".format(mae_distort))
    print("Correlation: {:.4f}".format(corr_distort))

    # Binary detection
    df['gt_has_distort'] = df['gt_distort'] > 0
    for thresh in [0.05, 0.1, 0.2, 0.3]:
        df['pred_has_distort'] = df['pred_distort'] > thresh
        det_correct = (df['gt_has_distort'] == df['pred_has_distort']).sum()
        tp = ((df['gt_has_distort']) & (df['pred_has_distort'])).sum()
        fn = ((df['gt_has_distort']) & (~df['pred_has_distort'])).sum()
        fp = ((~df['gt_has_distort']) & (df['pred_has_distort'])).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print("Detection (thresh={:.2f}): acc={:.1f}%, precision={:.1f}%, recall={:.1f}%".format(
            thresh, 100 * det_correct / total, 100 * precision, 100 * recall))

    # Save detailed CSV
    out_csv = args.labels_csv.replace('labels.csv', 'head_predictions.csv')
    df.to_csv(out_csv, index=False)
    print("\nSaved predictions to {}".format(out_csv))


if __name__ == '__main__':
    main()
