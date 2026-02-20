#!/usr/bin/env python3
"""Quick test: compare old vs new noise_w computation on a few files.

Usage:
    python scripts/test_noise_w_fix.py \
        --ckpt /path/to/checkpoint.ckpt \
        --pretrain_class_model /path/to/WavLM-Large.pt \
        --test_dir /path/to/test_multi/noisy \
        --labels_csv /path/to/test_multi/labels.csv \
        --n_samples 50
"""
import argparse
import glob
import torch
import torch.nn.functional as F
from torchaudio import load
from sgmse.model import ScoreModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--pretrain_class_model', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--labels_csv', required=True)
    parser.add_argument('--n_samples', type=int, default=50)
    args = parser.parse_args()

    import pandas as pd
    labels = pd.read_csv(args.labels_csv)
    labels_dict = {r['filename']: r for _, r in labels.iterrows()}

    model = ScoreModel.load_from_checkpoint(
        args.ckpt, base_dir='', batch_size=16, num_workers=0,
        pretrain_class_model=args.pretrain_class_model,
        encoder_type='wavlm', multi_degradation=True,
        inject_method='temb', kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()

    files = sorted(glob.glob(args.test_dir + '/*.wav'))[:args.n_samples]

    print("{:<30} {:<12} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "filename", "noise_type", "old_nw", "new_nw", "p_none", "reverb_w", "distort_w"))
    print("-" * 95)

    for f in files:
        fname = f.split('/')[-1]
        y, _ = load(f)

        with torch.no_grad():
            _, noise_logits, _, reverb_pred, distort_pred = model.noise_encoder(y.cuda())

            # Old: max sigmoid
            old_nw = noise_logits.max(dim=-1)[0].item()

            # New: 1 - softmax P(none)
            raw = torch.logit(noise_logits.clamp(1e-6, 1 - 1e-6))
            probs = F.softmax(raw, dim=-1)
            p_none = probs[0, 10].item()
            new_nw = 1.0 - p_none

            reverb_w = reverb_pred.squeeze(-1).item()
            distort_w = distort_pred.squeeze(-1).item()

        label = labels_dict.get(fname, {})
        ntype = label.get('noise_type', '?') if isinstance(label, dict) else '?'

        print("{:<30} {:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            fname, str(ntype), old_nw, new_nw, p_none, reverb_w, distort_w))


if __name__ == '__main__':
    main()
