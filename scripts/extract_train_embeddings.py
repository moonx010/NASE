"""Extract noise embeddings from training data and save as .pt files.

Outputs (for each mode: postcnn / raw):
  embeddings/{encoder_type}_train_all.pt       — (N_train, D) L2-normalized embeddings
  embeddings/{encoder_type}_train_prototypes.pt — (10, D) L2-normalized class prototypes

Usage:
  # post_cnn embeddings (256-dim, L2 normalized)
  python scripts/extract_train_embeddings.py \
      --ckpt logs/dx2ds38e/last.ckpt \
      --pretrain_class_model /path/to/BEATs_iter3_plus_AS2M.pt \
      --data_dir /path/to/voicebank-demand-16k \
      --encoder_type beats

  # Raw BEATs features (768-dim, L2 normalized)
  python scripts/extract_train_embeddings.py \
      --ckpt logs/dx2ds38e/last.ckpt \
      --pretrain_class_model /path/to/BEATs_iter3_plus_AS2M.pt \
      --data_dir /path/to/voicebank-demand-16k \
      --encoder_type beats --use_raw
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from os.path import join
from torchaudio import load

# Add project root to path when running from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sgmse.model import ScoreModel


noise2id = {
    'babble': 0, 'cafeteria': 1, 'car': 2, 'kitchen': 3, 'meeting': 4,
    'metro': 5, 'restaurant': 6, 'ssn': 7, 'station': 8, 'traffic': 9
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--pretrain_class_model", type=str, required=True, help="BEATs checkpoint path")
    parser.add_argument("--data_dir", type=str, required=True, help="VoiceBank-DEMAND 16kHz root (must have train/noisy/)")
    parser.add_argument("--encoder_type", type=str, default="beats", choices=("beats", "wavlm", "panns"))
    parser.add_argument("--output_dir", type=str, default="embeddings", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for extraction")
    parser.add_argument("--use_raw", action='store_true', help="Use raw encoder features (768-dim) instead of post_cnn (256-dim)")
    args = parser.parse_args()

    mode = "raw" if args.use_raw else "postcnn"
    print(f"Mode: {mode} ({'raw encoder features' if args.use_raw else 'post_cnn features'})")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    model = ScoreModel.load_from_checkpoint(
        args.ckpt,
        base_dir='',
        batch_size=16,
        num_workers=0,
        pretrain_class_model=args.pretrain_class_model,
        encoder_type=args.encoder_type,
        kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()

    # Load noise labels for prototypes
    label_file = join(args.data_dir, "train", "noise_label.txt")
    file_labels = {}
    if os.path.exists(label_file):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()[:2]
                file_labels[parts[0]] = noise2id[parts[1]]
        print(f"Loaded {len(file_labels)} noise labels from {label_file}")
    else:
        print(f"Warning: noise_label.txt not found at {label_file}, prototypes will not be computed")

    # Get training noisy files
    noisy_dir = join(args.data_dir, "train", "noisy")
    noisy_files = sorted(glob(join(noisy_dir, "*.wav")))
    print(f"Found {len(noisy_files)} training files in {noisy_dir}")

    all_embeddings = []
    all_labels = []

    # Process in batches
    for i in tqdm(range(0, len(noisy_files), args.batch_size), desc="Extracting embeddings"):
        batch_files = noisy_files[i:i + args.batch_size]

        # Load and pad to same length
        wavs = []
        labels = []
        for f in batch_files:
            wav, _ = load(f)  # (1, T)
            wavs.append(wav.squeeze(0))
            # Get label from filename
            fname = os.path.basename(f)
            fname_noext = os.path.splitext(fname)[0]
            if fname_noext in file_labels:
                labels.append(file_labels[fname_noext])
            elif fname in file_labels:
                labels.append(file_labels[fname])
            else:
                labels.append(-1)

        # Pad to max length in batch
        max_len = max(w.shape[-1] for w in wavs)
        padded = torch.zeros(len(wavs), max_len)
        for j, w in enumerate(wavs):
            padded[j, :w.shape[-1]] = w

        # Extract embeddings (already L2 normalized inside model)
        with torch.no_grad():
            emb = model.extract_noise_embedding(padded.cuda(), use_raw=args.use_raw)
        all_embeddings.append(emb.cpu())
        all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)  # (N_train, D)
    all_labels = torch.tensor(all_labels)

    # File suffix based on mode
    suffix = f"{args.encoder_type}_{mode}_train"

    # Save all embeddings
    out_all = join(args.output_dir, f"{suffix}_all.pt")
    torch.save(all_embeddings, out_all)
    print(f"Saved all embeddings: {out_all} — shape {all_embeddings.shape}")

    # Compute and save prototypes (class centroids, re-normalized)
    if file_labels:
        n_classes = max(noise2id.values()) + 1
        prototypes = torch.zeros(n_classes, all_embeddings.shape[1])
        counts = torch.zeros(n_classes)
        for idx in range(len(all_labels)):
            label = all_labels[idx].item()
            if label >= 0:
                prototypes[label] += all_embeddings[idx]
                counts[label] += 1
        # Average and re-normalize
        for c in range(n_classes):
            if counts[c] > 0:
                prototypes[c] /= counts[c]
            print(f"  Class {c} ({[k for k, v in noise2id.items() if v == c][0]}): {int(counts[c].item())} samples")
        # L2 normalize prototypes
        prototypes = F.normalize(prototypes, p=2, dim=-1)

        out_proto = join(args.output_dir, f"{suffix}_prototypes.pt")
        torch.save(prototypes, out_proto)
        print(f"Saved prototypes: {out_proto} — shape {prototypes.shape}")

    # Print stats
    norms = all_embeddings.norm(dim=-1)
    print(f"\nEmbedding stats (should be ~1.0 after L2 norm):")
    print(f"  norm: mean={norms.mean():.4f}, std={norms.std():.4f}, min={norms.min():.4f}, max={norms.max():.4f}")

    # Pairwise distance stats (sample 500 pairs)
    n = min(500, len(all_embeddings))
    sample = all_embeddings[:n]
    pw_dist = torch.cdist(sample, sample)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    dists = pw_dist[mask]
    print(f"  pairwise dist (sample {n}): mean={dists.mean():.4f}, std={dists.std():.4f}, min={dists.min():.4f}, max={dists.max():.4f}")


if __name__ == "__main__":
    main()
