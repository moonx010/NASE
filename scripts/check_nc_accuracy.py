"""Check NC classification accuracy on training data.

Usage:
    python scripts/check_nc_accuracy.py --ckpt <checkpoint> --data_dir <data_dir> \
        --pretrain_class_model <beats_ckpt> [--encoder_type beats] [--max_samples 500]
"""
import argparse
import torch
from glob import glob
from os.path import join
from torchaudio import load
from sgmse.model import ScoreModel

noise2id = {'babble': 0, 'cafeteria': 1, 'car': 2, 'kitchen': 3, 'meeting': 4,
             'metro': 5, 'restaurant': 6, 'ssn': 7, 'station': 8, 'traffic': 9}
id2noise = {v: k for k, v in noise2id.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrain_class_model", type=str, required=True)
    parser.add_argument("--encoder_type", type=str, default="beats")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ScoreModel.load_from_checkpoint(
        args.ckpt, pretrain_class_model=args.pretrain_class_model,
        encoder_type=args.encoder_type, map_location=device
    )
    model.eval().to(device)

    # Load noise labels
    label_file = join(args.data_dir, "train", "noise_label.txt")
    noise_labels = []
    with open(label_file) as f:
        for line in f.readlines():
            parts = line.strip().split()[:2]
            noise_labels.append(parts)
    noise_labels = sorted(noise_labels, key=lambda x: x[0])

    # Load noisy files
    noisy_files = sorted(glob(join(args.data_dir, "train", "noisy", "*.wav")))

    # Sample subset
    n = min(args.max_samples, len(noisy_files))
    indices = torch.randperm(len(noisy_files))[:n].sort().values.tolist()

    correct = 0
    total = 0
    per_class_correct = {k: 0 for k in noise2id}
    per_class_total = {k: 0 for k in noise2id}
    confidences = []

    print(f"Checking NC accuracy on {n} samples...")
    with torch.no_grad():
        for idx in indices:
            y_wav, sr = load(noisy_files[idx])
            y_wav = y_wav.to(device)

            # Get label
            fname = noise_labels[idx][0]
            label_str = noise_labels[idx][1]
            label = noise2id[label_str]

            # Forward through noise encoder
            noise_emb, nc_logits, _ = model.noise_encoder(y_wav)

            # BEATs returns (B, 10) already mean-pooled + sigmoid'd
            # Use sigmoid values directly (do NOT apply softmax on top)
            pred = torch.argmax(nc_logits, dim=-1).item()
            conf = nc_logits.max(dim=-1)[0].item()

            confidences.append(conf)
            per_class_total[label_str] += 1
            if pred == label:
                correct += 1
                per_class_correct[label_str] += 1
            total += 1

    print(f"\n=== NC Classification Results ===")
    print(f"Overall Accuracy: {correct}/{total} = {correct/total:.3f}")
    print(f"Mean Confidence: {sum(confidences)/len(confidences):.3f}")
    print(f"Min/Max Confidence: {min(confidences):.3f} / {max(confidences):.3f}")
    print(f"\nPer-class accuracy:")
    for cls in sorted(noise2id.keys()):
        t = per_class_total[cls]
        c = per_class_correct[cls]
        if t > 0:
            print(f"  {cls:12s}: {c}/{t} = {c/t:.3f}")
        else:
            print(f"  {cls:12s}: (no samples)")

if __name__ == "__main__":
    main()
