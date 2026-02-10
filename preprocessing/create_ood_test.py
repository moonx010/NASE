"""
OOD Test Mixture Generation Script.

Mixes clean speech with out-of-distribution noise at specified SNR levels.
Supports ESC-50 and other noise sources with category-level organization.

Usage:
    python preprocessing/create_ood_test.py \
        --clean_dir /path/to/voicebank-demand-16k/test/clean \
        --noise_dir /path/to/ESC-50-master/audio \
        --output_dir /path/to/ood_test/esc50 \
        --esc50_meta /path/to/ESC-50-master/meta/esc50.csv \
        --snr 0 --target_sr 16000

Output structure:
    output_dir/
    ├── all/
    │   ├── clean/   (symlinks or copies)
    │   └── noisy/   (mixed files)
    ├── stationary/
    │   ├── clean/
    │   └── noisy/
    ├── non_stationary/
    │   ├── clean/
    │   └── noisy/
    └── metadata.csv  (per-file noise info)
"""

import os
import argparse
import random
import csv
from glob import glob

import torch
import torchaudio
import numpy as np


# ESC-50 category to noise type mapping
ESC50_STATIONARY = {
    'rain', 'wind', 'water_drops', 'sea_waves', 'crackling_fire',
    'insects', 'engine', 'train', 'helicopter', 'chainsaw',
    'airplane', 'vacuum_cleaner', 'washing_machine',
}

ESC50_NON_STATIONARY = {
    'dog', 'cat', 'rooster', 'pig', 'cow', 'frog', 'hen', 'crow',
    'thunderstorm', 'door_wood_knock', 'mouse_click', 'keyboard_typing',
    'door_wood_creaks', 'can_opening', 'clock_alarm', 'clock_tick',
    'glass_breaking', 'siren', 'car_horn', 'hand_saw',
    'church_bells', 'fireworks', 'snoring', 'toilet_flush',
    'laughing', 'sneezing', 'coughing', 'breathing', 'crying_baby',
    'footsteps', 'drinking_sipping', 'brushing_teeth',
    'clapping', 'pouring_water',
}


def load_esc50_metadata(csv_path):
    """Load ESC-50 metadata CSV and return dict: filename -> category."""
    meta = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta[row['filename']] = row['category']
    return meta


def mix_at_snr(clean, noise, snr_db):
    """Mix clean and noise signals at specified SNR (dB)."""
    # Repeat or trim noise to match clean length
    clean_len = clean.shape[-1]
    noise_len = noise.shape[-1]

    if noise_len < clean_len:
        repeats = (clean_len // noise_len) + 1
        noise = noise.repeat(1, repeats)
    noise = noise[:, :clean_len]

    # Compute scaling factor for target SNR
    clean_power = (clean ** 2).mean()
    noise_power = (noise ** 2).mean()

    if noise_power == 0:
        return clean

    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(clean_power / (snr_linear * noise_power))
    noisy = clean + scale * noise
    return noisy


def main():
    parser = argparse.ArgumentParser(description="Create OOD test mixtures")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory with clean speech wav files")
    parser.add_argument("--noise_dir", type=str, required=True, help="Directory with noise wav files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for mixtures")
    parser.add_argument("--esc50_meta", type=str, default=None, help="Path to ESC-50 meta/esc50.csv (for category info)")
    parser.add_argument("--snr", type=float, default=0.0, help="Target SNR in dB (default: 0)")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--max_files", type=int, default=200, help="Max number of test files to create")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load clean files
    clean_files = sorted(glob(os.path.join(args.clean_dir, "*.wav")))
    if not clean_files:
        print(f"No clean files found in {args.clean_dir}")
        return

    # Load noise files
    noise_files = sorted(glob(os.path.join(args.noise_dir, "*.wav")))
    if not noise_files:
        # Try subdirectories (ESC-50 might have flat or nested structure)
        noise_files = sorted(glob(os.path.join(args.noise_dir, "**/*.wav"), recursive=True))
    if not noise_files:
        print(f"No noise files found in {args.noise_dir}")
        return

    print(f"Clean files: {len(clean_files)}")
    print(f"Noise files: {len(noise_files)}")

    # Load ESC-50 metadata if provided
    esc50_meta = None
    if args.esc50_meta and os.path.exists(args.esc50_meta):
        esc50_meta = load_esc50_metadata(args.esc50_meta)
        print(f"ESC-50 categories loaded: {len(set(esc50_meta.values()))} categories")

    # Limit files
    clean_files = clean_files[:args.max_files]
    random.shuffle(noise_files)

    # Create output dirs
    for subdir in ['all/clean', 'all/noisy', 'stationary/clean', 'stationary/noisy',
                   'non_stationary/clean', 'non_stationary/noisy']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    resampler = None
    metadata_rows = []

    for i, clean_path in enumerate(clean_files):
        noise_path = noise_files[i % len(noise_files)]
        noise_filename = os.path.basename(noise_path)
        clean_filename = os.path.basename(clean_path)

        # Determine noise category
        category = "unknown"
        noise_type = "unknown"
        if esc50_meta and noise_filename in esc50_meta:
            category = esc50_meta[noise_filename]
            if category in ESC50_STATIONARY:
                noise_type = "stationary"
            elif category in ESC50_NON_STATIONARY:
                noise_type = "non_stationary"
            else:
                noise_type = "other"

        # Load audio
        clean_wav, clean_sr = torchaudio.load(clean_path)
        noise_wav, noise_sr = torchaudio.load(noise_path)

        # Resample if needed
        if clean_sr != args.target_sr:
            clean_wav = torchaudio.transforms.Resample(clean_sr, args.target_sr)(clean_wav)
        if noise_sr != args.target_sr:
            noise_wav = torchaudio.transforms.Resample(noise_sr, args.target_sr)(noise_wav)

        # Convert to mono if needed
        if clean_wav.shape[0] > 1:
            clean_wav = clean_wav.mean(dim=0, keepdim=True)
        if noise_wav.shape[0] > 1:
            noise_wav = noise_wav.mean(dim=0, keepdim=True)

        # Mix
        noisy_wav = mix_at_snr(clean_wav, noise_wav, args.snr)

        # Save to all/
        torchaudio.save(os.path.join(args.output_dir, 'all', 'clean', clean_filename), clean_wav, args.target_sr)
        torchaudio.save(os.path.join(args.output_dir, 'all', 'noisy', clean_filename), noisy_wav, args.target_sr)

        # Save to category subdir
        if noise_type in ('stationary', 'non_stationary'):
            torchaudio.save(os.path.join(args.output_dir, noise_type, 'clean', clean_filename), clean_wav, args.target_sr)
            torchaudio.save(os.path.join(args.output_dir, noise_type, 'noisy', clean_filename), noisy_wav, args.target_sr)

        metadata_rows.append({
            'clean_file': clean_filename,
            'noise_file': noise_filename,
            'noise_category': category,
            'noise_type': noise_type,
            'snr_db': args.snr,
        })

    # Save metadata
    meta_path = os.path.join(args.output_dir, 'metadata.csv')
    with open(meta_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['clean_file', 'noise_file', 'noise_category', 'noise_type', 'snr_db'])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nDone! Created {len(metadata_rows)} mixtures at SNR={args.snr}dB")
    print(f"  all: {len(metadata_rows)} files")
    stat_count = sum(1 for r in metadata_rows if r['noise_type'] == 'stationary')
    nonstat_count = sum(1 for r in metadata_rows if r['noise_type'] == 'non_stationary')
    print(f"  stationary: {stat_count} files")
    print(f"  non_stationary: {nonstat_count} files")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
