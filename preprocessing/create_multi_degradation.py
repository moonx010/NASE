"""
Multi-Degradation Data Generation Script.

Creates training data with combinations of:
- Additive noise (from DEMAND, 10 types)
- Reverberation (pyroomacoustics, T60 0.3-1.0)
- Distortion (clipping-based, intensity 0.0-1.0)

Each clean file keeps its original noise-only version and gets 1-2 additional
degradation combinations randomly.

Usage:
    python preprocessing/create_multi_degradation.py \
        --clean_dir /path/to/voicebank-demand-16k/train/clean \
        --noisy_dir /path/to/voicebank-demand-16k/train/noisy \
        --noise_label /path/to/voicebank-demand-16k/train/noise_label.txt \
        --output_dir /path/to/multi_degradation_16k/train \
        --seed 42

Output:
    output_dir/
    ├── clean/       # clean files (originals + copies for each degraded variant)
    ├── noisy/       # degraded files
    └── labels.csv   # multi-label metadata
"""

import os
import argparse
import random
import csv

import numpy as np
import soundfile as sf
import torch
import torchaudio
from glob import glob
from tqdm import tqdm

try:
    import pyroomacoustics as pra
    HAS_PRA = True
except ImportError:
    HAS_PRA = False
    print("WARNING: pyroomacoustics not installed. Reverb generation disabled.")


# DEMAND noise types (same as data_module.py)
NOISE_TYPES = ['babble', 'cafeteria', 'car', 'kitchen', 'meeting',
               'metro', 'restaurant', 'ssn', 'station', 'traffic']
NOISE2ID = {n: i for i, n in enumerate(NOISE_TYPES)}
NOISE2ID['none'] = 10  # 11th class for no-noise


def apply_clipping_distortion(audio, intensity):
    """Apply clipping distortion.

    Args:
        audio: numpy array
        intensity: float 0.0-1.0 (0=no distortion, 1=heavy clipping)
    Returns:
        distorted audio
    """
    if intensity <= 0:
        return audio
    # Threshold decreases as intensity increases
    # intensity=0.1 -> threshold=0.9, intensity=1.0 -> threshold=0.1
    threshold = max(0.05, 1.0 - intensity * 0.9)
    peak = np.max(np.abs(audio)) + 1e-8
    normalized = audio / peak
    clipped = np.clip(normalized, -threshold, threshold)
    # Re-normalize to original peak
    return clipped * peak


def generate_rir(t60, sr=16000):
    """Generate a Room Impulse Response using pyroomacoustics.

    Args:
        t60: reverberation time in seconds
        sr: sample rate
    Returns:
        rir: 1D numpy array
    """
    if not HAS_PRA:
        raise RuntimeError("pyroomacoustics required for reverb generation")

    # Random room dimensions
    room_dim = [
        np.random.uniform(5, 12),
        np.random.uniform(5, 12),
        np.random.uniform(2.5, 4.5)
    ]

    # Compute absorption
    e_absorption, max_order = pra.inverse_sabine(t60, room_dim)
    max_order = min(max_order, 15)

    room = pra.ShoeBox(
        room_dim, fs=sr,
        materials=pra.Material(e_absorption),
        max_order=max_order
    )

    # Random mic and source positions (at least 1m from walls)
    margin = 1.0
    mic_pos = [np.random.uniform(margin, d - margin) for d in room_dim]
    src_pos = [np.random.uniform(margin, d - margin) for d in room_dim]

    room.add_microphone(mic_pos)
    room.add_source(src_pos)
    room.compute_rir()

    rir = room.rir[0][0]
    # Normalize RIR
    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    return rir


def apply_reverb(audio, rir):
    """Convolve audio with RIR."""
    reverbed = np.convolve(audio, rir)
    # Trim to original length
    reverbed = reverbed[:len(audio)]
    # Normalize to match original energy
    orig_energy = np.sqrt(np.mean(audio ** 2)) + 1e-8
    reverbed_energy = np.sqrt(np.mean(reverbed ** 2)) + 1e-8
    reverbed = reverbed * (orig_energy / reverbed_energy)
    return reverbed


def mix_at_snr(clean, noise, snr_db):
    """Mix clean and noise at specified SNR."""
    clean_len = len(clean)
    noise_len = len(noise)

    if noise_len < clean_len:
        repeats = (clean_len // noise_len) + 1
        noise = np.tile(noise, repeats)
    noise = noise[:clean_len]

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2) + 1e-8
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(clean_power / (snr_linear * noise_power))
    return clean + scale * noise


def parse_noise_labels(label_path):
    """Parse noise_label.txt → dict: filename_stem → noise_type."""
    labels = {}
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[parts[0]] = parts[1]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Create multi-degradation training data")
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--noisy_dir", type=str, required=True)
    parser.add_argument("--noise_label", type=str, required=True, help="Path to noise_label.txt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--extra_combos", type=int, default=2,
                        help="Number of extra degradation combos per file (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load existing files
    clean_files = sorted(glob(os.path.join(args.clean_dir, "*.wav")))
    noisy_files = sorted(glob(os.path.join(args.noisy_dir, "*.wav")))
    noise_labels = parse_noise_labels(args.noise_label)

    print(f"Clean files: {len(clean_files)}")
    print(f"Noisy files: {len(noisy_files)}")
    print(f"Noise labels: {len(noise_labels)}")

    # Output dirs
    out_clean = os.path.join(args.output_dir, "clean")
    out_noisy = os.path.join(args.output_dir, "noisy")
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_noisy, exist_ok=True)

    # Degradation combination types
    COMBO_TYPES = [
        "noise_reverb",       # noise + reverb
        "noise_distort",      # noise + distortion
        "noise_reverb_distort",  # noise + reverb + distortion
        "reverb_only",        # reverb only
        "distort_only",       # distortion only
    ]

    rows = []

    for i, (clean_path, noisy_path) in enumerate(tqdm(
            zip(clean_files, noisy_files), total=len(clean_files),
            desc="Generating multi-degradation data")):

        basename = os.path.basename(clean_path)
        stem = os.path.splitext(basename)[0]

        # Read audio
        clean_wav, sr = sf.read(clean_path)
        noisy_wav, _ = sf.read(noisy_path)

        # Get noise type from label
        noise_type = noise_labels.get(stem, "unknown")
        if noise_type not in NOISE2ID:
            noise_type = "unknown"

        # --- 1. Keep original noise-only pair ---
        out_name_n = f"{stem}_n.wav"
        sf.write(os.path.join(out_clean, out_name_n), clean_wav, sr)
        sf.write(os.path.join(out_noisy, out_name_n), noisy_wav, sr)

        # Estimate SNR from existing noisy
        noise_est = noisy_wav - clean_wav
        clean_power = np.mean(clean_wav ** 2) + 1e-8
        noise_power = np.mean(noise_est ** 2) + 1e-8
        snr_est = 10 * np.log10(clean_power / noise_power)

        rows.append({
            "filename": out_name_n,
            "noise_type": noise_type,
            "snr": round(float(snr_est), 1),
            "reverb_t60": 0.0,
            "distort_intensity": 0.0,
        })

        # --- 2. Generate extra combinations ---
        selected_combos = random.sample(COMBO_TYPES, min(args.extra_combos, len(COMBO_TYPES)))

        for combo in selected_combos:
            # Random parameters
            t60 = round(np.random.uniform(0.3, 1.0), 2) if "reverb" in combo else 0.0
            distort_intensity = round(np.random.uniform(0.2, 0.8), 2) if "distort" in combo else 0.0
            snr_db = round(np.random.uniform(0, 20), 1) if "noise" in combo else 0.0

            # Build degraded signal
            if combo == "reverb_only":
                # Apply reverb to clean
                suffix = "r"
                rir = generate_rir(t60, sr)
                degraded = apply_reverb(clean_wav, rir)
                cur_noise_type = "none"
                cur_snr = 0.0
            elif combo == "distort_only":
                # Apply distortion to clean
                suffix = "d"
                degraded = apply_clipping_distortion(clean_wav, distort_intensity)
                cur_noise_type = "none"
                cur_snr = 0.0
            elif combo == "noise_reverb":
                suffix = "nr"
                rir = generate_rir(t60, sr)
                # Add noise first, then reverb
                degraded = mix_at_snr(clean_wav, noise_est, snr_db)
                degraded = apply_reverb(degraded, rir)
                cur_noise_type = noise_type
                cur_snr = snr_db
            elif combo == "noise_distort":
                suffix = "nd"
                degraded = mix_at_snr(clean_wav, noise_est, snr_db)
                degraded = apply_clipping_distortion(degraded, distort_intensity)
                cur_noise_type = noise_type
                cur_snr = snr_db
            elif combo == "noise_reverb_distort":
                suffix = "nrd"
                rir = generate_rir(t60, sr)
                degraded = mix_at_snr(clean_wav, noise_est, snr_db)
                degraded = apply_reverb(degraded, rir)
                degraded = apply_clipping_distortion(degraded, distort_intensity)
                cur_noise_type = noise_type
                cur_snr = snr_db
            else:
                continue

            out_name = f"{stem}_{suffix}.wav"
            sf.write(os.path.join(out_clean, out_name), clean_wav, sr)
            sf.write(os.path.join(out_noisy, out_name), degraded.astype(np.float32), sr)

            rows.append({
                "filename": out_name,
                "noise_type": cur_noise_type,
                "snr": round(float(cur_snr), 1),
                "reverb_t60": t60,
                "distort_intensity": distort_intensity,
            })

    # Write labels.csv
    csv_path = os.path.join(args.output_dir, "labels.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f,
            fieldnames=["filename", "noise_type", "snr", "reverb_t60", "distort_intensity"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Generated {len(rows)} files")
    print(f"  Original noise-only: {len(clean_files)}")
    print(f"  Extra combos: {len(rows) - len(clean_files)}")
    print(f"  Labels: {csv_path}")

    # Stats
    combo_counts = {}
    for r in rows:
        has_noise = r["noise_type"] != "none"
        has_reverb = r["reverb_t60"] > 0
        has_distort = r["distort_intensity"] > 0
        key = "+".join(filter(None, [
            "noise" if has_noise else "",
            "reverb" if has_reverb else "",
            "distort" if has_distort else "",
        ])) or "clean"
        combo_counts[key] = combo_counts.get(key, 0) + 1
    print("\nDegradation distribution:")
    for k, v in sorted(combo_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
