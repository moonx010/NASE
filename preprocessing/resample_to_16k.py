"""
VoiceBank-DEMAND 48kHz → 16kHz 리샘플링 스크립트.

Usage:
    python preprocessing/resample_to_16k.py \
        --input_dir /path/to/voicebank-demand-48k \
        --output_dir /path/to/voicebank-demand-16k

Expected input structure:
    input_dir/
    ├── train/
    │   ├── clean/*.wav
    │   ├── noisy/*.wav
    │   └── noise_label.txt
    ├── valid/
    │   ├── clean/*.wav
    │   └── noisy/*.wav
    └── test/
        ├── clean/*.wav
        ├── noisy/*.wav
        └── noise_label.txt  (if exists)

Output structure will be identical, with all wav files resampled to 16kHz.
"""

import os
import argparse
from pathlib import Path
from glob import glob

import torch
import torchaudio


def resample_directory(input_dir, output_dir, orig_sr=48000, target_sr=16000):
    """Resample all wav files in input_dir and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(glob(os.path.join(input_dir, "*.wav")))
    if not wav_files:
        print(f"  No wav files found in {input_dir}, skipping.")
        return 0

    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    count = 0

    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        output_path = os.path.join(output_dir, filename)

        waveform, sr = torchaudio.load(wav_path)

        # 실제 sample rate 확인
        if sr != orig_sr:
            if sr == target_sr:
                # 이미 16kHz인 경우 그대로 복사
                torchaudio.save(output_path, waveform, target_sr)
                count += 1
                continue
            else:
                # 예상과 다른 sample rate
                resampler = torchaudio.transforms.Resample(sr, target_sr)

        waveform_16k = resampler(waveform)
        torchaudio.save(output_path, waveform_16k, target_sr)
        count += 1

    return count


def copy_text_files(input_dir, output_dir):
    """Copy noise_label.txt and other text files."""
    import shutil
    for txt_file in glob(os.path.join(input_dir, "*.txt")):
        filename = os.path.basename(txt_file)
        shutil.copy2(txt_file, os.path.join(output_dir, filename))
        print(f"  Copied {filename}")


def main():
    parser = argparse.ArgumentParser(description="Resample VoiceBank-DEMAND to 16kHz")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to original VoiceBank-DEMAND dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save 16kHz resampled dataset")
    parser.add_argument("--orig_sr", type=int, default=48000,
                        help="Original sample rate (default: 48000)")
    parser.add_argument("--target_sr", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    args = parser.parse_args()

    subsets = ["train", "valid", "test"]
    subdirs = ["clean", "noisy"]

    print(f"Resampling: {args.orig_sr}Hz → {args.target_sr}Hz")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print()

    total = 0
    for subset in subsets:
        subset_input = os.path.join(args.input_dir, subset)
        subset_output = os.path.join(args.output_dir, subset)

        if not os.path.exists(subset_input):
            print(f"[SKIP] {subset}/ not found")
            continue

        print(f"[{subset}]")

        # Resample wav files in clean/ and noisy/
        for subdir in subdirs:
            dir_input = os.path.join(subset_input, subdir)
            dir_output = os.path.join(subset_output, subdir)

            if not os.path.exists(dir_input):
                print(f"  [SKIP] {subset}/{subdir}/ not found")
                continue

            count = resample_directory(dir_input, dir_output,
                                       args.orig_sr, args.target_sr)
            total += count
            print(f"  {subset}/{subdir}: {count} files resampled")

        # Copy noise_label.txt and other text files
        copy_text_files(subset_input, subset_output)

    print(f"\nDone! Total {total} files resampled to {args.target_sr}Hz.")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
