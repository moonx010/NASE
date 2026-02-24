#!/usr/bin/env bash
# ============================================================
# Score-level CFG Sweep — 145 server, 8 GPUs parallel
# Model: multi-deg-wavlm-v2 ep159 (temb injection)
#
# CFG formula: score = (1+w) * score_cond - w * score_uncond
# w values: 0.1, 0.25, 0.5, 1.0, 2.0, 3.0
#
# Usage: cd /home/nas4_user/kyudanjung/seokhoonmoon/NASE && conda activate sgmse && bash scripts/eval_cfg_sweep.sh
# ============================================================
set -e

CKPT="logs/multi-deg-wavlm-v2/epoch=159-last.ckpt"
WAVLM="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"  # WavLM ignores this; argparse requires it
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=30
TAG="cfg-sweep"

MULTI="$D/multi_degradation_16k/test_multi"
NOISE="$D/voicebank-demand-16k/test"
OOD="$D/ood_test/esc50/all"

COMMON="--ckpt $CKPT --pretrain_class_model $WAVLM --encoder_type wavlm --multi_degradation --inject_method temb --N $N --test_set noisy"

echo "===== Score-level CFG Sweep (8 GPUs) ====="
echo "CKPT=$CKPT"
echo "w values: 0.1 0.25 0.5 1.0 2.0 3.0"

# --- Batch 1: multi-deg (6 GPUs) + noise-only (2 GPUs) ---
# Multi-deg (2472 files each)
CUDA_VISIBLE_DEVICES=0 python enhancement.py --test_dir $MULTI --enhanced_dir enhanced/$TAG/multi_w0.1 $COMMON --guidance_scale 0.1 &
P0=$!

CUDA_VISIBLE_DEVICES=1 python enhancement.py --test_dir $MULTI --enhanced_dir enhanced/$TAG/multi_w0.25 $COMMON --guidance_scale 0.25 &
P1=$!

CUDA_VISIBLE_DEVICES=2 python enhancement.py --test_dir $MULTI --enhanced_dir enhanced/$TAG/multi_w0.5 $COMMON --guidance_scale 0.5 &
P2=$!

CUDA_VISIBLE_DEVICES=3 python enhancement.py --test_dir $MULTI --enhanced_dir enhanced/$TAG/multi_w1.0 $COMMON --guidance_scale 1.0 &
P3=$!

CUDA_VISIBLE_DEVICES=4 python enhancement.py --test_dir $MULTI --enhanced_dir enhanced/$TAG/multi_w2.0 $COMMON --guidance_scale 2.0 &
P4=$!

CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $MULTI --enhanced_dir enhanced/$TAG/multi_w3.0 $COMMON --guidance_scale 3.0 &
P5=$!

# Noise-only sequential on GPU 6,7 (824 files each, faster)
(
for W in 0.1 0.25 0.5; do
    CUDA_VISIBLE_DEVICES=6 python enhancement.py --test_dir $NOISE --enhanced_dir enhanced/$TAG/noise_w$W $COMMON --guidance_scale $W
done
) &
P6=$!

(
for W in 1.0 2.0 3.0; do
    CUDA_VISIBLE_DEVICES=7 python enhancement.py --test_dir $NOISE --enhanced_dir enhanced/$TAG/noise_w$W $COMMON --guidance_scale $W
done
) &
P7=$!

echo "Batch 1: GPU 0-5 multi-deg, GPU 6-7 noise-only. Waiting..."
wait $P0 $P1 $P2 $P3 $P4 $P5 $P6 $P7
echo "Batch 1 done."

# --- Batch 2: OOD (200 files, fast) ---
echo "Batch 2: OOD evaluation..."
(
for W in 0.1 0.25 0.5; do
    CUDA_VISIBLE_DEVICES=0 python enhancement.py --test_dir $OOD --enhanced_dir enhanced/$TAG/ood_w$W $COMMON --guidance_scale $W
done
) &
Q0=$!

(
for W in 1.0 2.0 3.0; do
    CUDA_VISIBLE_DEVICES=1 python enhancement.py --test_dir $OOD --enhanced_dir enhanced/$TAG/ood_w$W $COMMON --guidance_scale $W
done
) &
Q1=$!

wait $Q0 $Q1
echo "All enhancement done."

# --- Metrics ---
echo ""
echo "===== Computing Metrics ====="

for W in 0.1 0.25 0.5 1.0 2.0 3.0; do
    echo "--- w=$W multi-deg ---"
    python calc_metrics.py --test_dir $MULTI --enhanced_dir enhanced/$TAG/multi_w$W --utmos
    echo "--- w=$W noise-only ---"
    python calc_metrics.py --test_dir $NOISE --enhanced_dir enhanced/$TAG/noise_w$W --utmos
    echo "--- w=$W OOD ---"
    python calc_metrics.py --test_dir $OOD --enhanced_dir enhanced/$TAG/ood_w$W --utmos
done

echo ""
echo "===== CFG Sweep Complete ====="
