#!/usr/bin/env bash
# ============================================================
# Evaluate SGMSE+ Dereverb pretrained model on all test sets
# Uses ORIGINAL SGMSE codebase (not NASE)
# ============================================================

set -e

SGMSE_DIR="/home/nas4_user/kyudanjung/seokhoonmoon/sgmse"
CKPT="${SGMSE_DIR}/checkpoints/wsj0_reverb.ckpt"
DATA_BASE="/home/nas4_user/kyudanjung/seokhoonmoon/data"
NASE_DIR="/home/kyudanjung/NASE"
OUT_BASE="${NASE_DIR}/logs/eval_dereverb"
N=50
GPU=${1:-0}

export CUDA_VISIBLE_DEVICES=${GPU}

mkdir -p ${OUT_BASE}

echo "===== SGMSE+ Dereverb Evaluation ====="
echo "Checkpoint: ${CKPT}"
echo "GPU: ${GPU}"
echo "N: ${N}"
echo ""

# 1. Multi-degradation test (2472 files)
echo "--- [1/3] Multi-degradation test (2472 files) ---"
cd ${SGMSE_DIR}
python enhancement.py \
    --test_dir ${DATA_BASE}/multi_degradation_16k/test_multi/noisy \
    --enhanced_dir ${OUT_BASE}/multi_deg \
    --ckpt ${CKPT} \
    --N ${N} \
    2>&1 | tee ${OUT_BASE}/multi_deg.log

cd ${NASE_DIR}
python calc_metrics.py \
    --test_dir ${DATA_BASE}/multi_degradation_16k/test_multi \
    --enhanced_dir ${OUT_BASE}/multi_deg \
    2>&1 | tee ${OUT_BASE}/multi_deg_metrics.log

# 2. Noise-only test (824 files)
echo "--- [2/3] Noise-only test (VoiceBank-DEMAND, 824 files) ---"
cd ${SGMSE_DIR}
python enhancement.py \
    --test_dir ${DATA_BASE}/voicebank-demand-16k/test/noisy \
    --enhanced_dir ${OUT_BASE}/noise_only \
    --ckpt ${CKPT} \
    --N ${N} \
    2>&1 | tee ${OUT_BASE}/noise_only.log

cd ${NASE_DIR}
python calc_metrics.py \
    --test_dir ${DATA_BASE}/voicebank-demand-16k/test \
    --enhanced_dir ${OUT_BASE}/noise_only \
    2>&1 | tee ${OUT_BASE}/noise_only_metrics.log

# 3. OOD test (ESC-50, 200 files)
echo "--- [3/3] OOD test (ESC-50, 200 files) ---"
cd ${SGMSE_DIR}
python enhancement.py \
    --test_dir ${DATA_BASE}/ood_test/esc50/all/noisy \
    --enhanced_dir ${OUT_BASE}/ood \
    --ckpt ${CKPT} \
    --N ${N} \
    2>&1 | tee ${OUT_BASE}/ood.log

cd ${NASE_DIR}
python calc_metrics.py \
    --test_dir ${DATA_BASE}/ood_test/esc50/all \
    --enhanced_dir ${OUT_BASE}/ood \
    2>&1 | tee ${OUT_BASE}/ood_metrics.log

echo ""
echo "===== All done! Results in ${OUT_BASE}/ ====="
