#!/usr/bin/env bash
# ============================================================
# E1: NASE Baseline Enhancement (Inference)
# - Standard inference (no CFG, w=0 equivalent)
# ============================================================

# ---- 경로 설정 ----
TEST_DIR="/path/to/voicebank-demand-16k/test"
ENHANCED_DIR="./enhanced/e1_baseline"
CKPT="/path/to/logs/<run_id>/epoch-XX-pesq-X.XX.ckpt"
BEATS_CKPT="/path/to/BEATs_iter3_plus_AS2M.pt"

# ---- Inference 설정 ----
N=50         # reverse steps
CORRECTOR="ald"
SNR=0.5

python enhancement.py \
    --test_dir ${TEST_DIR} \
    --test_set noisy \
    --enhanced_dir ${ENHANCED_DIR} \
    --ckpt ${CKPT} \
    --pretrain_class_model ${BEATS_CKPT} \
    --N ${N} \
    --corrector ${CORRECTOR} \
    --snr ${SNR}
