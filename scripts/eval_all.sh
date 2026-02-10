#!/usr/bin/env bash
# ============================================================
# Comprehensive Evaluation: Run enhancement + metrics for all models
#
# Usage: bash scripts/eval_all.sh <ckpt_path> <run_name> <beats_path>
# Example: bash scripts/eval_all.sh logs/e1-fast-baseline/epoch-39-last.ckpt e1-fast /path/to/BEATs.pt
#
# This will evaluate on:
# 1. In-distribution: VoiceBank-DEMAND test set
# 2. OOD: ESC-50 (all / stationary / non-stationary)
# ============================================================

CKPT=${1:?"Usage: $0 <ckpt_path> <run_name> <beats_path> [guidance_scale]"}
RUN_NAME=${2:?"Usage: $0 <ckpt_path> <run_name> <beats_path> [guidance_scale]"}
BEATS=${3:?"Usage: $0 <ckpt_path> <run_name> <beats_path> [guidance_scale]"}
W=${4:-""}  # optional guidance_scale

DATA_BASE="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50

# Build guidance flag
if [ -n "$W" ]; then
    GS_FLAG="--guidance_scale ${W}"
    RUN_NAME="${RUN_NAME}_w${W}"
else
    GS_FLAG=""
fi

echo "===== Evaluation: ${RUN_NAME} ====="
echo "Checkpoint: ${CKPT}"
echo "Guidance: ${W:-none}"
echo ""

# 1. In-distribution test
echo "--- [1/4] In-distribution (VoiceBank-DEMAND test) ---"
python enhancement.py --test_dir ${DATA_BASE}/voicebank-demand-16k/test --enhanced_dir enhanced/${RUN_NAME}/in_dist --ckpt ${CKPT} --pretrain_class_model ${BEATS} --N ${N} ${GS_FLAG}
python calc_metrics.py --test_dir ${DATA_BASE}/voicebank-demand-16k/test --enhanced_dir enhanced/${RUN_NAME}/in_dist

# 2. OOD - ESC-50 all
echo "--- [2/4] OOD ESC-50 (all) ---"
if [ -d "${DATA_BASE}/ood_test/esc50/all/noisy" ]; then
    python enhancement.py --test_dir ${DATA_BASE}/ood_test/esc50/all --enhanced_dir enhanced/${RUN_NAME}/ood_esc50_all --ckpt ${CKPT} --pretrain_class_model ${BEATS} --N ${N} ${GS_FLAG}
    python calc_metrics.py --test_dir ${DATA_BASE}/ood_test/esc50/all --enhanced_dir enhanced/${RUN_NAME}/ood_esc50_all
else
    echo "  [SKIP] OOD ESC-50 not found. Run create_ood_test.py first."
fi

# 3. OOD - ESC-50 stationary
echo "--- [3/4] OOD ESC-50 (stationary) ---"
if [ -d "${DATA_BASE}/ood_test/esc50/stationary/noisy" ]; then
    python enhancement.py --test_dir ${DATA_BASE}/ood_test/esc50/stationary --enhanced_dir enhanced/${RUN_NAME}/ood_stationary --ckpt ${CKPT} --pretrain_class_model ${BEATS} --N ${N} ${GS_FLAG}
    python calc_metrics.py --test_dir ${DATA_BASE}/ood_test/esc50/stationary --enhanced_dir enhanced/${RUN_NAME}/ood_stationary
else
    echo "  [SKIP] stationary subset not found."
fi

# 4. OOD - ESC-50 non-stationary
echo "--- [4/4] OOD ESC-50 (non-stationary) ---"
if [ -d "${DATA_BASE}/ood_test/esc50/non_stationary/noisy" ]; then
    python enhancement.py --test_dir ${DATA_BASE}/ood_test/esc50/non_stationary --enhanced_dir enhanced/${RUN_NAME}/ood_non_stationary --ckpt ${CKPT} --pretrain_class_model ${BEATS} --N ${N} ${GS_FLAG}
    python calc_metrics.py --test_dir ${DATA_BASE}/ood_test/esc50/non_stationary --enhanced_dir enhanced/${RUN_NAME}/ood_non_stationary
else
    echo "  [SKIP] non_stationary subset not found."
fi

echo ""
echo "===== Done: ${RUN_NAME} ====="
echo "Results saved in enhanced/${RUN_NAME}/"
