#!/usr/bin/env bash
# ============================================================
# WavLM Baseline Eval (40ep, p_uncond=0.0)
# Best ckpt: epoch=37-pesq=2.73
# In-dist + OOD (all/stat/nonstat)
#
# Usage: cd NASE && bash scripts/eval_wavlm_baseline.sh <GPU>
# ============================================================
set -e

GPU=${1:-4}
CKPT="logs/wavlm-fast-baseline/epoch=37-pesq=2.73.ckpt"
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50
TAG="wavlm-baseline"

IN_DIST="$D/voicebank-demand-16k/test"
OOD_ALL="$D/ood_test/esc50/all"
OOD_STAT="$D/ood_test/esc50/stationary"
OOD_NONSTAT="$D/ood_test/esc50/non_stationary"

echo "===== WavLM Baseline Eval (GPU $GPU) ====="

for TESTNAME in in_dist ood_esc50_all ood_stationary ood_non_stationary; do
    case $TESTNAME in
        in_dist)        TEST=$IN_DIST ;;
        ood_esc50_all)  TEST=$OOD_ALL ;;
        ood_stationary) TEST=$OOD_STAT ;;
        ood_non_stationary) TEST=$OOD_NONSTAT ;;
    esac
    echo "--- $TESTNAME ---"
    CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $TEST --enhanced_dir enhanced/$TAG/$TESTNAME --ckpt $CKPT --pretrain_class_model $B --N $N --encoder_type wavlm
done

echo ""
echo "===== Metrics ====="
for TESTNAME in in_dist ood_esc50_all ood_stationary ood_non_stationary; do
    case $TESTNAME in
        in_dist)        TEST=$IN_DIST ;;
        ood_esc50_all)  TEST=$OOD_ALL ;;
        ood_stationary) TEST=$OOD_STAT ;;
        ood_non_stationary) TEST=$OOD_NONSTAT ;;
    esac
    python calc_metrics.py --test_dir $TEST --enhanced_dir enhanced/$TAG/$TESTNAME
done

python scripts/summarize_results.py enhanced/$TAG
echo "===== Done ====="
