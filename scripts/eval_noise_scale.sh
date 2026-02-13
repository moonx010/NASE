#!/usr/bin/env bash
# ============================================================
# Embedding Scaling Test: E1-full (best model, no CFG needed)
# Tests alpha=0.0 (no conditioning) vs alpha=1.0 (normal)
# Key hypothesis: alpha=0 on OOD should improve over alpha=1
#
# Usage: cd NASE && bash scripts/eval_noise_scale.sh <GPU>
# ============================================================
set -e

GPU=${1:-4}
CKPT="logs/sqnju2ez/last.ckpt"  # E1-full (160ep, baseline)
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50

IN_DIST="$D/voicebank-demand-16k/test"
OOD_ALL="$D/ood_test/esc50/all"
OOD_STAT="$D/ood_test/esc50/stationary"
OOD_NONSTAT="$D/ood_test/esc50/non_stationary"

echo "===== Embedding Scaling Test (E1-full, GPU $GPU) ====="

# alpha=1.0 (normal) — should match E1-full baseline results
# Skip if already have E1-full results
echo ""
echo "--- alpha=1.0 (normal conditioning) ---"
for TESTNAME in in_dist ood_esc50_all ood_stationary ood_non_stationary; do
    case $TESTNAME in
        in_dist)        TEST=$IN_DIST ;;
        ood_esc50_all)  TEST=$OOD_ALL ;;
        ood_stationary) TEST=$OOD_STAT ;;
        ood_non_stationary) TEST=$OOD_NONSTAT ;;
    esac
    echo "  [$TESTNAME]"
    CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $TEST --enhanced_dir enhanced/e4-scale-1.0/$TESTNAME --ckpt $CKPT --pretrain_class_model $B --N $N --noise_scale 1.0
done

# alpha=0.0 (no conditioning) — if OOD improves, hypothesis validated!
echo ""
echo "--- alpha=0.0 (no conditioning) ---"
for TESTNAME in in_dist ood_esc50_all ood_stationary ood_non_stationary; do
    case $TESTNAME in
        in_dist)        TEST=$IN_DIST ;;
        ood_esc50_all)  TEST=$OOD_ALL ;;
        ood_stationary) TEST=$OOD_STAT ;;
        ood_non_stationary) TEST=$OOD_NONSTAT ;;
    esac
    echo "  [$TESTNAME]"
    CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $TEST --enhanced_dir enhanced/e4-scale-0.0/$TESTNAME --ckpt $CKPT --pretrain_class_model $B --N $N --noise_scale 0.0
done

# alpha=0.5 (half conditioning)
echo ""
echo "--- alpha=0.5 (half conditioning) ---"
for TESTNAME in in_dist ood_esc50_all ood_stationary ood_non_stationary; do
    case $TESTNAME in
        in_dist)        TEST=$IN_DIST ;;
        ood_esc50_all)  TEST=$OOD_ALL ;;
        ood_stationary) TEST=$OOD_STAT ;;
        ood_non_stationary) TEST=$OOD_NONSTAT ;;
    esac
    echo "  [$TESTNAME]"
    CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $TEST --enhanced_dir enhanced/e4-scale-0.5/$TESTNAME --ckpt $CKPT --pretrain_class_model $B --N $N --noise_scale 0.5
done

# Calc metrics
echo ""
echo "===== Metrics ====="
for ALPHA in 1.0 0.0 0.5; do
    for TESTNAME in in_dist ood_esc50_all ood_stationary ood_non_stationary; do
        case $TESTNAME in
            in_dist)        TEST=$IN_DIST ;;
            ood_esc50_all)  TEST=$OOD_ALL ;;
            ood_stationary) TEST=$OOD_STAT ;;
            ood_non_stationary) TEST=$OOD_NONSTAT ;;
        esac
        python calc_metrics.py --test_dir $TEST --enhanced_dir enhanced/e4-scale-${ALPHA}/$TESTNAME
    done
done

python scripts/summarize_results.py enhanced/e4-scale-1.0 enhanced/e4-scale-0.0 enhanced/e4-scale-0.5
echo "===== Done ====="
