#!/usr/bin/env bash
# 159-67 GPU 4: postcnn kNN adaptive v2
# Usage: cd NASE && bash scripts/eval_dist_v2_postcnn.sh 4
set -e

GPU=${1:-4}
CKPT="logs/dx2ds38e/last.ckpt"
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
REF="embeddings/beats_postcnn_train_all.pt"
N=50
K=10
TAU=1.0
TAG="e3v2-knn-postcnn"

echo "===== postcnn kNN v2 (GPU $GPU) ====="

for TESTNAME in in_dist ood_esc50_all ood_stationary ood_non_stationary; do
    case $TESTNAME in
        in_dist)        TEST="$D/voicebank-demand-16k/test" ;;
        ood_esc50_all)  TEST="$D/ood_test/esc50/all" ;;
        ood_stationary) TEST="$D/ood_test/esc50/stationary" ;;
        ood_non_stationary) TEST="$D/ood_test/esc50/non_stationary" ;;
    esac
    echo "--- $TESTNAME ---"
    CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $TEST --enhanced_dir enhanced/$TAG/$TESTNAME --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF --k_neighbors $K --tau $TAU
done

echo ""
echo "===== Metrics ====="
for TESTNAME in in_dist ood_esc50_all ood_stationary ood_non_stationary; do
    case $TESTNAME in
        in_dist)        TEST="$D/voicebank-demand-16k/test" ;;
        ood_esc50_all)  TEST="$D/ood_test/esc50/all" ;;
        ood_stationary) TEST="$D/ood_test/esc50/stationary" ;;
        ood_non_stationary) TEST="$D/ood_test/esc50/non_stationary" ;;
    esac
    python calc_metrics.py --test_dir $TEST --enhanced_dir enhanced/$TAG/$TESTNAME
done

python scripts/summarize_results.py enhanced/$TAG
echo "===== Done ====="
