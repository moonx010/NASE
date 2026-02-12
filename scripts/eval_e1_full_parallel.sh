#!/usr/bin/env bash
# ============================================================
# E1 Full Baseline (sqnju2ez) - N=50 evaluation on all test sets
# 8 GPUs parallel
# Usage: cd NASE && bash scripts/eval_e1_full_parallel.sh
# ============================================================
set -e

CKPT="logs/sqnju2ez/epoch=92-pesq=2.74.ckpt"
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50

echo "===== E1 Full Baseline Eval (N=$N) ====="
echo "CKPT=$CKPT"

# GPU 4: in-dist (824 files, slow)
# GPU 5: OOD 3개 순차 (각각 적으니 금방)
CUDA_VISIBLE_DEVICES=4 python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e1-full-N50/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N &
PID0=$!

(
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e1-full-N50/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N && \
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e1-full-N50/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N && \
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e1-full-N50/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N
) &
PID1=$!

echo "GPU 4: in-dist, GPU 5: OOD all→stat→non-stat. Waiting..."
wait $PID0 $PID1
echo "All enhancement done. Running metrics..."

# Calc metrics (CPU, sequential)
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e1-full-N50/in_dist
python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e1-full-N50/ood_esc50_all
python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e1-full-N50/ood_stationary
python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e1-full-N50/ood_non_stationary

echo ""
python scripts/summarize_results.py enhanced/e1-full-N50
echo "===== Done ====="
