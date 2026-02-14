#!/usr/bin/env bash
# ============================================================
# E2-v2 (fixed NC loss + CFG) Evaluation — Parallel
# Server: 159-67, GPU 1-5
#
# 1) Baseline (no guidance): in-dist + OOD×3
# 2) Adaptive guidance (confidence, fixed sigmoid): in-dist + OOD×3
# ============================================================
CKPT="logs/e2v2-fixed-cfg-fast/last.ckpt"
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50

echo "===== E2-v2 Evaluation (5 GPUs parallel) ====="

# --- Batch 1: 5 jobs on GPU 1-5 ---
# Baseline (no guidance)
CUDA_VISIBLE_DEVICES=1 python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e2v2-fixed/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N &
P1=$!

CUDA_VISIBLE_DEVICES=2 python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e2v2-fixed/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N &
P2=$!

CUDA_VISIBLE_DEVICES=3 python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e2v2-fixed/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N &
P3=$!

CUDA_VISIBLE_DEVICES=4 python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e2v2-fixed/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N &
P4=$!

# Adaptive guidance (confidence, linear mapping)
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e2v2-adaptive-conf/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear &
P5=$!

echo "Batch 1: 5 jobs (GPU 1-5). Waiting..."
wait $P1 $P2 $P3 $P4 $P5

# --- Batch 2: 3 jobs on GPU 1-3 ---
CUDA_VISIBLE_DEVICES=1 python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e2v2-adaptive-conf/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear &
P6=$!

CUDA_VISIBLE_DEVICES=2 python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e2v2-adaptive-conf/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear &
P7=$!

CUDA_VISIBLE_DEVICES=3 python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e2v2-adaptive-conf/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear &
P8=$!

echo "Batch 2: 3 jobs (GPU 1-3). Waiting..."
wait $P6 $P7 $P8

echo "All enhancement done. Running metrics..."

# Metrics (CPU)
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e2v2-fixed/in_dist
python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e2v2-fixed/ood_esc50_all
python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e2v2-fixed/ood_stationary
python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e2v2-fixed/ood_non_stationary

python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e2v2-adaptive-conf/in_dist
python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e2v2-adaptive-conf/ood_esc50_all
python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e2v2-adaptive-conf/ood_stationary
python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e2v2-adaptive-conf/ood_non_stationary

echo ""
echo "===== Done. Run: python scripts/summarize_results.py ====="
