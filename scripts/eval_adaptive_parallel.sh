#!/usr/bin/env bash
# ============================================================
# Adaptive Guidance Evaluation - Parallel on multiple GPUs
# GPU 1 is already running linear in-dist, so skip that.
# Usage: cd NASE && bash scripts/eval_adaptive_parallel.sh
# ============================================================
CKPT="logs/dx2ds38e/last.ckpt"
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50

echo "===== Launching Adaptive Guidance (parallel) ====="

# Linear mapping - OOD sets
CUDA_VISIBLE_DEVICES=2 python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-adaptive-linear/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear &
PID1=$!

CUDA_VISIBLE_DEVICES=3 python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-adaptive-linear/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear &
PID2=$!

CUDA_VISIBLE_DEVICES=4 python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-adaptive-linear/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear &
PID3=$!

# Scaled mapping - all sets
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-adaptive-scaled/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping scaled &
PID4=$!

CUDA_VISIBLE_DEVICES=6 python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-adaptive-scaled/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping scaled &
PID5=$!

CUDA_VISIBLE_DEVICES=7 python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-adaptive-scaled/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping scaled &
PID6=$!

CUDA_VISIBLE_DEVICES=0 python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-adaptive-scaled/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping scaled &
PID7=$!

echo "7 jobs launched on GPU 0,2,3,4,5,6,7. Waiting..."
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7
echo "All enhancement done. Running metrics..."

# Calc metrics (CPU, fast)
python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-adaptive-linear/ood_esc50_all
python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-adaptive-linear/ood_stationary
python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-adaptive-linear/ood_non_stationary
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-adaptive-scaled/in_dist
python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-adaptive-scaled/ood_esc50_all
python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-adaptive-scaled/ood_stationary
python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-adaptive-scaled/ood_non_stationary

# Also calc metrics for linear in-dist (running on GPU 1 separately)
if [ -d "enhanced/e3-adaptive-linear/in_dist" ]; then
    python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-adaptive-linear/in_dist
fi

echo ""
echo "===== All done. Run: python scripts/summarize_results.py ====="
