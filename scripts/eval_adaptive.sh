#!/usr/bin/env bash
# ============================================================
# Evaluate Adaptive Guidance on all test sets
# Uses E2-fast CFG checkpoint (trained with p_uncond=0.2)
#
# Usage: cd NASE && bash scripts/eval_adaptive.sh
# ============================================================
CKPT="logs/dx2ds38e/last.ckpt"
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50

echo "===== Adaptive Guidance Evaluation ====="

# 1. In-dist: linear mapping
echo "--- [1] In-dist, adaptive linear w_max=1.0 ---"
python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-adaptive-linear/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-adaptive-linear/in_dist

# 2. OOD all: linear mapping
echo "--- [2] OOD all, adaptive linear w_max=1.0 ---"
if [ -d "$D/ood_test/esc50/all/noisy" ]; then
    python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-adaptive-linear/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear
    python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-adaptive-linear/ood_esc50_all
fi

# 3. OOD stationary
echo "--- [3] OOD stationary, adaptive linear ---"
if [ -d "$D/ood_test/esc50/stationary/noisy" ]; then
    python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-adaptive-linear/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear
    python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-adaptive-linear/ood_stationary
fi

# 4. OOD non-stationary
echo "--- [4] OOD non-stationary, adaptive linear ---"
if [ -d "$D/ood_test/esc50/non_stationary/noisy" ]; then
    python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-adaptive-linear/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping linear
    python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-adaptive-linear/ood_non_stationary
fi

# 5. In-dist: scaled mapping (w always >= 0)
echo "--- [5] In-dist, adaptive scaled w_max=1.0 ---"
python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-adaptive-scaled/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping scaled
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-adaptive-scaled/in_dist

# 6. OOD all: scaled mapping
echo "--- [6] OOD all, adaptive scaled w_max=1.0 ---"
if [ -d "$D/ood_test/esc50/all/noisy" ]; then
    python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-adaptive-scaled/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --w_max 1.0 --guidance_mapping scaled
    python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-adaptive-scaled/ood_esc50_all
fi

echo ""
echo "===== Done. Run: python scripts/summarize_results.py ====="
