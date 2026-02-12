#!/usr/bin/env bash
# ============================================================
# EXP-2: Static w Sweep — E2-fast checkpoint
# Tests w = 0.0, 0.25, 0.5, 1.0, 2.0 on in-dist + OOD-all
# Also runs unconditional (forward_uncond) as w=-1
#
# Usage: cd NASE && bash scripts/eval_static_w_sweep.sh <GPU_ID>
# Example: bash scripts/eval_static_w_sweep.sh 0
# ============================================================
set -e

GPU=${1:-0}
CKPT="logs/dx2ds38e/last.ckpt"
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50

echo "===== EXP-2: Static w Sweep ====="
echo "GPU=$GPU, CKPT=$CKPT, N=$N"

# Test sets
IN_DIST="$D/voicebank-demand-16k/test"
OOD_ALL="$D/ood_test/esc50/all"

# w values to sweep
for W in 0.0 0.25 0.5 1.0 2.0; do
    echo ""
    echo "--- w=${W} ---"

    # In-dist
    echo "  [in-dist] enhancing..."
    CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $IN_DIST --enhanced_dir enhanced/e2-static-w${W}/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale $W

    # OOD-all
    echo "  [ood-all] enhancing..."
    CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $OOD_ALL --enhanced_dir enhanced/e2-static-w${W}/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale $W
done

# Unconditional (w=-1 signals pure unconditional in our code)
echo ""
echo "--- Unconditional (w=-1) ---"
CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $IN_DIST --enhanced_dir enhanced/e2-static-uncond/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale -1.0
CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $OOD_ALL --enhanced_dir enhanced/e2-static-uncond/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale -1.0

# Calc metrics for all
echo ""
echo "===== Calculating metrics ====="
for W in 0.0 0.25 0.5 1.0 2.0; do
    python calc_metrics.py --test_dir $IN_DIST --enhanced_dir enhanced/e2-static-w${W}/in_dist
    python calc_metrics.py --test_dir $OOD_ALL --enhanced_dir enhanced/e2-static-w${W}/ood_esc50_all
done
python calc_metrics.py --test_dir $IN_DIST --enhanced_dir enhanced/e2-static-uncond/in_dist
python calc_metrics.py --test_dir $OOD_ALL --enhanced_dir enhanced/e2-static-uncond/ood_esc50_all

# Summarize
echo ""
echo "===== Summary ====="
python scripts/summarize_results.py enhanced/e2-static-w0.0 enhanced/e2-static-w0.25 enhanced/e2-static-w0.5 enhanced/e2-static-w1.0 enhanced/e2-static-w2.0 enhanced/e2-static-uncond
echo "===== Done ====="
