#!/usr/bin/env bash
# ============================================================
# EXP-2: Static w Sweep (single GPU)
# w = 0.0, 0.25, 0.5, 1.0, 2.0 + unconditional
# in-dist + OOD-all only (2 test sets per w)
#
# Usage: cd NASE && bash scripts/eval_static_w_sweep_single.sh <GPU> [DATA_ROOT]
# Example (RTX):  bash scripts/eval_static_w_sweep_single.sh 4
# Example (A100): bash scripts/eval_static_w_sweep_single.sh 0 /home/nas5/kyudanjung/seokhoonmoon/data
# ============================================================
set -e

GPU=${1:-0}
D=${2:-/home/nas4_user/kyudanjung/seokhoonmoon/data}
CKPT="logs/dx2ds38e/last.ckpt"
B="$D/BEATs_iter3_plus_AS2M.pt"
N=50

IN_DIST="$D/voicebank-demand-16k/test"
OOD_ALL="$D/ood_test/esc50/all"

echo "===== EXP-2: Static w Sweep (GPU $GPU) ====="
echo "DATA=$D"

for W in 0.0 0.25 0.5 1.0 2.0; do
    echo ""
    echo "--- w=${W} ---"
    CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $IN_DIST --enhanced_dir enhanced/e2-static-w${W}/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale $W
    CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $OOD_ALL --enhanced_dir enhanced/e2-static-w${W}/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale $W
done

echo ""
echo "--- Unconditional (w=-1) ---"
CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $IN_DIST --enhanced_dir enhanced/e2-static-uncond/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale -1.0
CUDA_VISIBLE_DEVICES=$GPU python enhancement.py --test_dir $OOD_ALL --enhanced_dir enhanced/e2-static-uncond/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale -1.0

echo ""
echo "===== Metrics ====="
for W in 0.0 0.25 0.5 1.0 2.0; do
    python calc_metrics.py --test_dir $IN_DIST --enhanced_dir enhanced/e2-static-w${W}/in_dist
    python calc_metrics.py --test_dir $OOD_ALL --enhanced_dir enhanced/e2-static-w${W}/ood_esc50_all
done
python calc_metrics.py --test_dir $IN_DIST --enhanced_dir enhanced/e2-static-uncond/in_dist
python calc_metrics.py --test_dir $OOD_ALL --enhanced_dir enhanced/e2-static-uncond/ood_esc50_all

echo ""
python scripts/summarize_results.py enhanced/e2-static-w0.0 enhanced/e2-static-w0.25 enhanced/e2-static-w0.5 enhanced/e2-static-w1.0 enhanced/e2-static-w2.0 enhanced/e2-static-uncond
echo "===== Done ====="
