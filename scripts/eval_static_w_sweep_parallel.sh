#!/usr/bin/env bash
# ============================================================
# EXP-2: Static w Sweep — PARALLEL version (2 GPUs)
# GPU_A: in-dist (824 files, slow) for w=0.0,0.25,0.5,1.0,2.0
# GPU_B: OOD-all (200 files, fast) for w=0.0,0.25,0.5,1.0,2.0 + uncond
#
# Usage: cd NASE && bash scripts/eval_static_w_sweep_parallel.sh <GPU_A> <GPU_B>
# Example: bash scripts/eval_static_w_sweep_parallel.sh 4 5
# ============================================================
set -e

GPU_A=${1:-4}
GPU_B=${2:-5}
CKPT="logs/dx2ds38e/last.ckpt"
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50

echo "===== EXP-2: Static w Sweep (Parallel) ====="
echo "GPU_A=$GPU_A (in-dist), GPU_B=$GPU_B (OOD), N=$N"

IN_DIST="$D/voicebank-demand-16k/test"
OOD_ALL="$D/ood_test/esc50/all"

# GPU_A: in-dist for all w values + unconditional
(
    for W in 0.0 0.25 0.5 1.0 2.0; do
        echo "[GPU_A] in-dist w=${W}"
        CUDA_VISIBLE_DEVICES=$GPU_A python enhancement.py --test_dir $IN_DIST --enhanced_dir enhanced/e2-static-w${W}/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale $W
    done
    echo "[GPU_A] in-dist unconditional (w=-1)"
    CUDA_VISIBLE_DEVICES=$GPU_A python enhancement.py --test_dir $IN_DIST --enhanced_dir enhanced/e2-static-uncond/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale -1.0
) &
PID_A=$!

# GPU_B: OOD-all for all w values + unconditional
(
    for W in 0.0 0.25 0.5 1.0 2.0; do
        echo "[GPU_B] ood-all w=${W}"
        CUDA_VISIBLE_DEVICES=$GPU_B python enhancement.py --test_dir $OOD_ALL --enhanced_dir enhanced/e2-static-w${W}/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale $W
    done
    echo "[GPU_B] ood-all unconditional (w=-1)"
    CUDA_VISIBLE_DEVICES=$GPU_B python enhancement.py --test_dir $OOD_ALL --enhanced_dir enhanced/e2-static-uncond/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --guidance_scale -1.0
) &
PID_B=$!

echo "Waiting for both GPUs... (GPU_A=$PID_A, GPU_B=$PID_B)"
wait $PID_A $PID_B

# Calc metrics
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
