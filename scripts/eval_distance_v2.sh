#!/usr/bin/env bash
# ============================================================
# Distance-based Adaptive Guidance v2 — L2 normalized + w∈[-1,1]
# Tests both postcnn (256-dim) and raw BEATs (768-dim) embeddings
#
# Key fixes over v1:
#   1. L2 normalization → meaningful distances
#   2. w = 2/(1+d/τ) - 1 → w∈[-1,1], OOD gets negative w (uncond fallback)
#   3. Raw BEATs features as alternative to post_cnn
#
# Usage: cd NASE && bash scripts/eval_distance_v2.sh <GPU_A> <GPU_B>
# Example: bash scripts/eval_distance_v2.sh 4 5
# ============================================================
set -e

GPU_A=${1:-4}
GPU_B=${2:-5}
CKPT="logs/dx2ds38e/last.ckpt"
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50
K=10
TAU=1.0
EMBED_DIR="embeddings"

echo "===== Distance-based Adaptive Guidance v2 ====="
echo "GPU_A=$GPU_A, GPU_B=$GPU_B, K=$K, TAU=$TAU, N=$N"

IN_DIST="$D/voicebank-demand-16k/test"
OOD_ALL="$D/ood_test/esc50/all"
OOD_STAT="$D/ood_test/esc50/stationary"
OOD_NONSTAT="$D/ood_test/esc50/non_stationary"

# ---- Step 1: Extract embeddings (both postcnn and raw) ----
echo ""
echo "--- [0] Extracting training embeddings ---"

# postcnn (256-dim, L2 norm)
if [ ! -f "$EMBED_DIR/beats_postcnn_train_all.pt" ]; then
    echo "  Extracting postcnn embeddings..."
    CUDA_VISIBLE_DEVICES=$GPU_A python scripts/extract_train_embeddings.py --ckpt $CKPT --pretrain_class_model $B --data_dir $D/voicebank-demand-16k --encoder_type beats --output_dir $EMBED_DIR
else
    echo "  postcnn embeddings already exist, skipping"
fi

# raw (768-dim, L2 norm)
if [ ! -f "$EMBED_DIR/beats_raw_train_all.pt" ]; then
    echo "  Extracting raw BEATs embeddings..."
    CUDA_VISIBLE_DEVICES=$GPU_A python scripts/extract_train_embeddings.py --ckpt $CKPT --pretrain_class_model $B --data_dir $D/voicebank-demand-16k --encoder_type beats --output_dir $EMBED_DIR --use_raw
else
    echo "  raw embeddings already exist, skipping"
fi

REF_POSTCNN="$EMBED_DIR/beats_postcnn_train_all.pt"
REF_RAW="$EMBED_DIR/beats_raw_train_all.pt"
PROTO_POSTCNN="$EMBED_DIR/beats_postcnn_train_prototypes.pt"
PROTO_RAW="$EMBED_DIR/beats_raw_train_prototypes.pt"

# ---- Step 2: kNN postcnn (GPU_A=in-dist, GPU_B=OOD) ----
echo ""
echo "===== Round 1: kNN + postcnn (L2 norm) ====="
CUDA_VISIBLE_DEVICES=$GPU_A python enhancement.py --test_dir $IN_DIST --enhanced_dir enhanced/e3v2-knn-postcnn/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_POSTCNN --k_neighbors $K --tau $TAU &
PID0=$!
(
CUDA_VISIBLE_DEVICES=$GPU_B python enhancement.py --test_dir $OOD_ALL --enhanced_dir enhanced/e3v2-knn-postcnn/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_POSTCNN --k_neighbors $K --tau $TAU && \
CUDA_VISIBLE_DEVICES=$GPU_B python enhancement.py --test_dir $OOD_STAT --enhanced_dir enhanced/e3v2-knn-postcnn/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_POSTCNN --k_neighbors $K --tau $TAU && \
CUDA_VISIBLE_DEVICES=$GPU_B python enhancement.py --test_dir $OOD_NONSTAT --enhanced_dir enhanced/e3v2-knn-postcnn/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_POSTCNN --k_neighbors $K --tau $TAU
) &
PID1=$!
echo "Round 1: GPU_A=$PID0, GPU_B=$PID1. Waiting..."
wait $PID0 $PID1

# ---- Step 3: kNN raw (GPU_A=in-dist, GPU_B=OOD) ----
echo ""
echo "===== Round 2: kNN + raw BEATs ====="
CUDA_VISIBLE_DEVICES=$GPU_A python enhancement.py --test_dir $IN_DIST --enhanced_dir enhanced/e3v2-knn-raw/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_RAW --k_neighbors $K --tau $TAU --use_raw &
PID2=$!
(
CUDA_VISIBLE_DEVICES=$GPU_B python enhancement.py --test_dir $OOD_ALL --enhanced_dir enhanced/e3v2-knn-raw/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_RAW --k_neighbors $K --tau $TAU --use_raw && \
CUDA_VISIBLE_DEVICES=$GPU_B python enhancement.py --test_dir $OOD_STAT --enhanced_dir enhanced/e3v2-knn-raw/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_RAW --k_neighbors $K --tau $TAU --use_raw && \
CUDA_VISIBLE_DEVICES=$GPU_B python enhancement.py --test_dir $OOD_NONSTAT --enhanced_dir enhanced/e3v2-knn-raw/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_RAW --k_neighbors $K --tau $TAU --use_raw
) &
PID3=$!
echo "Round 2: GPU_A=$PID2, GPU_B=$PID3. Waiting..."
wait $PID2 $PID3

# ---- Step 4: Calc metrics ----
echo ""
echo "===== Calculating metrics ====="
for DIR in e3v2-knn-postcnn e3v2-knn-raw; do
    python calc_metrics.py --test_dir $IN_DIST --enhanced_dir enhanced/$DIR/in_dist
    python calc_metrics.py --test_dir $OOD_ALL --enhanced_dir enhanced/$DIR/ood_esc50_all
    python calc_metrics.py --test_dir $OOD_STAT --enhanced_dir enhanced/$DIR/ood_stationary
    python calc_metrics.py --test_dir $OOD_NONSTAT --enhanced_dir enhanced/$DIR/ood_non_stationary
done

echo ""
python scripts/summarize_results.py enhanced/e3v2-knn-postcnn enhanced/e3v2-knn-raw
echo "===== Done ====="
