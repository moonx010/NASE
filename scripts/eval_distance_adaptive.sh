#!/usr/bin/env bash
# ============================================================
# Evaluate Distance-based Adaptive Guidance (GPU 4,5 parallel)
# Uses E2-fast CFG checkpoint (trained with p_uncond=0.2)
#
# Usage: cd NASE && bash scripts/eval_distance_adaptive.sh
# ============================================================
set -e

CKPT="logs/dx2ds38e/last.ckpt"
B="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
N=50
K=10
TAU=1.0
EMBED_DIR="embeddings"

echo "===== Distance-based Adaptive Guidance Evaluation ====="
echo "CKPT=$CKPT"
echo "K=$K, TAU=$TAU, N=$N"

# ---- Step 1: Extract training embeddings (skip if already done) ----
if [ ! -f "$EMBED_DIR/beats_train_all.pt" ]; then
    echo ""
    echo "--- [0] Extracting training embeddings ---"
    CUDA_VISIBLE_DEVICES=4 python scripts/extract_train_embeddings.py --ckpt $CKPT --pretrain_class_model $B --data_dir $D/voicebank-demand-16k --encoder_type beats --output_dir $EMBED_DIR
else
    echo ""
    echo "--- [0] Training embeddings already exist, skipping ---"
fi

REF_ALL="$EMBED_DIR/beats_train_all.pt"
REF_PROTO="$EMBED_DIR/beats_train_prototypes.pt"

# ---- Step 2: knn adaptive (GPU 4 = in-dist, GPU 5 = OOD 3개 순차) ----
echo ""
echo "===== Round 1: k-NN distance adaptive ====="

CUDA_VISIBLE_DEVICES=4 python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-dist-knn/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_ALL --k_neighbors $K --tau $TAU &
PID0=$!

(
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-dist-knn/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_ALL --k_neighbors $K --tau $TAU && \
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-dist-knn/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_ALL --k_neighbors $K --tau $TAU && \
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-dist-knn/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_ALL --k_neighbors $K --tau $TAU
) &
PID1=$!

echo "Round 1: GPU 4 (knn in-dist), GPU 5 (knn OOD x3). Waiting..."
wait $PID0 $PID1

# ---- Step 3: prototype adaptive (GPU 4 = in-dist, GPU 5 = OOD 3개 순차) ----
echo ""
echo "===== Round 2: Prototype distance adaptive ====="

CUDA_VISIBLE_DEVICES=4 python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-dist-proto/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method prototype --ref_embeddings $REF_PROTO --tau $TAU &
PID2=$!

(
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-dist-proto/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method prototype --ref_embeddings $REF_PROTO --tau $TAU && \
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-dist-proto/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method prototype --ref_embeddings $REF_PROTO --tau $TAU && \
CUDA_VISIBLE_DEVICES=5 python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-dist-proto/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method prototype --ref_embeddings $REF_PROTO --tau $TAU
) &
PID3=$!

echo "Round 2: GPU 4 (proto in-dist), GPU 5 (proto OOD x3). Waiting..."
wait $PID2 $PID3

# ---- Step 4: Calc metrics ----
echo ""
echo "===== Calculating metrics ====="
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-dist-knn/in_dist
python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-dist-knn/ood_esc50_all
python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-dist-knn/ood_stationary
python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-dist-knn/ood_non_stationary
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-dist-proto/in_dist
python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-dist-proto/ood_esc50_all
python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-dist-proto/ood_stationary
python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-dist-proto/ood_non_stationary

echo ""
python scripts/summarize_results.py enhanced/e3-dist-knn enhanced/e3-dist-proto
echo "===== Done ====="
