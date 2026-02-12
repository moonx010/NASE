#!/usr/bin/env bash
# ============================================================
# Evaluate Distance-based Adaptive Guidance
# Uses E2-fast CFG checkpoint (trained with p_uncond=0.2)
#
# Step 1: Extract training embeddings (once)
# Step 2: Run knn / prototype adaptive guidance on in-dist + OOD
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
    python scripts/extract_train_embeddings.py --ckpt $CKPT --pretrain_class_model $B --data_dir $D/voicebank-demand-16k --encoder_type beats --output_dir $EMBED_DIR
else
    echo ""
    echo "--- [0] Training embeddings already exist, skipping ---"
fi

REF_ALL="$EMBED_DIR/beats_train_all.pt"
REF_PROTO="$EMBED_DIR/beats_train_prototypes.pt"

# ---- Step 2: k-NN distance adaptive ----

echo ""
echo "--- [1] In-dist, knn adaptive (k=$K, tau=$TAU) ---"
python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-dist-knn/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_ALL --k_neighbors $K --tau $TAU
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-dist-knn/in_dist

echo ""
echo "--- [2] OOD all, knn adaptive (k=$K, tau=$TAU) ---"
if [ -d "$D/ood_test/esc50/all/noisy" ]; then
    python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-dist-knn/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_ALL --k_neighbors $K --tau $TAU
    python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-dist-knn/ood_esc50_all
fi

echo ""
echo "--- [3] OOD stationary, knn adaptive ---"
if [ -d "$D/ood_test/esc50/stationary/noisy" ]; then
    python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-dist-knn/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_ALL --k_neighbors $K --tau $TAU
    python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-dist-knn/ood_stationary
fi

echo ""
echo "--- [4] OOD non-stationary, knn adaptive ---"
if [ -d "$D/ood_test/esc50/non_stationary/noisy" ]; then
    python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-dist-knn/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method knn --ref_embeddings $REF_ALL --k_neighbors $K --tau $TAU
    python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-dist-knn/ood_non_stationary
fi

# ---- Step 3: Prototype distance adaptive ----

echo ""
echo "--- [5] In-dist, prototype adaptive (tau=$TAU) ---"
python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-dist-proto/in_dist --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method prototype --ref_embeddings $REF_PROTO --tau $TAU
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e3-dist-proto/in_dist

echo ""
echo "--- [6] OOD all, prototype adaptive (tau=$TAU) ---"
if [ -d "$D/ood_test/esc50/all/noisy" ]; then
    python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-dist-proto/ood_esc50_all --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method prototype --ref_embeddings $REF_PROTO --tau $TAU
    python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/e3-dist-proto/ood_esc50_all
fi

echo ""
echo "--- [7] OOD stationary, prototype adaptive ---"
if [ -d "$D/ood_test/esc50/stationary/noisy" ]; then
    python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-dist-proto/ood_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method prototype --ref_embeddings $REF_PROTO --tau $TAU
    python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/e3-dist-proto/ood_stationary
fi

echo ""
echo "--- [8] OOD non-stationary, prototype adaptive ---"
if [ -d "$D/ood_test/esc50/non_stationary/noisy" ]; then
    python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-dist-proto/ood_non_stationary --ckpt $CKPT --pretrain_class_model $B --N $N --adaptive_guidance --distance_method prototype --ref_embeddings $REF_PROTO --tau $TAU
    python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/e3-dist-proto/ood_non_stationary
fi

echo ""
echo "===== Done. Run: python scripts/summarize_results.py ====="
