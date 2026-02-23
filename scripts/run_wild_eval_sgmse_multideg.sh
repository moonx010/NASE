#!/bin/bash
# Wild Dataset Evaluation — SGMSE+ baseline trained on multi-degradation data
# Run on 145RTX3090
# Usage: bash scripts/run_wild_eval_sgmse_multideg.sh [enhance|metrics|all]

set -e

# === Paths ===
NAS="/home/nas4_user/kyudanjung/seokhoonmoon"
NASE="$NAS/NASE"
PY="/home/nas4_user/kyudanjung/anaconda3/envs/sgmse/bin/python"
CONDA_BIN="/home/nas4_user/kyudanjung/anaconda3/envs/sgmse/bin"
ENV_PREFIX="export PATH=$CONDA_BIN:\$PATH"

# Checkpoint: SGMSE+ baseline trained on multi-deg data (encoder_type=none)
CKPT="$NASE/logs/sgmse-baseline-multideg/epoch=159-last.ckpt"
BEATS="$NAS/data/BEATs_iter3_plus_AS2M.pt"

# Eval output base
EVAL_BASE="$NASE/logs/eval_wild"

# Preprocessed test dirs (already prepared)
VOICES_TEST="$EVAL_BASE/voices_test"
DAPS_TEST="$EVAL_BASE/daps_test"
URGENT_TEST="$EVAL_BASE/urgent_test"

# Output dirs — suffix _sgmse_multideg to distinguish from paper pretrained
VOICES_OUT="$EVAL_BASE/voices_sgmse_multideg"
DAPS_OUT="$EVAL_BASE/daps_sgmse_multideg"
URGENT_OUT="$EVAL_BASE/urgent_sgmse_multideg"

MODE="${1:-all}"

# === Step 1: Enhancement ===
enhance() {
    echo "=== Enhancement: SGMSE+ baseline (multi-deg trained) ==="

    # Verify checkpoint exists
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Checkpoint not found: $CKPT"
        exit 1
    fi
    echo "Checkpoint: $CKPT"

    # --- VOiCES ---
    if [ ! -d "$VOICES_OUT" ] || [ "$(ls $VOICES_OUT/*.wav 2>/dev/null | wc -l)" -lt 10 ]; then
        echo "[VOiCES] Launching on GPU 0..."
        mkdir -p "$VOICES_OUT"
        tmux new-session -d -s wild-voices-sgmsemd \
            "$ENV_PREFIX && cd $NASE && CUDA_VISIBLE_DEVICES=0 $PY enhancement.py \
                --test_dir $VOICES_TEST \
                --enhanced_dir $VOICES_OUT \
                --ckpt $CKPT \
                --pretrain_class_model $BEATS \
                --encoder_type none \
                --N 30 \
            2>&1 | tee $VOICES_OUT/_log.txt; \
            echo 'DONE' >> $VOICES_OUT/_log.txt"
    else
        echo "[VOiCES] Already has $(ls $VOICES_OUT/*.wav 2>/dev/null | wc -l) files"
    fi

    # --- DAPS ---
    if [ ! -d "$DAPS_OUT" ] || [ "$(ls $DAPS_OUT/*.wav 2>/dev/null | wc -l)" -lt 10 ]; then
        echo "[DAPS] Launching on GPU 1..."
        mkdir -p "$DAPS_OUT"
        tmux new-session -d -s wild-daps-sgmsemd \
            "$ENV_PREFIX && cd $NASE && CUDA_VISIBLE_DEVICES=1 $PY enhancement.py \
                --test_dir $DAPS_TEST \
                --enhanced_dir $DAPS_OUT \
                --ckpt $CKPT \
                --pretrain_class_model $BEATS \
                --encoder_type none \
                --N 30 \
            2>&1 | tee $DAPS_OUT/_log.txt; \
            echo 'DONE' >> $DAPS_OUT/_log.txt"
    else
        echo "[DAPS] Already has $(ls $DAPS_OUT/*.wav 2>/dev/null | wc -l) files"
    fi

    # --- URGENT ---
    if [ ! -d "$URGENT_OUT" ] || [ "$(ls $URGENT_OUT/*.wav 2>/dev/null | wc -l)" -lt 10 ]; then
        echo "[URGENT] Launching on GPU 2..."
        mkdir -p "$URGENT_OUT"
        tmux new-session -d -s wild-urgent-sgmsemd \
            "$ENV_PREFIX && cd $NASE && CUDA_VISIBLE_DEVICES=2 $PY enhancement.py \
                --test_dir $URGENT_TEST \
                --enhanced_dir $URGENT_OUT \
                --ckpt $CKPT \
                --pretrain_class_model $BEATS \
                --encoder_type none \
                --N 30 \
            2>&1 | tee $URGENT_OUT/_log.txt; \
            echo 'DONE' >> $URGENT_OUT/_log.txt"
    else
        echo "[URGENT] Already has $(ls $URGENT_OUT/*.wav 2>/dev/null | wc -l) files"
    fi

    echo "=== All enhancement jobs launched ==="
    echo "Monitor with: tmux ls | grep wild"
}

# === Step 2: Metrics ===
metrics() {
    echo "=== Computing Metrics ==="

    for dataset in voices daps urgent; do
        ENHANCED="$EVAL_BASE/${dataset}_sgmse_multideg"
        if [ "$dataset" = "voices" ]; then
            TEST="$VOICES_TEST"
        elif [ "$dataset" = "daps" ]; then
            TEST="$DAPS_TEST"
        else
            TEST="$URGENT_TEST"
        fi

        N_WAV=$(ls "$ENHANCED"/*.wav 2>/dev/null | wc -l)
        if [ "$N_WAV" -gt 0 ] && [ ! -f "$ENHANCED/_results.csv" ]; then
            echo "[$dataset/sgmse_multideg] Computing metrics on $N_WAV files..."
            $PY "$NASE/calc_metrics.py" \
                --test_dir "$TEST" \
                --enhanced_dir "$ENHANCED" \
                --utmos \
            2>&1 | tee "$ENHANCED/_metrics_log.txt"
        elif [ -f "$ENHANCED/_results.csv" ]; then
            echo "[$dataset/sgmse_multideg] Metrics already computed"
        else
            echo "[$dataset/sgmse_multideg] No wav files yet, skipping"
        fi
    done

    echo "=== Metrics done ==="
}

# === Main ===
case "$MODE" in
    enhance) enhance ;;
    metrics) metrics ;;
    all)
        enhance
        echo ""
        echo "Enhancement jobs running in tmux. Check progress:"
        echo "  tmux ls | grep wild"
        echo "  tail -5 $EVAL_BASE/*_sgmse_multideg/_log.txt"
        echo ""
        echo "After enhancement completes, run:"
        echo "  bash scripts/run_wild_eval_sgmse_multideg.sh metrics"
        ;;
    *)
        echo "Usage: $0 [enhance|metrics|all]"
        exit 1
        ;;
esac
