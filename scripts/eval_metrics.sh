#!/usr/bin/env bash
# ============================================================
# Evaluation: Compute PESQ, ESTOI, SI-SDR metrics
# Usage: bash scripts/eval_metrics.sh <enhanced_dir> <test_dir>
# Example: bash scripts/eval_metrics.sh ./enhanced/e1_baseline /path/to/data-16k/test
# ============================================================

ENHANCED_DIR=${1:?"Usage: $0 <enhanced_dir> <test_dir>"}
TEST_DIR=${2:?"Usage: $0 <enhanced_dir> <test_dir>"}

python calc_metrics.py \
    --test_dir ${TEST_DIR} \
    --enhanced_dir ${ENHANCED_DIR}
