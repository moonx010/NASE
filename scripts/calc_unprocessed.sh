#!/usr/bin/env bash
# ============================================================
# Calculate metrics for unprocessed (noisy) signals
# Copies noisy files as "enhanced" and runs calc_metrics
#
# Usage: cd NASE && bash scripts/calc_unprocessed.sh
# ============================================================
set -e

D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
IN_DIST="$D/voicebank-demand-16k/test"
OOD_ALL="$D/ood_test/esc50/all"
OOD_STAT="$D/ood_test/esc50/stationary"
OOD_NONSTAT="$D/ood_test/esc50/non_stationary"

echo "===== Unprocessed (Noisy) Metrics ====="

# Symlink noisy dirs as enhanced dirs
for TESTNAME in in_dist ood_esc50_all ood_stationary ood_non_stationary; do
    mkdir -p enhanced/unprocessed/$TESTNAME
done

# Copy noisy files (symlinks don't always work across NAS)
echo "Copying noisy files..."
cp $IN_DIST/noisy/*.wav enhanced/unprocessed/in_dist/
cp $OOD_ALL/noisy/*.wav enhanced/unprocessed/ood_esc50_all/
cp $OOD_STAT/noisy/*.wav enhanced/unprocessed/ood_stationary/
cp $OOD_NONSTAT/noisy/*.wav enhanced/unprocessed/ood_non_stationary/

echo "Calculating metrics..."
python calc_metrics.py --test_dir $IN_DIST --enhanced_dir enhanced/unprocessed/in_dist
python calc_metrics.py --test_dir $OOD_ALL --enhanced_dir enhanced/unprocessed/ood_esc50_all
python calc_metrics.py --test_dir $OOD_STAT --enhanced_dir enhanced/unprocessed/ood_stationary
python calc_metrics.py --test_dir $OOD_NONSTAT --enhanced_dir enhanced/unprocessed/ood_non_stationary

echo ""
python scripts/summarize_results.py enhanced/unprocessed
echo "===== Done ====="
