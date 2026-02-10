#!/usr/bin/env bash
# ============================================================
# Run full evaluation on all trained models
# Usage: cd NASE && bash scripts/run_eval.sh
# ============================================================
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
B="$D/BEATs_iter3_plus_AS2M.pt"
N=50

echo "===== E1-fast baseline (epoch=32, pesq=2.78) ====="
python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e1-fast-baseline/in_dist --ckpt "logs/e1-fast-baseline/epoch=32-pesq=2.78.ckpt" --pretrain_class_model $B --N $N
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e1-fast-baseline/in_dist

echo "===== E1-full baseline (epoch=92, pesq=2.74) ====="
python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e1-full-baseline/in_dist --ckpt "logs/sqnju2ez/epoch=92-pesq=2.74.ckpt" --pretrain_class_model $B --N $N
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e1-full-baseline/in_dist

echo "===== E2-fast CFG p=0.2 (epoch=32, pesq=2.71) ====="
python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e2-fast-cfg/in_dist --ckpt "logs/dx2ds38e/epoch=32-pesq=2.71.ckpt" --pretrain_class_model $B --N $N
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e2-fast-cfg/in_dist

echo "===== E2-fast CFG w=1.0 (same ckpt, with guidance) ====="
python enhancement.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e2-fast-cfg-w1.0/in_dist --ckpt "logs/dx2ds38e/epoch=32-pesq=2.71.ckpt" --pretrain_class_model $B --N $N --guidance_scale 1.0
python calc_metrics.py --test_dir $D/voicebank-demand-16k/test --enhanced_dir enhanced/e2-fast-cfg-w1.0/in_dist

echo ""
echo "===== OOD Evaluation (if ood_test exists) ====="
if [ -d "$D/ood_test/esc50/all/noisy" ]; then
    for MODEL_DIR in e1-fast-baseline e1-full-baseline e2-fast-cfg e2-fast-cfg-w1.0; do
        echo "--- OOD: ${MODEL_DIR} ---"
        if [ "$MODEL_DIR" = "e1-fast-baseline" ]; then
            CKPT="logs/e1-fast-baseline/epoch=32-pesq=2.78.ckpt"
            GS=""
        elif [ "$MODEL_DIR" = "e1-full-baseline" ]; then
            CKPT="logs/sqnju2ez/epoch=92-pesq=2.74.ckpt"
            GS=""
        elif [ "$MODEL_DIR" = "e2-fast-cfg" ]; then
            CKPT="logs/dx2ds38e/epoch=32-pesq=2.71.ckpt"
            GS=""
        elif [ "$MODEL_DIR" = "e2-fast-cfg-w1.0" ]; then
            CKPT="logs/dx2ds38e/epoch=32-pesq=2.71.ckpt"
            GS="--guidance_scale 1.0"
        fi

        python enhancement.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/${MODEL_DIR}/ood_all --ckpt "$CKPT" --pretrain_class_model $B --N $N $GS
        python calc_metrics.py --test_dir $D/ood_test/esc50/all --enhanced_dir enhanced/${MODEL_DIR}/ood_all

        if [ -d "$D/ood_test/esc50/stationary/noisy" ]; then
            python enhancement.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/${MODEL_DIR}/ood_stat --ckpt "$CKPT" --pretrain_class_model $B --N $N $GS
            python calc_metrics.py --test_dir $D/ood_test/esc50/stationary --enhanced_dir enhanced/${MODEL_DIR}/ood_stat
        fi

        if [ -d "$D/ood_test/esc50/non_stationary/noisy" ]; then
            python enhancement.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/${MODEL_DIR}/ood_nonstat --ckpt "$CKPT" --pretrain_class_model $B --N $N $GS
            python calc_metrics.py --test_dir $D/ood_test/esc50/non_stationary --enhanced_dir enhanced/${MODEL_DIR}/ood_nonstat
        fi
    done
else
    echo "[SKIP] Run 'bash scripts/create_ood_esc50.sh' first"
fi

echo ""
echo "===== All evaluations complete ====="
