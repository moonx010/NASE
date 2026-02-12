#!/usr/bin/env bash
# ============================================================
# E2 Full Training: NASE + CFG (p_uncond=0.2), 160 epochs
# Run on 159-145 (8x RTX 3090)
#
# Usage: cd NASE && bash scripts/train_e2_full.sh
# ============================================================

BASE_DIR="/home/nas4_user/kyudanjung/seokhoonmoon/data/voicebank-demand-16k"
BEATS_CKPT="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"

echo "===== E2 Full Training: NASE + CFG p_uncond=0.2, 160 epochs ====="
echo "GPUs: 8x RTX 3090 (159-145)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --backbone ncsnpp --sde ouve --base_dir ${BASE_DIR} --gpus 8 --batch_size 4 --lr 1e-4 --ema_decay 0.999 --num_eval_files 20 --pretrain_class_model ${BEATS_CKPT} --inject_type addition --max_epochs 160 --p_uncond 0.2 --wandb_name e2-full-cfg-p0.2
