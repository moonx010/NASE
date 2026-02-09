#!/usr/bin/env bash
# ============================================================
# E2-fast: NASE + CFG (quick iteration)
# - 4 GPU × batch 4 = effective batch 16
# - 40 epochs (~2h)
# - p_uncond=0.2
# ============================================================

BASE_DIR="/home/nas4_user/kyudanjung/seokhoonmoon/data/voicebank-demand-16k"
BEATS_CKPT="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --backbone ncsnpp --sde ouve --base_dir ${BASE_DIR} --gpus 4 --batch_size 4 --lr 1e-4 --ema_decay 0.999 --num_eval_files 20 --pretrain_class_model ${BEATS_CKPT} --inject_type addition --max_epochs 40 --p_uncond 0.2 --wandb_name e2-fast-cfg-p0.2
