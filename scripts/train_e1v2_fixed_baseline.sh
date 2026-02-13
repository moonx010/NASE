#!/usr/bin/env bash
# ============================================================
# E1-v2: NASE Baseline with fixed NC loss (sigmoid→logit→CE)
# - 8 GPU × batch 4 = effective batch 32 (matches paper)
# - 160 epochs (full training)
# - Server: 159-145 (GPU 0-7)
# ============================================================

BASE_DIR="/home/nas4_user/kyudanjung/seokhoonmoon/data/voicebank-demand-16k"
BEATS_CKPT="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --backbone ncsnpp --sde ouve --base_dir ${BASE_DIR} --gpus 8 --batch_size 4 --lr 1e-4 --ema_decay 0.999 --num_eval_files 20 --pretrain_class_model ${BEATS_CKPT} --inject_type addition --max_epochs 160 --wandb_name e1v2-fixed-full
