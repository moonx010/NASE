#!/usr/bin/env bash
# ============================================================
# E2-v2: NASE + CFG with fixed NC loss (sigmoid→logit→CE)
# - 4 GPU × batch 4 = effective batch 16
# - 40 epochs (fast iteration)
# - p_uncond=0.2
# - Server: 159-67 (GPU 1-4)
# ============================================================

BASE_DIR="/home/nas4_user/kyudanjung/seokhoonmoon/data/voicebank-demand-16k"
BEATS_CKPT="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"

CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py --backbone ncsnpp --sde ouve --base_dir ${BASE_DIR} --gpus 4 --batch_size 4 --lr 1e-4 --ema_decay 0.999 --num_eval_files 20 --pretrain_class_model ${BEATS_CKPT} --inject_type addition --max_epochs 40 --p_uncond 0.2 --wandb_name e2v2-fixed-cfg-fast
