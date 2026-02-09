#!/usr/bin/env bash
# ============================================================
# E1-fast: NASE Baseline (quick iteration)
# - 4 GPU × batch 4 = effective batch 16
# - 40 epochs (~2h)
# - For trend comparison only, NOT final results
# ============================================================

BASE_DIR="/home/nas4_user/kyudanjung/seokhoonmoon/data/voicebank-demand-16k"
BEATS_CKPT="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone ncsnpp --sde ouve --base_dir ${BASE_DIR} --gpus 4 --batch_size 4 --lr 1e-4 --ema_decay 0.999 --num_eval_files 20 --pretrain_class_model ${BEATS_CKPT} --inject_type addition --max_epochs 40 --wandb_name e1-fast-baseline
