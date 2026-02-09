#!/usr/bin/env bash
# ============================================================
# E2: NASE + CFG Training
# - BEATs + NC loss + Classifier-Free Guidance dropout
# - p_uncond=0.2: 20% 확률로 noise embedding을 zero out
# - 나머지 설정은 E1과 동일
# ============================================================

BASE_DIR="/home/nas4_user/kyudanjung/seokhoonmoon/data/voicebank-demand-16k"
BEATS_CKPT="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --backbone ncsnpp --sde ouve --base_dir ${BASE_DIR} --gpus 8 --batch_size 4 --lr 1e-4 --ema_decay 0.999 --num_eval_files 20 --pretrain_class_model ${BEATS_CKPT} --inject_type addition --max_epochs 160 --p_uncond 0.2 --wandb_name e2-nase-cfg-p0.2
