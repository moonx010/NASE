#!/usr/bin/env bash
# ============================================================
# WavLM encoder fast training (baseline + CFG)
# - encoder_type=wavlm (pretrained WavLM Base, fine-tuned)
# - 40 epochs for quick comparison with BEATs
# ============================================================

BASE_DIR="/home/nas4_user/kyudanjung/seokhoonmoon/data/voicebank-demand-16k"
BEATS_CKPT="/home/nas4_user/kyudanjung/seokhoonmoon/data/BEATs_iter3_plus_AS2M.pt"

# WavLM baseline (no CFG)
CUDA_VISIBLE_DEVICES=1,2,3,5 python train.py --backbone ncsnpp --sde ouve --base_dir ${BASE_DIR} --gpus 4 --batch_size 4 --lr 1e-4 --ema_decay 0.999 --num_eval_files 20 --pretrain_class_model ${BEATS_CKPT} --inject_type addition --max_epochs 40 --encoder_type wavlm --wandb_name wavlm-fast-baseline

# WavLM + CFG (p_uncond=0.2)
CUDA_VISIBLE_DEVICES=1,2,3,5 python train.py --backbone ncsnpp --sde ouve --base_dir ${BASE_DIR} --gpus 4 --batch_size 4 --lr 1e-4 --ema_decay 0.999 --num_eval_files 20 --pretrain_class_model ${BEATS_CKPT} --inject_type addition --max_epochs 40 --encoder_type wavlm --p_uncond 0.2 --wandb_name wavlm-fast-cfg-p0.2
