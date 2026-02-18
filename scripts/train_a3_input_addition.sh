#!/usr/bin/env bash
# ============================================================
# A3 Ablation: Input Addition + Multi-Degradation
# Same data, same encoder, same multi-task loss
# Only difference: inject_method=addition (shallow, 1 point)
# vs E2-1 which used inject_method=temb (deep, 37 points)
#
# Run on yj-2 (8x A100-PCIE-40GB)
# Batch: 8 GPU x 4 = 32 (matches E2-1)
# ============================================================

BASE_DIR="/home/kyudanjung/data/multi_degradation_16k"
BEATS_CKPT="/home/kyudanjung/data/BEATs_iter3_plus_AS2M.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --backbone ncsnpp \
    --sde ouve \
    --base_dir ${BASE_DIR} \
    --gpus 8 \
    --batch_size 4 \
    --lr 1e-4 \
    --ema_decay 0.999 \
    --num_eval_files 20 \
    --pretrain_class_model ${BEATS_CKPT} \
    --inject_type addition \
    --encoder_type wavlm \
    --multi_degradation \
    --inject_method addition \
    --p_uncond 0.1 \
    --aux_loss_weight 0.3 \
    --max_epochs 160 \
    --wandb_name a3-input-addition
