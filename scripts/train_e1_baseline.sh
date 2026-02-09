#!/usr/bin/env bash
# ============================================================
# E1: NASE Baseline Training
# - Original NASE: BEATs + NC loss, no CFG
# - inject_type: addition
# - Expected: PESQ > 2.9 on VoiceBank-DEMAND
# ============================================================

# ---- 경로 설정 (서버 환경에 맞게 수정) ----
BASE_DIR="/path/to/voicebank-demand-16k"       # 16kHz 리샘플링된 데이터
BEATS_CKPT="/path/to/BEATs_iter3_plus_AS2M.pt" # BEATs 체크포인트

# ---- wandb 설정 ----
export WANDB_PROJECT="nase-adaptive-guidance"
export WANDB_RUN_GROUP="e1-baseline"
export WANDB_NAME="e1-nase-baseline-addition"
export WANDB_TAGS="baseline,nase,addition,e1"

# ---- 학습 ----
python train.py \
    --backbone ncsnpp \
    --sde ouve \
    --base_dir ${BASE_DIR} \
    --gpus 2 \
    --batch_size 8 \
    --lr 1e-4 \
    --ema_decay 0.999 \
    --num_eval_files 20 \
    --pretrain_class_model ${BEATS_CKPT} \
    --inject_type addition \
    --max_epochs 160
