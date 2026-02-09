#!/usr/bin/env bash
# ============================================================
# E2: NASE + CFG Training
# - BEATs + NC loss + Classifier-Free Guidance dropout
# - p_uncond=0.2: 20% 확률로 noise embedding을 zero out
# - inject_type: addition
# - 이 실험은 E1 baseline과 동일한 설정 + p_uncond만 추가
# ============================================================

# ---- 경로 설정 (서버 환경에 맞게 수정) ----
BASE_DIR="/path/to/voicebank-demand-16k"
BEATS_CKPT="/path/to/BEATs_iter3_plus_AS2M.pt"

# ---- wandb 설정 ----
export WANDB_PROJECT="nase-adaptive-guidance"
export WANDB_RUN_GROUP="e2-cfg"
export WANDB_NAME="e2-nase-cfg-p0.2"
export WANDB_TAGS="cfg,nase,p_uncond=0.2,e2"

# ---- 학습 ----
# NOTE: p_uncond 인자는 model.py 수정 후 사용 가능
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
    --p_uncond 0.2 \
    --max_epochs 160
