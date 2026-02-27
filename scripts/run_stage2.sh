#!/bin/bash
# Stage 2: Train score network with frozen encoder pipeline from Stage 1
#
# Usage: bash scripts/run_stage2.sh <stage1_ckpt_path>
# Example: bash scripts/run_stage2.sh logs/stage1/best-epoch=15-val_nc_acc=0.950.ckpt

STAGE1_CKPT=${1:?Usage: bash scripts/run_stage2.sh <stage1_ckpt_path>}

source /home/nas4_user/kyudanjung/anaconda3/etc/profile.d/conda.sh
conda activate sgmse
cd /home/nas4_user/kyudanjung/seokhoonmoon/NASE

python train.py \
    --backbone ncsnpp --sde ouve \
    --encoder_type wavlm --multi_degradation \
    --pretrain_class_model dummy \
    --base_dir /home/nas4_user/kyudanjung/seokhoonmoon/data/multi_degradation_16k \
    --gpus 8 --batch_size 4 --p_uncond 0.1 \
    --num_eval_files 10 \
    --stage1_ckpt "$STAGE1_CKPT" \
    --wandb_name stage2-frozen-encoder \
    --max_epochs 160 \
    2>&1 | tee logs/stage2_train.log
