#!/bin/bash
export MODEL_NAME="/home/huayu/stable-diffusion-v1-5"
export DATASET_NAME="/home/huayu/pickapic_v2"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 6001 train.py  \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=2 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 500 \
  --beta_dpo 2000 \
   --output_dir="tmp-sd15-fulltest-dpo_latest22-0319" > sddpo0319.txt 2>&1 &

