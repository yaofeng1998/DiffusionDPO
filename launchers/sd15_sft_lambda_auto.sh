#!/bin/bash
export MODEL_NAME="$HOME/cephfs-thu/stable-diffusion-v1-5"
export DATASET_NAME="$HOME/cephfs-thu/pickapic_v2"
export CACHE_DIR="$HOME/cephfs-thu/.cache"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps
NUM_PROCESSES=$(($1))
BATCH_SIZE=$(($2))
LAMBDA=$3
TODAY=$(date +'%m%d')
LENGTH_LIMIT=$((2 * $NUM_PROCESSES - 1))
DATALOADER_NUM_WORKERS=$((2 * $NUM_PROCESSES))
GRADIENT_ACCUMULATION_STEPS=$((2048 / $NUM_PROCESSES / $BATCH_SIZE))
GPUs=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
sort -nk 2 -r | awk '$2>30000 {print $1}' | tr -d "\n")
while [ ${#GPUs} -lt $LENGTH_LIMIT ];
do
  echo "No enough GPU memory"
  sleep 5
  GPUs=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  sort -nk 2 -r | awk '$2>30000 {print $1}' | tr -d "\n")
done
GPUs="${GPUs:0:$LENGTH_LIMIT}"
echo "using GPUs ${GPUs}"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPUs} accelerate launch --num_processes $NUM_PROCESSES --main_process_port 6003 train.py  \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=$BATCH_SIZE \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 500 \
  --sft \
  --sft_lambda $LAMBDA \
  --cache_dir=$CACHE_DIR \
   --output_dir="tmp-sd15-sft-lambda-$LAMBDA-$TODAY" > sd15-sft-lambda-$LAMBDA-$TODAY.txt 2>&1 &
