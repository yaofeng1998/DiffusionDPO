export MODEL_NAME="/home/huayu/stable-diffusion-v1-5"
export DATASET_NAME="/home/huayu/pickapic_v2"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 accelerate launch --num_processes 8 --main_process_port 6001 train.py  \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=32 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 500 \
  --beta_dpo 2000 \
   --output_dir="tmp-sd15-fulltest-dpo_latest22-0310" > sddpo0310.txt 2>&1 &

