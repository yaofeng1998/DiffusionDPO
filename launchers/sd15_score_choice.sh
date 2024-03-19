export MODEL_NAME="/home/huayu/stable-diffusion-v1-5"
export DATASET_NAME="/home/huayu/pickapic_v2"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 accelerate launch --num_processes 8 --main_process_port 6002 train.py  \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=32 \
#   --max_train_steps=2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --checkpointing_steps 500 \
#   --beta_dpo 1.0 \
#   --weight_temp 1000.0 \
#   --train_method "score_dpo" \
  # --choice_model "pickscore" \ 
#   --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
#   --output_dir="tmp-sd15-scoredpo_b1w1000aligned" > tmp-sd15-scoredpo_b1w1000aligned.txt 2>&1 &


# g40


# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 6010 train.py  \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=128 \
#   --max_train_steps=2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --checkpointing_steps 500 \
#   --beta_dpo 100.0 \
#   --weight_temp 1000.0 \
#   --train_method "score_dpo" \
#   --choice_model "pickscore" \
#   --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
#   --output_dir="tmp-sd15-scoredpo_b100w1000aligned" > tmp-sd15-scoredpo_b100w1000aligned.txt 2>&1 &


# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 6011 train.py  \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=128 \
#   --max_train_steps=2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --checkpointing_steps 500 \
#   --beta_dpo 10.0 \
#   --weight_temp 1000.0 \
#   --train_method "score_dpo" \
#   --choice_model "pickscore" \
#   --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
#   --output_dir="tmp-sd15-scoredpo_b10w1000aligned" > tmp-sd15-scoredpo_b10w1000aligned.txt 2>&1 &

# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 6012 train.py  \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=128 \
#   --max_train_steps=2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --checkpointing_steps 500 \
#   --beta_dpo 1.0 \
#   --weight_temp 1000.0 \
#   --train_method "score_dpo" \
#   --choice_model "pickscore" \
#   --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
#   --output_dir="tmp-sd15-scoredpo_b1w1000aligned0" > tmp-sd15-scoredpo_b1w1000aligned0.txt 2>&1 &

# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 6013 train.py  \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=128 \
#   --max_train_steps=2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --checkpointing_steps 500 \
#   --beta_dpo 0.1 \
#   --weight_temp 1000.0 \
#   --train_method "score_dpo" \
#   --choice_model "pickscore" \
#   --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
#   --output_dir="tmp-sd15-scoredpo_b01w1000aligned" > tmp-sd15-scoredpo_b01w1000aligned.txt 2>&1 &

# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=8,9 accelerate launch --num_processes 2 --main_process_port 6014 train.py  \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=128 \
#   --max_train_steps=2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --checkpointing_steps 500 \
#   --beta_dpo 0.01 \
#   --weight_temp 1000.0 \
#   --train_method "score_dpo" \
#   --choice_model "pickscore" \
#   --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
#   --output_dir="tmp-sd15-scoredpo_b001w1000aligned" > tmp-sd15-scoredpo_b001w1000aligned.txt 2>&1 &








# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 6011 train.py  \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=128 \
#   --max_train_steps=2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --checkpointing_steps 500 \
#   --beta_dpo 0.01 \
#   --weight_temp 10000.0 \
#   --train_method "score_dpo" \
#   --choice_model "pickscore" \
#   --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
#   --output_dir="tmp-sd15-scoredpo_b001w10000aligned" > tmp-sd15-scoredpo_b001w10000aligned.txt 2>&1 &

# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 6012 train.py  \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=128 \
#   --max_train_steps=2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --checkpointing_steps 500 \
#   --beta_dpo 100.0 \
#   --weight_temp 10000.0 \
#   --train_method "score_dpo" \
#   --choice_model "pickscore" \
#   --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
#   --output_dir="tmp-sd15-scoredpo_b100w10000aligned" > tmp-sd15-scoredpo_b100w10000aligned.txt 2>&1 &

# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 6013 train.py  \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=128 \
#   --max_train_steps=2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --checkpointing_steps 500 \
#   --beta_dpo 100.0 \
#   --weight_temp 100.0 \
#   --train_method "score_dpo" \
#   --choice_model "pickscore" \
#   --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
#   --output_dir="tmp-sd15-scoredpo_b100w100aligned" > tmp-sd15-scoredpo_b100w100aligned.txt 2>&1 &

# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=8,9 accelerate launch --num_processes 2 --main_process_port 6014 train.py  \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=128 \
#   --max_train_steps=2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --checkpointing_steps 500 \
#   --beta_dpo 0.01 \
#   --weight_temp 100.0 \
#   --train_method "score_dpo" \
#   --choice_model "pickscore" \
#   --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
#   --output_dir="tmp-sd15-scoredpo_b001w100aligned" > tmp-sd15-scoredpo_b001w100aligned.txt 2>&1 &













  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 6012 train.py  \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 500 \
  --beta_dpo 1.0 \
  --weight_temp 10000.0 \
  --train_method "score_dpo" \
  --choice_model "pickscore" \
  --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
  --output_dir="tmp-sd15-scoredpo_b1w10000aligned" > tmp-sd15-scoredpo_b1w10000aligned.txt 2>&1 &

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 6013 train.py  \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 500 \
  --beta_dpo 1.0 \
  --weight_temp 100.0 \
  --train_method "score_dpo" \
  --choice_model "pickscore" \
  --unet_init "/home/huayu/git/DiffusionDPO/sd15-fullsft/checkpoint-500" \
  --output_dir="tmp-sd15-scoredpo_b1w100aligned" > tmp-sd15-scoredpo_b1w100aligned.txt 2>&1 &
