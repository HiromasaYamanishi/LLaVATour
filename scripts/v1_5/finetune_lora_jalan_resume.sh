#!/bin/bash
OUTPUT_DIR=$1
RESUME_DIR=$2
TRAIN_FILE=$3

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 20004 llava/train/train_mem.py\
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-6 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --resume_from_ckpt True\
    --model_path $RESUME_DIR \
    --model_base lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path $TRAIN_FILE \
    --image_folder /home/yamanishi/project/trip_recommend/data/jalan_image_with_caption \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
