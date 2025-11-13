#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
swift sft \
    --model OpenGVLab/InternVL2_5-8B \
    --dataset '../Math_Vision_train_CoT_concate.json' \
    --split_dataset_ratio 0 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --save_total_limit 5 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps 5 \
    --max_length 16384 \
    --output_dir output_concate \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --use_hf true \