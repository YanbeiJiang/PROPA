#!/bin/bash


base_output_dir="output/Qwen2.5-VL-7B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset '../Math_Vision_train.json' \
    --split_dataset_ratio 0 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --save_total_limit 5 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir $base_output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --use_hf true \
    --acc_strategy seq

if [ -d "$base_output_dir" ]; then
    subfolder1=$(find "$base_output_dir" -maxdepth 1 -type d -name "v*" | sort -V | tail -1)
    
    checkpoint_dir="$subfolder1"
fi

./Clevr_test_qwen_7b.sh "${checkpoint_dir}"