#!/bin/bash


max_epoch=7

for epoch in $(seq 0 $((max_epoch-1))); do
    python create_RM_train_intern_grpo.py --epoch $epoch

   
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MAX_PIXELS=1003520 \
    NPROC_PER_NODE=4 \
    swift sft \
        --model OpenGVLab/InternVL2_5-8B \
        --train_type lora \
        --lora_rank 8 \
        --lora_alpha 32 \
        --dataset "./rm_data_intern_grpo/rm_data_intern_epoch${epoch}.json" \
        --split_dataset_ratio 0 \
        --torch_dtype bfloat16 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --learning_rate 1e-4 \
        --target_modules all-linear \
        --gradient_accumulation_steps 2 \
        --save_strategy epoch \
        --save_total_limit 2 \
        --logging_steps 10 \
        --max_length 16384 \
        --output_dir "./output_rm_grpo/output_rm_epoch${epoch}" \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 1 \
        --num_labels 1 \
        --task_type seq_cls \
        --use_chat_template false \
        --problem_type regression \
        --use_hf true \
        --seed 42 \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}'
   

    echo "Completed epoch $epoch"

done