#!/bin/bash


max_epoch=7

for epoch in $(seq 0 $((max_epoch-1))); do
    python create_RM_train_qwen_grpo.py --epoch $epoch

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MAX_PIXELS=1003520 \
    NPROC_PER_NODE=4 \
    swift sft \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --train_type lora \
        --lora_rank 8 \
        --lora_alpha 32 \
        --dataset "./rm_data_qwen_grpo/rm_data_qwen_epoch${epoch}.json" \
        --split_dataset_ratio 0 \
        --torch_dtype bfloat16 \
        --num_train_epochs 5 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --learning_rate 1e-4 \
        --target_modules all-linear \
        --gradient_accumulation_steps 1 \
        --save_strategy epoch \
        --save_total_limit 5 \
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
    # base_output_dir="./output_rm_grpo/output_rm_epoch${epoch}"

    # if [ -d "$base_output_dir" ]; then
    #     subfolder1=$(find "$base_output_dir" -maxdepth 1 -type d -name "v*" | sort -V | tail -1)
    #     if [ -n "$subfolder1" ]; then
    #         # Find the checkpoint directory (subfolder2) within subfolder1
    #         subfolder2=$(find "$subfolder1" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
    #         if [ -n "$subfolder2" ]; then
    #             checkpoint_dir="$subfolder2"
    #         fi
    #     fi
    # fi

    # # cp $checkpoint_dir/model.safetensors /data/projects/punim1996/Data/AVR-RL/RL_PRM_qwen_lora/checkpoints/qwen_checkpoint/rm_model
    # # cp $checkpoint_dir/config.json /data/projects/punim1996/Data/AVR-RL/RL_PRM_qwen_lora/checkpoints/qwen_checkpoint/rm_model
    # # cp $checkpoint_dir/generation_config.json /data/projects/punim1996/Data/AVR-RL/RL_PRM_qwen_lora/checkpoints/qwen_checkpoint/rm_model
        
    # SOURCE_DIR=$checkpoint_dir
    # DEST_DIR="./checkpoints_grpo/qwen_checkpoint/rm_model"

    # # Delete all contents in the destination directory
    # rm -rf "${DEST_DIR:?}/"*

    # # Copy all contents from the source directory to the destination directory
    # cp -r "${SOURCE_DIR}/"* "$DEST_DIR"
    # sleep 10

    echo "Completed epoch $epoch"

done
