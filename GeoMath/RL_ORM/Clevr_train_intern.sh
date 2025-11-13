#!/bin/bash


base_output_dir="output/InternVL2_5-2B"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model OpenGVLab/InternVL2_5-2B \
    --reward_funcs external_r1v_acc format \
    --reward_weights 0.9 0.1 \
    --use_vllm true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --dataset '../GeoMath_train_ORM.json' \
    --split_dataset_ratio 0 \
    --max_length 8192 \
    --max_completion_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --save_strategy 'epoch' \
    --eval_strategy 'epoch' \
    --save_total_limit 10 \
    --logging_steps 10 \
    --output_dir $base_output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --num_generations 6 \
    --temperature 1.0 \
    --log_completions true \
    --num_infer_workers 1 \
    --async_generate false \
    --beta 0.001 \
    --max_grad_norm 0.5 \
    --use_hf true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --ddp_find_unused_parameters false \
    --gc_collect_after_offload true \
    --offload_model true \
    --vllm_max_model_len 8192 \
    --vllm_gpu_memory_utilization 0.5 \

if [ -d "$base_output_dir" ]; then
    subfolder1=$(find "$base_output_dir" -maxdepth 1 -type d -name "v*" | sort -V | tail -1)
    if [ -n "$subfolder1" ]; then
        # Find the checkpoint directory (subfolder2) within subfolder1
        subfolder2=$(find "$subfolder1" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
        if [ -n "$subfolder2" ]; then
            checkpoint_dir="$subfolder2"
        else
            echo "Error: Could not find checkpoint subdirectory in $subfolder1"
            exit 1
        fi
    else
        echo "Error: Could not find version subdirectory in $base_output_dir"
        exit 1
    fi
else
    echo "Error: Base output directory $base_output_dir does not exist"
    exit 1
fi

./Clevr_test_intern.sh "${checkpoint_dir}"
