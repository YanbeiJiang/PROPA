#!/bin/bash

base_output_dir="output_SFT_RL/InternVL2_5-2B"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model OpenGVLab/InternVL2_5-2B \
    --resume_from_checkpoint ../Long_SFT/output_SFT/InternVL2_5-2B/v1-20251011-235247/checkpoint-1500 \
    --resume_only_model true \
    --reward_funcs external_r1v_acc format \
    --reward_weights 0.9 0.1 \
    --train_type lora \
    --use_vllm true \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --dataset ../Trance_train_ORM.json \
    --split_dataset_ratio 0 \
    --max_length 16384 \
    --max_completion_length 1024 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --save_strategy 'epoch' \
    --eval_strategy 'epoch' \
    --save_total_limit 5 \
    --logging_steps 5 \
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
    --ddp_find_unused_parameters false \
    --gc_collect_after_offload true \
    --offload_model true \
    --vllm_max_model_len 16384 \
    --vllm_gpu_memory_utilization 0.5 \
    --output_dir $base_output_dir \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --vllm_limit_mm_per_prompt '{"image": 2}' \

if [ -d "$base_output_dir" ]; then
    subfolder1=$(find "$base_output_dir" -maxdepth 1 -type d -name "v*" | sort -V | tail -1)
    
    checkpoint_dir="$subfolder1"
fi
./Clevr_test_intern_SFT_RL.sh "${checkpoint_dir}"