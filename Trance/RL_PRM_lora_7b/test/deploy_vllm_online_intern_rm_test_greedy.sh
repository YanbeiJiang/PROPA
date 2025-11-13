#!/bin/bash

# Set initial parameters

CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ../checkpoints_grpo_greedy/intern_checkpoint_test/rm_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 40000 \
  --seed 42 \
  --gpu_memory_utilization 0.40 \
  --max_model_len 16384 \
  --use_hf true \
  --limit_mm_per_prompt '{"image": 2}' \
  > ../logs/vllm_server_40000_test_greedy.log 2>&1 &

