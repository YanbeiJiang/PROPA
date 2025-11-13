#!/bin/bash

# Set initial parameters

CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ../checkpoints_grpo/intern_checkpoint_test/policy_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 30000 \
  --seed 42 \
  --gpu_memory_utilization 0.6 \
  --max_model_len 16384 \
  --use_hf true \
  --limit_mm_per_prompt '{"image": 2}' \
  > ../logs/vllm_server_30000_test.log 2>&1 &
  
CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ../checkpoints_grpo/intern_checkpoint_test/policy_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 30001 \
  --seed 42 \
  --gpu_memory_utilization 0.6 \
  --max_model_len 16384 \
  --use_hf true \
  --limit_mm_per_prompt '{"image": 2}' \
  > ../logs/vllm_server_30001_test.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 MAX_PIXELS=1003520 nohup swift deploy \
#   --adapters ../checkpoints_grpo/intern_checkpoint_test/policy_model \
#   --infer_backend pt \
#   --host 0.0.0.0 \
#   --port 30002 \
#   --seed 42 \
#   --gpu_memory_utilization 0.6 \
#   --max_model_len 16384 \
#   --use_hf true \
#   --limit_mm_per_prompt '{"image": 2}' \
#   > ../logs/vllm_server_30002_test.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 MAX_PIXELS=1003520 nohup swift deploy \
#   --adapters ../checkpoints_grpo/intern_checkpoint_test/policy_model \
#   --infer_backend pt \
#   --host 0.0.0.0 \
#   --port 30003 \
#   --seed 42 \
#   --gpu_memory_utilization 0.6 \
#   --max_model_len 16384 \
#   --use_hf true \
#   --limit_mm_per_prompt '{"image": 2}' \
#   > ../logs/vllm_server_30003_test.log 2>&1 &

