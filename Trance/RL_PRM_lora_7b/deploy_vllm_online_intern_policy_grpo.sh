#!/bin/bash

# Set initial parameters
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ./checkpoints_grpo/intern_checkpoint/policy_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 40000 \
  --seed 42 \
  --gpu_memory_utilization 0.5 \
  --use_hf true \
  --max_model_len 8192 \
  --limit_mm_per_prompt '{"image": 2}' \
  > logs/vllm_server_10000.log 2>&1 &

# GPU 1, port 8001
CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ./checkpoints_grpo/intern_checkpoint/policy_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 40001 \
  --seed 42 \
  --gpu_memory_utilization 0.5 \
  --use_hf true \
  --max_model_len 8192 \
  --limit_mm_per_prompt '{"image": 2}' \
  > logs/vllm_server_10001.log 2>&1 &

#GPU 2, port 8002
CUDA_VISIBLE_DEVICES=2 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ./checkpoints_grpo/intern_checkpoint/policy_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 40002 \
  --seed 42 \
  --gpu_memory_utilization 0.5 \
  --use_hf true \
  --max_model_len 8192 \
  --limit_mm_per_prompt '{"image": 2}' \
  > logs/vllm_server_10002.log 2>&1 &

# GPU 3, port 8003
CUDA_VISIBLE_DEVICES=3 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ./checkpoints_grpo/intern_checkpoint/policy_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 40004 \
  --seed 42 \
  --gpu_memory_utilization 0.5 \
  --use_hf true \
  --max_model_len 8192 \
  --limit_mm_per_prompt '{"image": 2}' \
  > logs/vllm_server_10003.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=1003520 nohup swift deploy \
#   --adapters ./checkpoints_grpo/intern_checkpoint/policy_model \
#   --infer_backend pt \
#   --host 0.0.0.0 \
#   --port 50000 \
#   --seed 42 \
#   --gpu_memory_utilization 0.5 \
#   --use_hf true \
#   --max_model_len 8192 \
#   --limit_mm_per_prompt '{"image": 2}' \
#   > logs/vllm_server_20000.log 2>&1 &

# # GPU 1, port 8001
# CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=1003520 nohup swift deploy \
#   --adapters ./checkpoints_grpo/intern_checkpoint/policy_model \
#   --infer_backend pt \
#   --host 0.0.0.0 \
#   --port 50001 \
#   --seed 42 \
#   --gpu_memory_utilization 0.5 \
#   --use_hf true \
#   --max_model_len 8192 \
#   --limit_mm_per_prompt '{"image": 2}' \
#   > logs/vllm_server_20001.log 2>&1 &

# # GPU 2, port 8002
# CUDA_VISIBLE_DEVICES=2 MAX_PIXELS=1003520 nohup swift deploy \
#   --adapters ./checkpoints_grpo/intern_checkpoint/policy_model \
#   --infer_backend pt \
#   --host 0.0.0.0 \
#   --port 50002 \
#   --seed 42 \
#   --gpu_memory_utilization 0.5 \
#   --use_hf true \
#   --max_model_len 8192 \
#   --limit_mm_per_prompt '{"image": 2}' \
#   > logs/vllm_server_20002.log 2>&1 &

# # GPU 3, port 8003
# CUDA_VISIBLE_DEVICES=3 MAX_PIXELS=1003520 nohup swift deploy \
#   --adapters ./checkpoints_grpo/intern_checkpoint/policy_model \
#   --infer_backend pt \
#   --host 0.0.0.0 \
#   --port 50003 \
#   --seed 42 \
#   --gpu_memory_utilization 0.5 \
#   --use_hf true \
#   --max_model_len 8192 \
#   --limit_mm_per_prompt '{"image": 2}' \
#   > logs/vllm_server_20003.log 2>&1 &


