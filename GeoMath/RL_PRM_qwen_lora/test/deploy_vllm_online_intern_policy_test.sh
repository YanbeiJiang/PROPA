#!/bin/bash

# Set initial parameters

CUDA_VISIBLE_DEVICES=0 nohup swift deploy \
  --adapters ../checkpoints_grpo/qwen_checkpoint_test/policy_model \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 30000 \
  --seed 42 \
  --gpu_memory_utilization 0.45 \
  --max_model_len 18000 \
  --use_hf true \
  > ../logs/vllm_server_30000_test.log 2>&1 &
  
CUDA_VISIBLE_DEVICES=1 nohup swift deploy \
  --adapters ../checkpoints_grpo/qwen_checkpoint_test/policy_model \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 30001 \
  --seed 42 \
  --gpu_memory_utilization 0.45 \
  --max_model_len 18000 \
  --use_hf true \
  > ../logs/vllm_server_30001_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup swift deploy \
  --adapters ../checkpoints_grpo/qwen_checkpoint_test/policy_model \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 30002 \
  --seed 42 \
  --gpu_memory_utilization 0.45 \
  --max_model_len 18000 \
  --use_hf true \
  > ../logs/vllm_server_30002_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup swift deploy \
  --adapters ../checkpoints_grpo/qwen_checkpoint_test/policy_model \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 30003 \
  --seed 42 \
  --gpu_memory_utilization 0.45 \
  --max_model_len 18000 \
  --use_hf true \
  > ../logs/vllm_server_30003_test.log 2>&1 &

