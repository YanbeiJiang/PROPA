#!/bin/bash

# Set initial parameters

CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ../checkpoints_grpo/intern_checkpoint_test/rm_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 40000 \
  --seed 42 \
  --gpu_memory_utilization 0.4 \
  --max_model_len 16384 \
  --use_hf true \
  > ../logs/vllm_server_40000_test.log 2>&1 &
  
CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ../checkpoints_grpo/intern_checkpoint_test/rm_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 40001 \
  --seed 42 \
  --gpu_memory_utilization 0.4 \
  --max_model_len 16384 \
  --use_hf true \
  > ../logs/vllm_server_40001_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ../checkpoints_grpo/intern_checkpoint_test/rm_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 40002 \
  --seed 42 \
  --gpu_memory_utilization 0.4 \
  --max_model_len 16384 \
  --use_hf true \
  > ../logs/vllm_server_40002_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 MAX_PIXELS=1003520 nohup swift deploy \
  --adapters ../checkpoints_grpo/intern_checkpoint_test/rm_model \
  --infer_backend pt \
  --host 0.0.0.0 \
  --port 40003 \
  --seed 42 \
  --gpu_memory_utilization 0.4 \
  --max_model_len 16384 \
  --use_hf true \
  > ../logs/vllm_server_40003_test.log 2>&1 &
