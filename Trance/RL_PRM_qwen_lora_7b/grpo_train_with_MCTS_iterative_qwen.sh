#!/bin/bash

# Set initial parameters
max_epoch=7
max_batch_num=25
batch_size=40
initial_checkpoint_dir='../output_concate/v0-20251003-093053/checkpoint-10368'
#initial_checkpoint_dir='../output_concate/v3-20250806-162955/checkpoint-5090'

# Main epoch loop
for epoch in $(seq 6 $((max_epoch-1))); do
    echo "Starting epoch $epoch"
    # Inner batch loop
    for batch_num in $(seq 0 $((max_batch_num-1))); do
        PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        for pid in $PIDS; do
            echo "Killing process $pid"
            kill -9 $pid
        done

        PORTS=(30000 30001 30002 30003 40000 40001 40002 40003)

        for port in "${PORTS[@]}"; do
            # 查找端口对应的 PID
            pid=$(lsof -ti:$port)
            if [ -n "$pid" ]; then
                echo "Killing process $pid on port $port"
                kill -9 $pid
            else
                echo "No process found on port $port"
            fi
        done
        
        TIME_WAIT_COUNT=$(ss -tan | grep ":$PORT " | grep TIME-WAIT | wc -l)
        if [ "$TIME_WAIT_COUNT" -gt 0 ]; then
            echo "⏳ 检测到 $TIME_WAIT_COUNT 个 TIME_WAIT 连接，等待 $WAIT_TIME 秒..."
            sleep $WAIT_TIME
        fi

        echo "Processing epoch $epoch, batch $batch_num"
        
        # if [ "$epoch" -ne 0 ]; then
        #     ./deploy_vllm_online_qwen_rm.sh
        # fi
        # Set checkpoint directory based on conditions
        if [ $epoch -eq 0 ] && [ $batch_num -eq 0 ]; then
            checkpoint_dir=$initial_checkpoint_dir
        else
            if [ $batch_num -eq 0 ]; then
                epoch_num_prev=$((epoch - 1))
                max_batch_num_prev=$((max_batch_num - 1))
                base_output_dir="../output_grpo/output_grpo_epoch${epoch_num_prev}_${max_batch_num_prev}"
            else
                batch_num_prev=$((batch_num - 1))
                base_output_dir="../output_grpo/output_grpo_epoch${epoch}_${batch_num_prev}"
            fi

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
        fi
        
        echo "Using checkpoint directory: $checkpoint_dir"
        
        # Step 1: Run vllm_qwen.py
        echo "Step 1: Running run_vllm_qwen.py"
        #python save_checkpoints_qwen.py --checkpoint $checkpoint_dir
        # Delete all contents in the destination directory
        rm -rf "../checkpoints_grpo/qwen_checkpoint/policy_model/"*

        # Copy all contents from the source directory to the destination directory
        cp -r "${checkpoint_dir}/"* "../checkpoints_grpo/qwen_checkpoint/policy_model"
        sleep 2

        ./deploy_vllm_online_qwen_policy_grpo.sh

        sleep 120

        # python run_MCTS_train_qwen_grpo.py --epoch $epoch --batch_num $batch_num
        for i in {1..5}; do
            echo "尝试第 $i 次..."
            if python run_MCTS_train_qwen_grpo.py --epoch $epoch --batch_num $batch_num --batch_size $batch_size; then
                echo "执行成功！"
                break
            else
                echo "第 $i 次执行失败"
                if [ $i -eq 5 ]; then
                    echo "已达到最大重试次数(5次)，执行失败"
                    exit 1
                fi
                echo "等待5秒后重试..."
                sleep 5
            fi
        done
        if [ $? -ne 0 ]; then
            echo "Error: run_MCTS_train_qwen.py failed for epoch $epoch, batch $batch_num"
            exit 1
        fi
        
        # Step 2: Run create_GRPO_train_qwen.py
        echo "Step 2: Running create_GRPO_train_qwen.py"
        python create_GRPO_train_qwen.py --epoch $epoch --batch_num $batch_num --batch_size $batch_size
        if [ $? -ne 0 ]; then
            echo "Error: create_GRPO_train_qwen.py failed for epoch $epoch, batch $batch_num"
            exit 1
        fi
        
        PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        for pid in $PIDS; do
            echo "Killing process $pid"
            kill -9 $pid
        done

        # Step 3: Run swift rlhf command
        json_file="../grpo_data_qwen/grpo_data_qwen_epoch${epoch}_${batch_num}.json"

        skip_grpo=false
        if [ -s "$json_file" ] && [ "$(jq 'length' "$json_file")" -gt 0 ]; then
            echo "Step 3: Running swift rlhf training"
            CUDA_VISIBLE_DEVICES=0,1,2,3 \
            NPROC_PER_NODE=3 \
            MAX_PIXELS=1003520 \
            swift rlhf \
                --rlhf_type grpo \
                --model Qwen/Qwen2.5-VL-7B-Instruct \
                --resume_from_checkpoint "$checkpoint_dir" \
                --resume_only_model true \
                --reward_funcs custom_grpo \
                --freeze_vit true \
                --use_vllm true \
                --train_type lora \
                --lora_rank 8 \
                --lora_alpha 32 \
                --seed 42 \
                --torch_dtype bfloat16 \
                --dataset "../grpo_data_qwen/grpo_data_qwen_epoch${epoch}_${batch_num}.json" \
                --split_dataset_ratio 0 \
                --max_length 8192 \
                --max_completion_length 128 \
                --num_train_epochs 1 \
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 4 \
                --learning_rate 1e-6 \
                --gradient_accumulation_steps 1 \
                --save_strategy epoch \
                --eval_strategy epoch \
                --save_total_limit 10 \
                --logging_steps 5 \
                --output_dir "../output_grpo/output_grpo_epoch${epoch}_${batch_num}" \
                --dataloader_num_workers 8 \
                --num_generations 4 \
                --temperature 1.0 \
                --log_completions true \
                --beta 0.001 \
                --max_grad_norm 0.5 \
                --use_hf true \
                --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
                --ddp_find_unused_parameters false \
                --gc_collect_after_offload true \
                --offload_model true \
                --dataset_num_proc 8 \
                --num_infer_workers 1
            if [ $? -ne 0 ]; then
                echo "Error: swift rlhf failed for epoch $epoch, batch $batch_num"
                exit 1
            fi
        else
            skip_grpo=true
            echo "$json_file 空文件或空 JSON，跳过训练"
        fi
        
        python create_SFT_train_qwen_after_grpo.py --epoch $epoch --batch_num $batch_num
        
        json_file="../SFT_after_grpo/SFT_data_qwen_epoch${epoch}_${batch_num}.json"
        if [ -f "$json_file" ]; then
            if [ "$skip_grpo" = true ]; then
                if [ $batch_num -eq 0 ]; then
                    epoch_num_prev=$((epoch - 1))
                    max_batch_num_prev=$((max_batch_num - 1))
                    base_output_dir="../output_grpo/output_grpo_epoch${epoch_num_prev}_${max_batch_num_prev}"
                else
                    prev_batch_num=$((batch_num - 1))
                    base_output_dir="../output_grpo/output_grpo_epoch${epoch}_${prev_batch_num}"
                fi
            else
                base_output_dir="../output_grpo/output_grpo_epoch${epoch}_${batch_num}"
            fi
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

            CUDA_VISIBLE_DEVICES=0,1,2,3 \
            MAX_PIXELS=1003520 \
            swift sft \
                --model Qwen/Qwen2.5-VL-7B-Instruct \
                --resume_from_checkpoint "$checkpoint_dir" \
                --resume_only_model true \
                --dataset "../SFT_after_grpo/SFT_data_qwen_epoch${epoch}_${batch_num}.json" \
                --split_dataset_ratio 0 \
                --train_type lora \
                --lora_rank 8 \
                --lora_alpha 32 \
                --torch_dtype bfloat16 \
                --num_train_epochs 1 \
                --per_device_train_batch_size 1 \
                --per_device_eval_batch_size 1 \
                --learning_rate 1e-4 \
                --target_modules all-linear \
                --freeze_vit true \
                --gradient_accumulation_steps 1 \
                --save_total_limit 5 \
                --save_strategy epoch \
                --eval_strategy epoch \
                --logging_steps 5 \
                --max_length 32000 \
                --output_dir "../output_grpo/output_grpo_epoch${epoch}_${batch_num}" \
                --warmup_ratio 0.05 \
                --dataloader_num_workers 1 \
                --use_hf true
        fi
        echo "Completed epoch $epoch, batch $batch_num"
        mid_batch_num=$(( (max_batch_num - 1) / 2 ))

        # 判断是否需要跳过删除
        if [ "$batch_num" -eq 0 ] || [ "$batch_num" -eq "$mid_batch_num" ]; then
            echo "batch_num = $batch_num, 跳过删除"
        else
            prev_batch_num=$((batch_num - 1))
            target_dir="../output_grpo/output_grpo_epoch${epoch}_${prev_batch_num}"

            if [ -d "$target_dir" ]; then
                echo "删除 $target_dir 下的所有文件和子文件夹..."
                rm -rf "${target_dir:?}/"*
                rm -rf "${target_dir:?}/".*
            else
                echo "目录 $target_dir 不存在, 跳过删除"
            fi
        fi
    done
done

echo "All training completed successfully!"