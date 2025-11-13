#!/bin/bash

# Set initial parameters

best_checkpoint=""
best_accuracy=0
best_epoch_num=0
for epoch in $(seq 4 8); do
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

    max_batch_num=25
    max_batch_num_prev=$((max_batch_num - 1))
    epoch_prev=$((epoch - 1))
    base_output_dir_policy="../output_grpo/output_grpo_epoch${epoch_prev}_${max_batch_num_prev}"

    if [ -d "$base_output_dir_policy" ]; then
        subfolder1=$(find "$base_output_dir_policy" -maxdepth 1 -type d -name "v*" | sort -V | tail -1)
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
            echo "Error: Could not find version subdirectory in $base_output_dir_policy"
            exit 1
        fi
    else
        echo "Error: Base output directory $base_output_dir_policy does not exist"
        exit 1
    fi

    #checkpoint_dir=../../RL_PRM/output_concate/v4-20250703-154815/checkpoint-5090
    echo "Using checkpoint directory: $checkpoint_dir"

    # Step 1: Run vllm_intern.py
    echo "Step 1: Running run_vllm_intern.py"
    # python save_checkpoints_intern.py --checkpoint $checkpoint_dir --save_dir "/data/projects/punim1996/Data/AVR-RL/RL_PRM/checkpoints/intern_checkpoint_test/policy_model"
    # Delete all contents in the destination directory
    rm -rf "../checkpoints_grpo/intern_checkpoint_test/policy_model/"*

    # Copy all contents from the source directory to the destination directory
    cp -r "${checkpoint_dir}/"* "../checkpoints_grpo/intern_checkpoint_test/policy_model"


    ./deploy_vllm_online_intern_policy_test_greedy.sh
    #./deploy_vllm_online_intern_rm_test.sh
    sleep 120


    python greedy_test_intern.py --input "../../GeoMath_val.json" --output "../test/greedy_result/val_result/greedy_epoch${epoch_prev}.json"

    accuracy_output=$(python get_accuracy.py --output "../test/greedy_result/val_result/greedy_epoch${epoch_prev}.json")
    echo "$accuracy_output"
    
    # 提取accuracy数值 (从 "Accuracy: XX.XX%" 中提取)
    current_accuracy=$(echo "$accuracy_output" | grep -oP 'Accuracy: \K[0-9.]+')
    
    if [ -n "$current_accuracy" ]; then
        
        # 比较并更新最佳checkpoint (使用bc进行浮点数比较)
        if (( $(echo "$current_accuracy > $best_accuracy" | bc -l) )); then
            best_accuracy=$current_accuracy
            best_checkpoint=$checkpoint_dir
            echo ">>> found better checkpoint: $checkpoint_dir (Accuracy: $best_accuracy%)"
        fi
    else
        echo "cannot export accuracy"
    fi

    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    for pid in $PIDS; do
        echo "Killing process $pid"
        kill -9 $pid
    done
done



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



#checkpoint_dir=../../RL_PRM/output_concate/v4-20250703-154815/checkpoint-5090
echo "Using checkpoint directory: $best_checkpoint"

# Step 1: Run vllm_intern.py
echo "Step 1: Running run_vllm_intern.py"
# python save_checkpoints_intern.py --checkpoint $checkpoint_dir --save_dir "/data/projects/punim1996/Data/AVR-RL/RL_PRM/checkpoints/intern_checkpoint_test/policy_model"
# Delete all contents in the destination directory
rm -rf "../checkpoints_grpo/intern_checkpoint_test/policy_model/"*

# Copy all contents from the source directory to the destination directory
cp -r "${best_checkpoint}/"* "../checkpoints_grpo/intern_checkpoint_test/policy_model"


./deploy_vllm_online_intern_policy_test_greedy.sh
#./deploy_vllm_online_intern_rm_test.sh
sleep 120


python greedy_test_intern.py --input "../../GeoMath_test.json" --output "../test/greedy_result/infer_result/greedy_epoch${epoch_prev}.json"

python get_accuracy.py --output "../test/greedy_result/infer_result/greedy_epoch${epoch_prev}.json"

python greedy_test_intern.py --input "../../Geometry_test.json" --output "../test/greedy_result/infer_result/greedy_epoch${epoch_prev}_OOD.json"

python get_accuracy.py --output "../test/greedy_result/infer_result/greedy_epoch${epoch_prev}_OOD.json"

PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
for pid in $PIDS; do
    echo "Killing process $pid"
    kill -9 $pid
done

echo "All testing completed successfully!"