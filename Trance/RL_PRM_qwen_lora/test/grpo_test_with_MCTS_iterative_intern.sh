#!/bin/bash

# Set initial parameters

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

best_checkpoint=""
best_checkpoint_rm=""
best_accuracy=0
best_epoch_num=0
for epoch in $(seq 3 7); do
    max_batch_num=25
    max_batch_num_prev=$((max_batch_num - 1))
    epoch_prev=$((epoch - 1))
    val_result_dir="../tree_data_qwen_test/val_result/checkpoint_${epoch}"
    mkdir -p "$val_result_dir"

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


    echo "Using checkpoint directory: $checkpoint_dir"

    # Step 1: Run vllm_intern.py
    echo "Step 1: Running run_vllm_intern.py"
    # python save_checkpoints_intern.py --checkpoint $checkpoint_dir --save_dir "/data/projects/punim1996/Data/AVR-RL/RL_PRM/checkpoints/intern_checkpoint_test/policy_model"
    #Delete all contents in the destination directory
    rm -rf "../checkpoints_grpo/qwen_checkpoint_test/policy_model/"*

    # Copy all contents from the source directory to the destination directory
    cp -r "${checkpoint_dir}/"* "../checkpoints_grpo/qwen_checkpoint_test/policy_model"


    epoch_num_prev=$((epoch - 1))
    base_output_dir_reward="../output_rm_grpo/output_rm_epoch${epoch_num_prev}"

    if [ -d "$base_output_dir_reward" ]; then
        subfolder1=$(find "$base_output_dir_reward" -maxdepth 1 -type d -name "v*" | sort -V | tail -1)
        if [ -n "$subfolder1" ]; then
            # Find the checkpoint directory (subfolder2) within subfolder1
            subfolder2=$(find "$subfolder1" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
            if [ -n "$subfolder2" ]; then
                checkpoint_dir_rm="$subfolder2"
            fi
        fi
    fi

    SOURCE_DIR=$checkpoint_dir_rm
    DEST_DIR="../checkpoints_grpo/qwen_checkpoint_test/rm_model"

    # Delete all contents in the destination directory
    rm -rf "${DEST_DIR:?}/"*

    # Copy all contents from the source directory to the destination directory
    cp -r "${SOURCE_DIR}/"* "$DEST_DIR"


    ./deploy_vllm_online_intern_policy_test.sh
    ./deploy_vllm_online_intern_rm_test.sh
    sleep 120

    max_batch_num_test=15

    for batch_num in $(seq 0 $((max_batch_num_test-1))); do
        python MCTS_test_intern.py --epoch $epoch_num_prev --batch_num $batch_num --batch_size 20 --input '../../Trance_val.json' --output_base $val_result_dir
    done

    #python find_best_path.py --epoch $epoch_num_prev --dataset '../../Trance_val.json' --tree_path $val_result_dir


    accuracy_output=$(python find_best_path.py --epoch $epoch_num_prev --dataset '../../Trance_val.json' --tree_path $val_result_dir)
    echo "$accuracy_output"
    
    # 提取accuracy数值 (从 "Accuracy: XX.XX%" 中提取)
    current_accuracy=$(echo "$accuracy_output" | grep -oP 'Accuracy: \K[0-9.]+')
    
    if [ -n "$current_accuracy" ]; then
        
        # 比较并更新最佳checkpoint (使用bc进行浮点数比较)
        if (( $(echo "$current_accuracy > $best_accuracy" | bc -l) )); then
            best_accuracy=$current_accuracy
            best_checkpoint=$checkpoint_dir
            best_checkpoint_rm=$checkpoint_dir_rm
            best_epoch_num=$epoch_num_prev
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

# epoch=5
# max_batch_num=25
# max_batch_num_prev=$((max_batch_num - 1))
# epoch_prev=$((epoch - 1))
# base_output_dir_policy="../output_grpo/output_grpo_epoch${epoch_prev}_${max_batch_num_prev}"

# if [ -d "$base_output_dir_policy" ]; then
#     subfolder1=$(find "$base_output_dir_policy" -maxdepth 1 -type d -name "v*" | sort -V | tail -1)
#     if [ -n "$subfolder1" ]; then
#         # Find the checkpoint directory (subfolder2) within subfolder1
#         subfolder2=$(find "$subfolder1" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
#         if [ -n "$subfolder2" ]; then
#             checkpoint_dir="$subfolder2"
#         else
#             echo "Error: Could not find checkpoint subdirectory in $subfolder1"
#             exit 1
#         fi
#     else
#         echo "Error: Could not find version subdirectory in $base_output_dir_policy"
#         exit 1
#     fi
# else
#     echo "Error: Base output directory $base_output_dir_policy does not exist"
#     exit 1
# fi
#best_checkpoint="../output_grpo/output_grpo_epoch7_24/v0-20251003-115425/checkpoint-111"
#best_checkpoint_rm="../output_rm_grpo/output_rm_epoch7/v0-20251005-094800/checkpoint-24645"
#epoch_num_prev=7
echo "Using checkpoint directory: $best_checkpoint"

# Step 1: Run vllm_intern.py
echo "Step 1: Running run_vllm_intern.py"
# python save_checkpoints_intern.py --checkpoint $checkpoint_dir --save_dir "/data/projects/punim1996/Data/AVR-RL/RL_PRM/checkpoints/intern_checkpoint_test/policy_model"
#Delete all contents in the destination directory
rm -rf "../checkpoints_grpo/qwen_checkpoint_test/policy_model/"*

# Copy all contents from the source directory to the destination directory
cp -r "${best_checkpoint}/"* "../checkpoints_grpo/qwen_checkpoint_test/policy_model"


# epoch_num_prev=$((epoch - 1))
# base_output_dir_reward="../output_rm_grpo/output_rm_epoch${epoch_num_prev}"

# if [ -d "$base_output_dir_reward" ]; then
#     subfolder1=$(find "$base_output_dir_reward" -maxdepth 1 -type d -name "v*" | sort -V | tail -1)
#     if [ -n "$subfolder1" ]; then
#         # Find the checkpoint directory (subfolder2) within subfolder1
#         subfolder2=$(find "$subfolder1" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
#         if [ -n "$subfolder2" ]; then
#             checkpoint_dir="$subfolder2"
#         fi
#     fi
# fi

SOURCE_DIR=$best_checkpoint_rm
DEST_DIR="../checkpoints_grpo/qwen_checkpoint_test/rm_model"

# Delete all contents in the destination directory
rm -rf "${DEST_DIR:?}/"*

# Copy all contents from the source directory to the destination directory
cp -r "${SOURCE_DIR}/"* "$DEST_DIR"


./deploy_vllm_online_intern_policy_test.sh
./deploy_vllm_online_intern_rm_test.sh
sleep 120

max_batch_num_test=50

for batch_num in $(seq 0 $((max_batch_num_test-1))); do
    python MCTS_test_intern.py --epoch $best_epoch_num --batch_num $batch_num --batch_size 20 --input '../../Trance_test.json' --output_base '../tree_data_qwen_test'
done
echo "In Domain results: "
python find_best_path.py --epoch $best_epoch_num --dataset '../../Trance_test.json' --tree_path '../tree_data_qwen_test'

max_batch_num_test=50

for batch_num in $(seq 0 $((max_batch_num_test-1))); do
    python MCTS_test_intern.py --epoch $best_epoch_num --batch_num $batch_num --batch_size 20 --input '../../TranceL_test.json' --output_base '../tree_data_qwen_test_OOD1'
done

echo "Out of Domain Left results: "
python find_best_path.py --epoch $best_epoch_num --dataset '../../TranceL_test.json' --tree_path '../tree_data_qwen_test_OOD1'

max_batch_num_test=50

for batch_num in $(seq 0 $((max_batch_num_test-1))); do
    python MCTS_test_intern.py --epoch $best_epoch_num --batch_num $batch_num --batch_size 20 --input '../../TranceR_test.json' --output_base '../tree_data_qwen_test_OOD2'
done

echo "Out of Domain Right results: "
python find_best_path.py --epoch $best_epoch_num --dataset '../../TranceR_test.json' --tree_path '../tree_data_qwen_test_OOD2'

PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
for pid in $PIDS; do
    echo "Killing process $pid"
    kill -9 $pid
done

echo "All testing completed successfully!"