#!/bin/bash

model_name="InternVL2_5-2B"
output_dir="$1"
val_dataset='../Math_Vision_val_ORM.json'
val_result_dir="./result_rl/$model_name/val_result"

mkdir -p "$val_result_dir"

best_checkpoint=""
best_accuracy=0

for checkpoint_path in "$output_dir"/checkpoint-*; do
    if [ -d "$checkpoint_path" ]; then
        checkpoint_name=$(basename "$checkpoint_path")
        #echo "正在评估: $checkpoint_name"
        
        val_result_path="$val_result_dir/${checkpoint_name}_val.jsonl"
        
        # 删除旧的结果文件
        if [ -f "$val_result_path" ]; then
            rm "$val_result_path"
        fi
        # 在验证集上运行inference
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        swift infer \
            --adapters "$checkpoint_path" \
            --infer_backend pt \
            --val_dataset "$val_dataset" \
            --temperature 0 \
            --max_new_tokens 4096 \
            --use_hf true \
            --max_batch_size 4 \
            --result_path "$val_result_path"
        
        # 获取accuracy并提取数值
        accuracy_output=$(python get_accuracy.py --file_path "$val_result_path")
        echo "$accuracy_output"
        
        # 提取accuracy数值 (从 "Accuracy: XX.XX%" 中提取)
        current_accuracy=$(echo "$accuracy_output" | grep -oP 'Accuracy: \K[0-9.]+')
        
        if [ -n "$current_accuracy" ]; then
            echo "$checkpoint_name: Accuracy = $current_accuracy%"
            
            # 比较并更新最佳checkpoint (使用bc进行浮点数比较)
            if (( $(echo "$current_accuracy > $best_accuracy" | bc -l) )); then
                best_accuracy=$current_accuracy
                best_checkpoint=$checkpoint_path
                echo ">>> found better checkpoint: $checkpoint_name (Accuracy: $best_accuracy%)"
            fi
        else
            echo "cannot export accuracy"
        fi
        
        echo "----------------------------------------"
    fi
done

checkpoint_path="$best_checkpoint"

result_path="./result_rl/$model_name/infer_result/sft.jsonl"
#checkpoint_path="./output/v4-20250926-105406/checkpoint-1250"
if [ -f "$result_path" ]; then
    echo "文件存在，正在删除..."
    rm "$result_path"
    echo "文件已删除: $result_path"
else
    echo "文件不存在: $result_path"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --adapters "$checkpoint_path" \
    --infer_backend pt \
    --val_dataset '../Math_Vision_test_ORM.json' \
    --temperature 0 \
    --max_new_tokens 4096 \
    --use_hf true \
    --max_batch_size 4 \
    --result_path $result_path

echo "In-Domain results:"

python get_accuracy.py --file_path $result_path

result_path="./result_rl/$model_name/infer_result/sft_OOD.jsonl"
if [ -f "$result_path" ]; then
    echo "文件存在，正在删除..."
    rm "$result_path"
    echo "文件已删除: $result_path"
else
    echo "文件不存在: $result_path"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --adapters "$checkpoint_path" \
    --infer_backend pt \
    --val_dataset '../DynaMath_test_ORM.json' \
    --temperature 0 \
    --max_new_tokens 4096 \
    --use_hf true \
    --max_batch_size 4 \
    --result_path $result_path

echo "Out-of-Domain results:"

python get_accuracy.py --file_path $result_path