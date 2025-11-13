#!/bin/bash


result_path="./result/Qwen2.5-VL-3B-Instruct/infer_result/zero_shot.jsonl"
if [ -f "$result_path" ]; then
    echo "文件存在，正在删除..."
    rm "$result_path"
    echo "文件已删除: $result_path"
else
    echo "文件不存在: $result_path"
fi

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --val_dataset '../DynaMath_test.json' \
    --temperature 0 \
    --max_new_tokens 2048 \
    --use_hf true \
    --max_batch_size 8 \
    --result_path $result_path

python get_accuracy.py --file_path $result_path