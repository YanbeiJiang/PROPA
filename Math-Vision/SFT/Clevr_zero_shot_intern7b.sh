#!/bin/bash


result_path="./result/InternVL2_5-2B/infer_result/zero_shot7b_OOD.jsonl"
if [ -f "$result_path" ]; then
    echo "文件存在，正在删除..."
    rm "$result_path"
    echo "文件已删除: $result_path"
else
    echo "文件不存在: $result_path"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model OpenGVLab/InternVL2_5-8B \
    --val_dataset '../DynaMath_test.json' \
    --temperature 0 \
    --max_new_tokens 2048 \
    --use_hf true \
    --max_batch_size 8 \
    --result_path $result_path

python get_accuracy.py --file_path $result_path