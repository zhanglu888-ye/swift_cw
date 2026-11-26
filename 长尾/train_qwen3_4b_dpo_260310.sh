#!/bin/bash

# 设置显卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 解决 PyTorch 显存碎片化问题 (根据报错提示添加)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SFT_MODEL_PATH="/host/ssddata/disk2/zhanglu/swift/model/Qwen/Qwen3-VL-4B-Instruct"
OUTPUT_DIR="/host/ssddata/disk2/zhanglu/swift/4cls/output/train_260306cls3v1_epochs20/train_260312cls3v1_dpo_epochs2"

NPROC_PER_NODE=8 swift rlhf \
    --rlhf_type dpo \
    --model "$SFT_MODEL_PATH" \
    --train_type lora \
    --dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/dpo_train_strict.jsonl \
    --val_dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/dpo_val_strict.jsonl \
    --load_from_cache_file true \
    --learning_rate 5e-6 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.03 \
    --freeze_vit true \
    --gradient_checkpointing true \
    --rpo_alpha 0.1 \
    --beta 0.1 \
    --max_length 8192 \
    --deepspeed zero3
