#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SFT_MODEL_PATH="/host/ssddata/disk2/zhanglu/swift/4cls/output/train_260306cls3v1_epochs20/v0-20260306-165019/model-1800"

# 奖励函数脚本路径
REWARD_SCRIPT="/host/ssddata/disk2/zhanglu/swift/4cls/custom_rewards.py"

OUTPUT_DIR="/host/ssddata/disk2/zhanglu/swift/4cls/output/train_260306cls3v1_epochs20/train_260327cls3v1_gspo_epochs2"

NPROC_PER_NODE=8 swift rlhf \
    --rlhf_type grpo \
    --model "$SFT_MODEL_PATH" \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/grpo_train.jsonl \
    --val_dataset "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/grpo_val.jsonl#512" \
    --external_plugins "$REWARD_SCRIPT" \
    --reward_funcs my_format_reward my_accuracy_reward \
    --num_generations 2 \
    --num_train_epochs 2 \
    --max_new_tokens 800 \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 100 \
    --output_dir "$OUTPUT_DIR" \
    --freeze_vit true \
    --gradient_checkpointing true \
    --use_vllm true \
    --deepspeed zero3 \
    --vllm_gpu_memory_utilization 0.55 \
    --vllm_enforce_eager true \
    --vllm_max_model_len 3072 \
    --max_length 3072 \
    --max_pixels 602112 \
    --importance_sampling_level sequence \
    --eval_accumulation_steps 1