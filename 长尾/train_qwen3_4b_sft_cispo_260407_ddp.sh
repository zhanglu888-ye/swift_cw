#!/bin/bash
export NNODES=2
export NODE_RANK=0   
export MASTER_ADDR="10.23.32.13"
export MASTER_PORT=12358
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export NCCL_IGNORE_IPV6=1
export NCCL_IB_DISABLE=1   
export NCCL_SOCKET_IFNAME=bond0 
export GLOO_SOCKET_IFNAME=bond0

SFT_MODEL_PATH="/host/ssddata/disk2/zhanglu/swift/4cls/output/train_260306cls3v1_epochs20/v0-20260306-165019/model-1800"
REWARD_SCRIPT="/host/ssddata/disk2/zhanglu/swift/4cls/custom_rewards.py"
OUTPUT_DIR="/host/ssddata/disk2/zhanglu/swift/4cls/output/train_260306cls3v1_epochs20/train_260327cls3v1_cispo_epochs2"

swift rlhf \
    --rlhf_type grpo \
    --model "$SFT_MODEL_PATH" \
    --train_type lora \
    --dataset "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/grpo_train.jsonl" \
    --val_dataset "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/grpo_val.jsonl" \
    --external_plugins "$REWARD_SCRIPT" \
    --reward_funcs my_format_reward my_accuracy_reward \
    --num_generations 2 \
    --max_new_tokens 800 \
    --num_train_epochs 2 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --eval_accumulation_steps 1 \
    --save_steps 100 \
    --output_dir "$OUTPUT_DIR" \
    --freeze_vit true \
    --gradient_checkpointing true \
    --use_vllm true \
    --vllm_mode colocate \
    --deepspeed zero3 \
    --vllm_gpu_memory_utilization 0.54 \
    --vllm_enforce_eager true \
    --vllm_max_model_len 2816 \
    --max_length 2816 \
    --max_pixels 602112 \
    --importance_sampling_level token \
    --truncation_strategy delete \
    --loss_type sapo \
    --tau_pos 1.0 \
    --tau_neg 1.05



