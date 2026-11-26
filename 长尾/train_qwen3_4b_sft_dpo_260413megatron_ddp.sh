#!/bin/bash

# ==================== 1. 分布式环境变量配置 ====================
export NNODES=2              # 机器总数
export NODE_RANK=0           # 当前机器序号
export MASTER_ADDR="10.23.32.13" 
export MASTER_PORT=12359
export NPROC_PER_NODE=8      
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export NCCL_IB_DISABLE=1 
export NCCL_SOCKET_IFNAME=bond0 
export GLOO_SOCKET_IFNAME=bond0

# ==================== 2. 路径配置 ====================
# 注意：Megatron 模式需指向 MCore 转换后的权重路径
MCORE_PATH="/host/ssddata/disk2/zhanglu/swift/model/Qwen/Qwen3-VL-4B-mcore"
SAVE_DIR="/host/ssddata/disk2/zhanglu/swift/4cls/output/megatron_output/Qwen3-VL-4B-DPO"

# ==================== 3. 启动 Megatron DPO ====================
megatron rlhf \
    --rlhf_type dpo \
    --model_type qwen3_vl \
    --mcore_model "$MCORE_PATH" \
    --dataset "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/dpo_train_mixed_strict_reject2_new.jsonl" \
    --val_dataset "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/dpo_val_strict_reject2_new.jsonl" \
    --output_dir "$SAVE_DIR" \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --finetune true \
    \
    --tensor_model_parallel_size 8 \
    --pipeline_model_parallel_size 1 \
    \
    --micro_batch_size 1 \
    --global_batch_size 64 \
    \
    --max_length 12288 \
    --max_pixels 602112 \
    --freeze_vit true \
    \
    --lr 5e-6 \
    --lr_warmup_fraction 0.03 \
    --train_iters 500 \
    --save_steps 100 \
    --save_total_limit 3 \
    \
    --beta 0.1 \
    --rpo_alpha 0.1 \
    \
    --sequence_parallel true \
    --attention_backend flash \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    \
    --offload_optimizer true \
    --offload_model true \
    --no_save_optim true