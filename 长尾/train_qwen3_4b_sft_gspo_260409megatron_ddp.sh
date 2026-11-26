#!/bin/bash

# ==================== 1. 分布式环境变量配置 ====================
export NNODES=2              # 机器总数
export NODE_RANK=0           # 当前机器序号：机器0设为0，机器1设为1
export MASTER_ADDR="10.23.32.13" # 机器0的内部IP
export MASTER_PORT=12359
export NPROC_PER_NODE=8      # 每台机器的GPU数量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 内存优化：参考官方示例
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 多机通信优化 (根据你的网络环境调整)
export NCCL_IGNORE_IPV6=1
# 如果你的两台机器之间有显存/RDMA，建议关闭下行 (将 DISABLE 设为 0)
# 如果只有普通千兆/万兆网卡，保持现状
export NCCL_IB_DISABLE=1 
export NCCL_SOCKET_IFNAME=bond0 
export GLOO_SOCKET_IFNAME=bond0

# ==================== 2. 路径配置 ====================
# 注意：Megatron 训练通常建议使用转换后的 MCore 权重
# 如果 --load 指向 HF 权重，swift 会在启动时尝试转换，多机环境下建议先离线转换好或指向 MCore 路径
MCORE_PATH="/host/ssddata/disk2/zhanglu/swift/model/Qwen/Qwen3-VL-4B-mcore"
REWARD_SCRIPT="/host/ssddata/disk2/zhanglu/swift/4cls/custom_rewards.py"
SAVE_DIR="megatron_output/Qwen3-VL-4B-SAPO"

# ==================== 3. 启动 Megatron SAPO ====================
# 使用 NPROC_PER_NODE 和 NNODES 启动分布式环境
megatron rlhf \
    --rlhf_type grpo \
    --model_type qwen3_vl \
    --mcore_model "$MCORE_PATH" \
    --dataset "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/grpo_train.jsonl" \
    --val_dataset "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/grpo_val.jsonl" \
    --external_plugins "$REWARD_SCRIPT" \
    --reward_funcs my_format_reward my_accuracy_reward \
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
    --num_generations 4 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    \
    --max_new_tokens 800 \
    --max_length 3072 \
    --max_pixels 602112 \
    \
    --lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --train_iters 2000 \
    --lr_decay_style cosine \
    --save_steps 100 \
    --eval_iters 100 \
    --eval_steps 100 \
    --save_total_limit 3 \
    \
    --truncation_strategy delete \
    --beta 0 \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --steps_per_generation 4\
    --sequence_parallel true \
    --attention_backend flash \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --importance_sampling_level sequence \
    \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.52 \
    --vllm_max_model_len 4096 \
    --vllm_enable_prefix_caching true \
    --vllm_max_num_seqs 2 \
    \
    --offload_optimizer true \
    --offload_model true \
    --no_save_optim true