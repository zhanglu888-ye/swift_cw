#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(dirname "$PROJECT_DIR")"
export MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/Qwen3-VL-2B-Instruct}"
export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export MAX_PIXELS="${MAX_PIXELS:-262144}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON_ARGS=(--model "$MODEL_DIR" --model_type qwen3_vl --train_type lora
    --torch_dtype bfloat16 --attn_impl sdpa --target_modules all-linear
    --lora_rank 16 --lora_alpha 32 --freeze_vit true --freeze_aligner true
    --gradient_checkpointing true --max_length 2048
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1
    --gradient_accumulation_steps "${GRAD_ACCUM:-8}" --seed 42
    --save_total_limit 2 --logging_steps 5 --report_to none
    --dataloader_num_workers 4)
