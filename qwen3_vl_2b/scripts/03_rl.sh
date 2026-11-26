#!/usr/bin/env bash
source "$(dirname "$0")/config.sh"
# MODEL_DIR can point to either original Instruct or exported SFT model.
# Native transformers generation avoids the vLLM/CUDA compatibility matrix.
swift rlhf "${COMMON_ARGS[@]}" \
  --rlhf_type grpo --dataset "$DATA_DIR/grpo_train.jsonl" \
  --val_dataset "$DATA_DIR/grpo_val.jsonl" --split_dataset_ratio 0 \
  --external_plugins "$PROJECT_DIR/rewards.py" \
  --reward_funcs danger_format danger_accuracy --reward_weights 0.2 1.0 \
  --num_generations 4 --num_generations_eval 1 --generation_batch_size "$((NPROC_PER_NODE * NNODES * 4))" \
  --max_completion_length 256 --temperature 0.9 --beta 0.04 \
  --learning_rate "${RL_LR:-5e-6}" --num_train_epochs "${RL_EPOCHS:-1}" \
  --use_vllm false --remove_unused_columns false \
  --eval_strategy epoch --save_strategy epoch \
  --output_dir "${RL_OUTPUT_DIR:-$OUTPUT_ROOT/rl}" "$@"
