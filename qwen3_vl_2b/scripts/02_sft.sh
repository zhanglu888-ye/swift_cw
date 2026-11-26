#!/usr/bin/env bash
source "$(dirname "$0")/config.sh"
swift sft "${COMMON_ARGS[@]}" \
  --dataset "$DATA_DIR/sft_train.jsonl" --val_dataset "$DATA_DIR/sft_val.jsonl" \
  --split_dataset_ratio 0 --learning_rate "${SFT_LR:-1e-4}" \
  --num_train_epochs "${SFT_EPOCHS:-3}" --warmup_ratio 0.05 \
  --eval_strategy epoch --save_strategy epoch \
  --output_dir "$OUTPUT_ROOT/sft" "$@"
