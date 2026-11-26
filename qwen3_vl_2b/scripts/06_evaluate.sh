#!/usr/bin/env bash
source "$(dirname "$0")/config.sh"
python "$PROJECT_DIR/evaluate.py" --model "$MODEL_DIR" \
  --dataset "$DATA_DIR/grpo_val.jsonl" \
  --output "${EVAL_OUTPUT:-$OUTPUT_ROOT/eval/predictions.jsonl}" --max-pixels "$MAX_PIXELS" "$@"
