#!/usr/bin/env bash
source "$(dirname "$0")/config.sh"
: "${CHECKPOINT:?设置待合并 LoRA 的 CHECKPOINT}"
: "${EXPORT_DIR:?设置完整模型导出目录 EXPORT_DIR}"
# For SFT+RL adapters, MODEL_DIR must be the merged SFT model used in RL.
swift export --model "$MODEL_DIR" --model_type qwen3_vl \
  --adapters "$CHECKPOINT" --merge_lora true --output_dir "$EXPORT_DIR" "$@"
