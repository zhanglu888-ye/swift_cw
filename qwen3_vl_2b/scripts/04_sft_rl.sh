#!/usr/bin/env bash
source "$(dirname "$0")/config.sh"
: "${SFT_CHECKPOINT:?请指定本次 SFT 的 checkpoint 目录，不自动猜测最新目录}"
MERGED_DIR="${MERGED_DIR:-$OUTPUT_ROOT/sft_merged}"
if [[ -e "$MERGED_DIR" ]]; then
  echo "合并目标已存在，请设置新的 MERGED_DIR: $MERGED_DIR" >&2
  exit 1
fi
swift export --model "$MODEL_DIR" --model_type qwen3_vl \
  --adapters "$SFT_CHECKPOINT" --merge_lora true --output_dir "$MERGED_DIR"
MODEL_DIR="$MERGED_DIR" RL_OUTPUT_DIR="$OUTPUT_ROOT/sft_rl" \
  bash "$PROJECT_DIR/scripts/03_rl.sh" "$@"
