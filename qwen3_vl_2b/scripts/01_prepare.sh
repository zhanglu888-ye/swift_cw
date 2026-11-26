#!/usr/bin/env bash
source "$(dirname "$0")/config.sh"
: "${IMAGE_ROOT:?请设置 IMAGE_ROOT，目录下需包含 positive/negative 原始图片}"
python "$PROJECT_DIR/prepare_data.py" \
  --source "${ANNOTATION_DIR:-$REPO_DIR/人体小图危险物品qwen3-27B标注}" \
  --image-root "$IMAGE_ROOT" --output "$DATA_DIR" "$@"
