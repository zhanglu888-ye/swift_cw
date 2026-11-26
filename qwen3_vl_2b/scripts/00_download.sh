#!/usr/bin/env bash
source "$(dirname "$0")/config.sh"
python "$PROJECT_DIR/download_model.py" --output "$MODEL_DIR" "$@"
