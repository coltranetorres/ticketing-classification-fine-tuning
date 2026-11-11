#!/usr/bin/env bash
# Usage: scripts/train_lora.sh [LEARNING_RATE] [ADAPTER_DIR] [DATA_DIR] [ITERS]
set -euo pipefail
LR="${1:-1e-5}"
OUT="${2:-adapters}"
DATA="${3:-data/mlx}"
ITERS="${4:-750}"
uv run mlx_lm.lora --model mlx-community/gemma-3-1b-it-bf16 --train --data "$DATA" \
  --adapter-path "$OUT" --batch-size 4 --iters "$ITERS" --save-every 250 \
  --learning-rate "$LR" --num-layers 16
