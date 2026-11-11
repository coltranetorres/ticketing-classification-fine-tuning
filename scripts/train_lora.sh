#!/usr/bin/env bash
# Usage: scripts/train_lora.sh [LEARNING_RATE] [ADAPTER_DIR]
set -euo pipefail
LR="${1:-1e-5}"
OUT="${2:-adapters}"
uv run mlx_lm.lora --model mlx-community/gemma-3-1b-it-bf16 --train --data data/mlx \
  --adapter-path "$OUT" --batch-size 4 --iters 750 --save-every 250 \
  --learning-rate "$LR" --num-layers 16
