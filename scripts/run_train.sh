#!/usr/bin/env bash
# scripts/run_train.sh
# Run training for Lit2Vec TL;DR DistilBART

set -euo pipefail  # safer: exit on error, undefined vars, and pipe fails

# === Variables ===
CONFIG_FILE="configs/train_distilbart.yaml"
PYTHON_SCRIPT="src/train.py"
LOG_DIR="logs"
RUN_NAME="lit2vec_distilbart_run"

mkdir -p "$LOG_DIR"

echo "🚀 Starting training with config: $CONFIG_FILE"
echo "📂 Logs will be saved in $LOG_DIR/train.log"

# === Run training ===
python3 "$PYTHON_SCRIPT" \
  --config "$CONFIG_FILE" \
  --run_name "$RUN_NAME" \
  2>&1 | tee "$LOG_DIR/train.log"
