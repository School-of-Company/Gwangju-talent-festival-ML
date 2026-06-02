#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_path> <output_dir>"
    exit 1
fi

DATASET_PATH="$1"
OUTPUT_DIR="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

PYTHON_BIN="python"
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" trainer/train_iforest.py \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR"
