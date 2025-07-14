#!/bin/bash
# 数据预处理脚本
# 用法: bash scripts/preprocess.sh /path/to/raw_data /path/to/processed_data

INPUT_DIR=${1:-"data/raw"}
OUTPUT_DIR=${2:-"data/processed"}
BATCH_SIZE=100

mkdir -p "$OUTPUT_DIR"

python3 src/data_process.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo "[Preprocess] Success! Output: $OUTPUT_DIR"
else
    echo "[Preprocess] Failed!"
    exit 1
fi 