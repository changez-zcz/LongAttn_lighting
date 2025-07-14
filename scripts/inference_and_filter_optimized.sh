#!/bin/bash
# 优化推理+过滤主流程脚本（deepseek-v3）
# 用法: bash scripts/inference_and_filter_optimized.sh /input /output /model_dir /index_json

INPUT_DIR=${1:-"data/processed"}
INFER_DIR=${2:-"data/infer"}
MODEL_DIR=${3:-"DeepSeek-V3-0324"}
INDEX_JSON=${4:-"DeepSeek-V3-0324/model.safetensors.index.json"}
FILTER_DIR=${5:-"data/filtered"}
BATCH_SIZE=2

mkdir -p "$INFER_DIR"
mkdir -p "$FILTER_DIR"

python3 src/inference_optimized.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$INFER_DIR" \
    --model_dir "$MODEL_DIR" \
    --index_json "$INDEX_JSON" \
    --batch_size $BATCH_SIZE

if [ $? -ne 0 ]; then
    echo "[Inference] Failed!"
    exit 1
fi

python3 src/filtering.py \
    --inference_dir "$INFER_DIR" \
    --data_dir "$INPUT_DIR" \
    --output_dir "$FILTER_DIR"

if [ $? -eq 0 ]; then
    echo "[Inference+Filter] Success! Output: $FILTER_DIR"
else
    echo "[Filter] Failed!"
    exit 1
fi 