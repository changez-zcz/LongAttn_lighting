#!/bin/bash

# Optimized inference and filtering script for 8-GPU A800 setup
# This script uses memory-optimized model loading to reduce GPU memory usage

# Set input parameters
input_path="/path/to/processed/data"              # Preprocessed data directory
output_inference_dir="/path/to/inference_results" # Inference results directory
output_filtered_dir="/path/to/filtered_results"   # Final filtered results directory
batch_size=8                                      # Batch size (adjusted for 8-GPU setup)
model_path="deepseek-ai/deepseek-coder-33b-instruct"  # Model path

# Check batch_size
if ! [[ "$batch_size" =~ ^[0-9]+$ ]]; then
    echo "Error: Batch size must be a number."
    exit 1
fi

# Define error log file
ERROR_LOG="error_optimized.log"

# Clear the existing error log file
> "$ERROR_LOG"

# Check GPU availability
echo "Checking GPU setup..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | while IFS=, read -r name total free; do
    echo "GPU: $name, Total: ${total}MB, Free: ${free}MB"
done

# Run the optimized inference script
echo "Running optimized inference..."
if ! accelerate launch --multi_gpu --num_processes=8 src/1_inference_dp_optimized.py \
    "$input_path" \
    "$output_inference_dir" \
    "$batch_size" \
    --model_path "$model_path" \
    --use_optimized 2>>"$ERROR_LOG"; then
    echo "Optimized inference script failed. Check $ERROR_LOG for details."
    exit 1
fi

# Run the filtering script
echo "Running filtering..."
if ! python src/2_filtering_by_Lds.py \
    --inference_path "$output_inference_dir" \
    --data_path "$input_path" \
    --output_path "$output_filtered_dir" 2>>"$ERROR_LOG"; then
    echo "Filtering script failed. Check $ERROR_LOG for details."
    exit 1
fi

echo "Processing complete. Results saved to $output_filtered_dir."

# Print final memory usage
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader,nounits | while IFS=, read -r name used free; do
    echo "GPU: $name, Used: ${used}MB, Free: ${free}MB"
done 