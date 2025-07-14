#!/bin/bash

# Set input parameters for the first script
input_path="/path/to/input_directory"      # Input directory path (containing .jsonl files)
output_inference_dir="/path/to/inference_output_directory"  # Output directory path for inference
batch_size=6  # Batch size

# Check batch_size
if ! [[ "$batch_size" =~ ^[0-9]+$ ]]; then
    echo "Error: Batch size must be a number."
    exit 1
fi

# Define error log file
ERROR_LOG="error.log"

# Clear the existing error log file
> "$ERROR_LOG"

# Run the first inference script
echo "Running inference..."
if ! accelerate launch src/1_inference_dp.py "$input_path" "$output_inference_dir" "$batch_size" 2>>"$ERROR_LOG"; then
    echo "Inference script failed. Check $ERROR_LOG for details."
    exit 1
fi

# Define parameters for the second script (DateSorted)
data_path="$input_path"  # Original data directory
output_filtered_dir="/path/to/final_filtered_directory"  # Final filtered output directory

# Run the second script (DateSorted)
echo "Running DateSorted processing..."
if ! python src/2_filtering_by_Lds.py \
    --inference_path "$output_inference_dir" \
    --data_path "$data_path" \
    --output_path "$output_filtered_dir" 2>>"$ERROR_LOG"; then
    echo "DateSorted processing script failed. Check $ERROR_LOG for details."
    exit 1
fi

echo "Processing complete. Results saved to $output_filtered_dir."
