#!/bin/bash

# Set input parameters
INPUT_FOLDER="/path/to/input"          # Input folder path (supports nested directories)
OUTPUT_FOLDER="/path/to/output"        # Output folder path (will preserve directory structure)
MAX_WORKERS=100                         # Maximum number of worker threads
BATCH_SIZE=500                         # Batch size

# Run the Python script
python src/0_pre-process.py \
    --input_folder "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --max_workers "$MAX_WORKERS" \
    --batch_size "$BATCH_SIZE"

if [ $? -eq 0 ]; then
    echo "Data processing completed successfully!"
else
    echo "Data processing failed, please check the error messages."
fi
