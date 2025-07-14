import os
import argparse
from DataProcess import DataProcessor

def main():

    parser = argparse.ArgumentParser(description="Process data using DataProcessor.")
    
    # Arguments for DataProcessor
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing raw data files (supports nested directories).')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder for processed data (will preserve directory structure).')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of workers for parallel processing.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing.')

    args = parser.parse_args()

    # Process data using DataProcessor (no segmentation, preserves directory structure)
    data_processor = DataProcessor(max_workers=args.max_workers, batch_size=args.batch_size)
    data_processor.data_part(args.input_folder, args.output_folder)
    
    print(f"Preprocessing completed. Results saved to {args.output_folder}")

if __name__ == "__main__":
    main()