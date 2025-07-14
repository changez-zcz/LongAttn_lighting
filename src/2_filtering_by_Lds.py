import argparse
import os
import jsonlines
from tqdm import tqdm
from DataProcess import DateSorted

def process_single_file(inference_file, data_file, output_file):
    """Process a single inference file and corresponding data file"""
    print(f"Processing {inference_file} -> {output_file}")
    
    # Check if files exist
    if not os.path.exists(inference_file):
        print(f"Warning: Inference file {inference_file} not found, skipping...")
        return
    
    if not os.path.exists(data_file):
        print(f"Warning: Data file {data_file} not found, skipping...")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process the file
    DateSorted(
        inference_path=inference_file,
        file_path=data_file,
        output_path=output_file
    ).write_to_file()
    
    print(f"Completed filtering {inference_file}")

def process_directory(inference_dir, data_dir, output_dir):
    """Process all files in inference directory and corresponding data files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all inference files
    inference_files = []
    for root, dirs, filenames in os.walk(inference_dir):
        for filename in filenames:
            if filename.endswith('.jsonl'):
                inference_files.append(os.path.join(root, filename))
    
    if not inference_files:
        print(f"No .jsonl files found in {inference_dir}")
        return
    
    print(f"Found {len(inference_files)} inference files to process")
    
    # Process each file
    for inference_file in tqdm(inference_files, desc="Processing files"):
        # Create corresponding data file path
        rel_path = os.path.relpath(inference_file, inference_dir)
        data_file = os.path.join(data_dir, rel_path)
        output_file = os.path.join(output_dir, rel_path)
        
        # Process the file
        process_single_file(inference_file, data_file, output_file)

def main():
    parser = argparse.ArgumentParser(description="Sort data with LDS using DateSorted class")
    parser.add_argument('--inference_path', type=str, required=True, help='Path to the inference data (file or directory)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the original data (file or directory)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data (file or directory)')

    args = parser.parse_args()

    # Check if input is file or directory
    if os.path.isfile(args.inference_path):
        # Process single file
        if not os.path.isfile(args.data_path):
            print(f"Error: When inference_path is a file, data_path must also be a file")
            exit(1)
        process_single_file(args.inference_path, args.data_path, args.output_path)
    elif os.path.isdir(args.inference_path):
        # Process directory
        if not os.path.isdir(args.data_path):
            print(f"Error: When inference_path is a directory, data_path must also be a directory")
            exit(1)
        process_directory(args.inference_path, args.data_path, args.output_path)
    else:
        print(f"Error: {args.inference_path} is neither a file nor a directory")
        exit(1)

if __name__ == "__main__":
    main()
