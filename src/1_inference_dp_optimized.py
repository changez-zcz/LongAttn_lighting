import os
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from tqdm import tqdm
from LongAttnOptimized import OptimizedLlamaForCausalLM, load_optimized_model
import jsonlines
import time
from accelerate import Accelerator
from accelerate.utils import gather_object


def inference(data_id, input_ids, attention_mask, model):
    print(f"start process the batch of {data_id[0]}")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    output = []
    for i in range(len(data_id)):
        data = {
            "data_id": data_id[i],
            "first_layer_proportion_score": outputs.proportions[0][i].cpu().numpy().tolist(),
            "variance": outputs.uniformities[0][i].cpu().numpy().tolist()
        }
        output.append(data)
    return output

def process_batch(batch, llm_tokenizer):
    # Extract text content from the batch
    texts = []
    for item in batch:
        if isinstance(item, dict):
            text = item.get('content', '')
        else:
            text = str(item)
        texts.append(text)
    
    tokens = llm_tokenizer.batch_encode_plus(
        texts, 
        add_special_tokens=True, 
        padding='max_length', 
        truncation=True, 
        max_length=131072,  # Updated to 128K for DeepSeek V3
        return_tensors='pt'
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens['attention_mask']

    return input_ids, attention_mask

def process_file(data_list, llm_tokenizer, model, batch_size, output_file):
    batch = []
    data_id = []
    for data in data_list:
        batch.append(data)  # Pass the entire data object
        data_id.append(data['data_id'])
        if len(batch) >= batch_size:
            input_ids, attention_mask = process_batch(batch, llm_tokenizer)
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            inference_res = inference(data_id, input_ids, attention_mask, model)
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write_all(inference_res)
            batch = []
            data_id = []

    if batch:
        input_ids, attention_mask = process_batch(batch, llm_tokenizer)
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        inference_res = inference(data_id, input_ids, attention_mask, model)
        with jsonlines.open(output_file, mode='a') as writer:
            writer.write_all(inference_res)
        batch = []
        data_id = []

def process_single_file(input_file, output_file, llm_tokenizer, model, batch_size):
    """Process a single input file and write results to output file"""
    print(f"Processing {input_file} -> {output_file}")
    
    # Read input data
    with jsonlines.open(input_file) as reader:
        data_list = [obj for obj in reader]
    
    if not data_list:
        print(f"No data found in {input_file}")
        return
    
    print(f"Found {len(data_list)} records in {input_file}")
    
    # Process the file
    process_file(data_list, llm_tokenizer, model, batch_size, output_file)
    print(f"Completed processing {input_file}")

def process_directory(input_dir, output_dir, llm_tokenizer, model, batch_size):
    """Process all .jsonl files in input directory and save results to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .jsonl files in input directory (including subdirectories)
    input_files = []
    for root, dirs, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jsonl'):
                input_files.append(os.path.join(root, filename))
    
    if not input_files:
        print(f"No .jsonl files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} .jsonl files to process")
    
    # Process each file
    for input_file in tqdm(input_files, desc="Processing files"):
        # Create corresponding output file path
        rel_path = os.path.relpath(input_file, input_dir)
        output_file = os.path.join(output_dir, rel_path)
        
        # Create output directory if needed
        output_file_dir = os.path.dirname(output_file)
        os.makedirs(output_file_dir, exist_ok=True)
        
        # Process the file
        process_single_file(input_file, output_file, llm_tokenizer, model, batch_size)

def setup_multi_gpu():
    """Setup for multi-GPU inference"""
    # Check available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")
        
        # For A800 (80GB each), we can use all 8 GPUs
        if num_gpus >= 8:
            print("Using 8-GPU setup for A800 cluster")
            device_map = "auto"  # Let accelerate handle distribution
        else:
            print(f"Using {num_gpus} GPU(s)")
            device_map = "auto"
    else:
        print("No GPU available, using CPU")
        device_map = "cpu"
    
    return device_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process inference on data files with optimized memory usage.')
    
    parser.add_argument('input_path', type=str, help='Path to input file or directory')
    parser.add_argument('output_path', type=str, help='Path to output file or directory')
    parser.add_argument('batch_size', type=int, help='The batch size for processing')
    parser.add_argument('--model_path', type=str, default="deepseek-ai/deepseek-coder-33b-instruct", 
                       help='Path to the model')
    parser.add_argument('--use_optimized', action='store_true', 
                       help='Use optimized model loading (recommended for 8-GPU setup)')
    
    args = parser.parse_args()
    
    accelerator = Accelerator()
    batch_size = args.batch_size

    # Setup device mapping for multi-GPU
    device_map = setup_multi_gpu()
    
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with optimization
    if args.use_optimized:
        print("Loading optimized model (first layer only)...")
        model = load_optimized_model(
            args.model_path,
            device_map=device_map,
            torch_dtype=torch.float16
        )
    else:
        print("Loading full model...")
        config_kwargs = {
            "cache_dir": None,
            "revision": 'main',
            "use_auth_token": None,
            "rope_theta": 1000000.0,  # Updated RoPE theta for DeepSeek V3
        }

        config = AutoConfig.from_pretrained(args.model_path, **config_kwargs)
        config.num_hidden_layers = 1
        
        model = OptimizedLlamaForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map
        )
    
    model.eval()

    start_time = time.time()
    
    # Check if input is file or directory
    if os.path.isfile(args.input_path):
        # Process single file
        process_single_file(args.input_path, args.output_path, tokenizer, model, batch_size)
    elif os.path.isdir(args.input_path):
        # Process directory
        process_directory(args.input_path, args.output_path, tokenizer, model, batch_size)
    else:
        print(f"Error: {args.input_path} is neither a file nor a directory")
        exit(1)
    
    end_time = time.time()  

    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time} 秒")
    
    # Print memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB") 