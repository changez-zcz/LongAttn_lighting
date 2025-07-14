import os
import argparse
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import jsonlines
from longattn_optimized import LongAttnOptimized

def process_file(input_file, output_file, tokenizer, model, batch_size):
    with jsonlines.open(input_file) as reader:
        data_list = [obj for obj in reader]
    batch = []
    data_ids = []
    for data in data_list:
        batch.append(data['content'])
        data_ids.append(data['data_id'])
        if len(batch) >= batch_size:
            run_inference(batch, data_ids, output_file, tokenizer, model)
            batch, data_ids = [], []
    if batch:
        run_inference(batch, data_ids, output_file, tokenizer, model)

def run_inference(batch, data_ids, output_file, tokenizer, model):
    tokens = tokenizer(batch, padding='max_length', truncation=True, max_length=131072, return_tensors='pt')
    input_ids = tokens['input_ids'].to('cuda')
    with torch.no_grad():
        outputs = model(input_ids)
    # 这里只做示例，实际应提取注意力分数等
    results = [{"data_id": did, "dummy_output": "..."} for did in data_ids]
    with jsonlines.open(output_file, mode='a') as writer:
        writer.write_all(results)

def process_dir(input_dir, output_dir, tokenizer, model, batch_size):
    os.makedirs(output_dir, exist_ok=True)
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jsonl'):
                files.append(os.path.join(root, filename))
    for file_path in tqdm(files, desc="Inference files"):
        rel_path = os.path.relpath(file_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        process_file(file_path, out_path, tokenizer, model, batch_size)

def main():
    parser = argparse.ArgumentParser(description="Optimized inference for LongAttn (deepseek-v3)")
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--index_json', required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = LongAttnOptimized.from_pretrained(args.model_dir, index_json=args.index_json, device='cuda')
    process_dir(args.input_dir, args.output_dir, tokenizer, model, args.batch_size)
    print(f"Inference complete. Output saved to {args.output_dir}")

if __name__ == '__main__':
    main() 