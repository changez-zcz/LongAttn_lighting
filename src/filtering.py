import os
import argparse
import jsonlines
from tqdm import tqdm

def filter_file(infer_file, data_file, output_file):
    # 示例：直接复制，实际应根据LDS等指标过滤
    with jsonlines.open(data_file) as reader, jsonlines.open(output_file, 'w') as writer:
        for obj in reader:
            writer.write(obj)

def filter_dir(infer_dir, data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = []
    for root, _, filenames in os.walk(infer_dir):
        for filename in filenames:
            if filename.endswith('.jsonl'):
                files.append(os.path.join(root, filename))
    for infer_path in tqdm(files, desc="Filtering files"):
        rel_path = os.path.relpath(infer_path, infer_dir)
        data_path = os.path.join(data_dir, rel_path)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        filter_file(infer_path, data_path, out_path)

def main():
    parser = argparse.ArgumentParser(description="Filter data by LDS (deepseek-v3)")
    parser.add_argument('--inference_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    filter_dir(args.inference_dir, args.data_dir, args.output_dir)
    print(f"Filtering complete. Output saved to {args.output_dir}")

if __name__ == '__main__':
    main() 