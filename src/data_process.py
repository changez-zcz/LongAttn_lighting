import os
import json
import jsonlines
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def parse_line(line):
    line = line.strip()
    if not line:
        return None
    parts = line.split('\t')
    # 单列JSON
    if len(parts) == 1:
        try:
            json_data = json.loads(parts[0])
            data_id = json_data.get('data_id')
            text = json_data.get('text')
            if data_id and text:
                return data_id, text
        except Exception:
            pass
    # 两列TSV
    if len(parts) >= 2:
        data_id = parts[0].strip()
        json_str = parts[1].strip()
        try:
            json_data = json.loads(json_str)
            text = json_data.get('text')
            if text:
                if 'data_id' in json_data:
                    data_id = json_data['data_id']
                return data_id, text
        except Exception:
            pass
    return None

def process_batch(batch):
    processed = []
    for line in batch:
        result = parse_line(line)
        if result:
            data_id, text = result
            processed.append({"data_id": data_id, "content": text})
    return processed

def process_file(input_path, output_path, batch_size=100):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    batches = [lines[i:i+batch_size] for i in range(0, len(lines), batch_size)]
    with jsonlines.open(output_path, 'w') as writer:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{os.path.basename(input_path)}"):
                result = future.result()
                if result:
                    writer.write_all(result)

def process_dir(input_dir, output_dir, batch_size=100):
    os.makedirs(output_dir, exist_ok=True)
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    for file_path in tqdm(files, desc="Preprocessing files"):
        rel_path = os.path.relpath(file_path, input_dir)
        out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.jsonl')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        process_file(file_path, out_path, batch_size=batch_size)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess raw data to standardized jsonl format.")
    parser.add_argument('--input_dir', required=True, help='Input directory (supports nested)')
    parser.add_argument('--output_dir', required=True, help='Output directory (will mirror structure)')
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()
    process_dir(args.input_dir, args.output_dir, batch_size=args.batch_size)
    print(f"Preprocessing complete. Output saved to {args.output_dir}")

if __name__ == '__main__':
    main() 