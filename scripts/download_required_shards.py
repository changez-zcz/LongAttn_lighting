import json
import os
import requests

MODEL_INDEX = "DeepSeek-V3-0324/model.safetensors.index.json"
BASE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/resolve/main/"
OUT_DIR = "DeepSeek-V3-0324"

NEEDED_PREFIX = ["model.embed_tokens.weight", "model.layers.0", "model.norm"]

def download_shard(url, out_path):
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        return
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    with open(MODEL_INDEX, "r") as f:
        index = json.load(f)
    needed_shards = set()
    for k, v in index["weight_map"].items():
        if any(k.startswith(prefix) for prefix in NEEDED_PREFIX):
            needed_shards.add(v)
    print("Needed shards:", needed_shards)
    os.makedirs(OUT_DIR, exist_ok=True)
    for shard in needed_shards:
        url = BASE_URL + shard
        out_path = os.path.join(OUT_DIR, shard)
        download_shard(url, out_path)

if __name__ == "__main__":
    main() 