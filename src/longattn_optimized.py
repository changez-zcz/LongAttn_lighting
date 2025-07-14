import torch
import torch.nn as nn
from transformers import AutoConfig
from safetensors.torch import load_file
import os

class LongAttnOptimized(nn.Module):
    def __init__(self, config, state_dict=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 只加载第一层
        self.layer0 = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True
        )
        if state_dict:
            self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def from_pretrained(model_dir, index_json=None, device='cuda'):
        config = AutoConfig.from_pretrained(model_dir)
        state_dict = None
        if index_json:
            # 只加载必要权重
            state_dict = load_partial_state_dict(model_dir, index_json)
        model = LongAttnOptimized(config, state_dict=state_dict)
        model.to(device)
        return model

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        x = self.layer0(x)
        x = self.norm(x)
        return x

def load_partial_state_dict(model_dir, index_json):
    import json
    with open(index_json, 'r') as f:
        index = json.load(f)
    needed_prefix = ["model.embed_tokens.weight", "model.layers.0", "model.norm"]
    needed_shards = set()
    for k, v in index["weight_map"].items():
        if any(k.startswith(p) for p in needed_prefix):
            needed_shards.add(v)
    state_dict = {}
    for shard in needed_shards:
        shard_path = os.path.join(model_dir, shard)
        sd = load_file(shard_path)
        for k, v in sd.items():
            if any(k.startswith(p) for p in needed_prefix):
                state_dict[k] = v
    return state_dict 