"""
统一的优化模型实现
支持本地模型加载，只加载必要的组件以减少内存使用
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import os
import json
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class OptimizedModel(nn.Module):
    """
    优化模型，只加载必要的组件：
    - 词嵌入层 (embed_tokens)
    - 第一层Transformer (layers.0)
    - 最终层归一化 (norm)
    """
    
    def __init__(self, config, model_path: str, local_path: str = None):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.local_path = local_path or model_path
        
        # 初始化组件
        self.embed_tokens = None
        self.layers = None
        self.norm = None
        
        # 加载必要的组件
        self._load_components()
        
    def _load_components(self):
        """加载必要的组件"""
        print(f"正在从 {self.local_path} 加载必要的组件...")
        
        # 加载完整模型（仅用于提取权重）
        with torch.no_grad():
            full_model = AutoModelForCausalLM.from_pretrained(
                self.local_path,
                torch_dtype=torch.float16,
                device_map="cpu",  # 先加载到CPU
                trust_remote_code=True
            )
            
            # 提取词嵌入层
            if hasattr(full_model.model, 'embed_tokens'):
                self.embed_tokens = full_model.model.embed_tokens
                print("✓ 词嵌入层加载完成")
            else:
                raise ValueError("模型没有找到 embed_tokens 层")
            
            # 提取第一层Transformer
            if hasattr(full_model.model, 'layers') and len(full_model.model.layers) > 0:
                self.layers = nn.ModuleList([full_model.model.layers[0]])
                print("✓ 第一层Transformer加载完成")
            else:
                raise ValueError("模型没有找到 layers")
            
            # 提取最终层归一化
            if hasattr(full_model.model, 'norm'):
                self.norm = full_model.model.norm
                print("✓ 最终层归一化加载完成")
            else:
                raise ValueError("模型没有找到 norm 层")
            
            # 清理完整模型以释放内存
            del full_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        print("所有必要组件加载完成！")
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """前向传播"""
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 第一层Transformer
        if self.layers is not None and len(self.layers) > 0:
            layer_outputs = self.layers[0](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        
        # 最终层归一化
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        
        return hidden_states


def download_model_to_local(
    model_path: str,
    local_dir: str = None,
    cache_dir: str = None
) -> str:
    """
    将HuggingFace模型下载到本地
    
    Args:
        model_path: HuggingFace模型路径 (例如 "deepseek-ai/DeepSeek-V3-0324")
        local_dir: 本地保存目录，如果为None则使用模型名称
        cache_dir: 缓存目录
    
    Returns:
        str: 本地模型路径
    """
    if local_dir is None:
        # 从模型路径提取模型名称
        model_name = model_path.split('/')[-1]
        local_dir = f"./models/{model_name}"
    
    # 创建本地目录
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"正在下载模型 {model_path} 到本地目录: {local_dir}")
    
    try:
        # 下载模型文件
        snapshot_download(
            repo_id=model_path,
            local_dir=local_dir,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )
        
        print(f"✓ 模型下载完成: {local_dir}")
        return local_dir
        
    except Exception as e:
        print(f"模型下载失败: {e}")
        raise


def load_optimized_model(
    model_path: str,
    local_path: str = None,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    download_if_not_exists: bool = True
) -> OptimizedModel:
    """
    加载优化模型
    
    Args:
        model_path: HuggingFace模型路径
        local_path: 本地模型路径，如果为None则自动检测
        device: 设备 ("auto", "cpu", "cuda", "cuda:0" 等)
        torch_dtype: 数据类型
        download_if_not_exists: 如果本地不存在是否自动下载
    
    Returns:
        OptimizedModel: 优化后的模型
    """
    print(f"正在加载优化模型: {model_path}")
    
    # 确定本地路径
    if local_path is None:
        model_name = model_path.split('/')[-1]
        local_path = f"./models/{model_name}"
    
    # 检查本地模型是否存在
    if not os.path.exists(local_path):
        if download_if_not_exists:
            print(f"本地模型不存在，正在下载...")
            local_path = download_model_to_local(model_path, local_path)
        else:
            raise FileNotFoundError(f"本地模型不存在: {local_path}")
    else:
        print(f"使用本地模型: {local_path}")
    
    # 加载配置
    config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
    print(f"模型配置: {config.model_type}")
    print(f"隐藏层大小: {config.hidden_size}")
    print(f"注意力头数: {config.num_attention_heads}")
    print(f"层数: {config.num_hidden_layers}")
    
    # 创建优化模型
    model = OptimizedModel(config, model_path, local_path)
    
    # 设置数据类型
    model = model.to(dtype=torch_dtype)
    
    # 设置设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device != "cpu":
        model = model.to(device)
    
    # 计算模型大小
    total_params = sum(p.numel() for p in model.parameters())
    total_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    
    print(f"优化模型加载完成!")
    print(f"参数数量: {total_params:,}")
    print(f"模型大小: {total_size_gb:.2f} GB")
    print(f"设备: {device}")
    
    return model


def get_model_info(model_path: str, local_path: str = None) -> dict:
    """
    获取模型信息
    
    Args:
        model_path: HuggingFace模型路径
        local_path: 本地模型路径
    
    Returns:
        dict: 模型信息
    """
    if local_path is None:
        model_name = model_path.split('/')[-1]
        local_path = f"./models/{model_name}"
    
    if not os.path.exists(local_path):
        # 如果本地不存在，从HuggingFace获取
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    else:
        # 从本地获取
        config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
    
    info = {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
        "rope_theta": getattr(config, 'rope_theta', None),
        "sliding_window": getattr(config, 'sliding_window', None),
        "local_path": local_path,
        "exists_locally": os.path.exists(local_path)
    }
    
    return info


def list_local_models() -> list:
    """
    列出本地已下载的模型
    
    Returns:
        list: 本地模型列表
    """
    models_dir = "./models"
    if not os.path.exists(models_dir):
        return []
    
    local_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            # 检查是否包含必要的文件
            config_file = os.path.join(model_path, "config.json")
            if os.path.exists(config_file):
                local_models.append({
                    "name": model_name,
                    "path": model_path,
                    "config_file": config_file
                })
    
    return local_models


if __name__ == "__main__":
    # 测试代码
    model_path = "deepseek-ai/DeepSeek-V3-0324"
    
    print("=== 模型信息 ===")
    info = get_model_info(model_path)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n=== 本地模型列表 ===")
    local_models = list_local_models()
    if local_models:
        for model in local_models:
            print(f"- {model['name']}: {model['path']}")
    else:
        print("没有找到本地模型")
    
    print("\n=== 加载优化模型 ===")
    model = load_optimized_model(model_path)
    
    print("\n=== 测试前向传播 ===")
    # 创建测试输入
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    test_text = "Hello, this is a test."
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"输入形状: {inputs['input_ids'].shape}")
        print(f"输出形状: {outputs.shape}")
        print("测试成功！") 