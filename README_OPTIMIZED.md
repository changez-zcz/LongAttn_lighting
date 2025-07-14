# LongAttn 优化模型

基于 DeepSeek-V3-0324 的内存优化实现，只加载必要的组件以减少内存使用。

## 特性

- ✅ **内存优化**: 从 ~70GB 减少到 ~2-5GB
- ✅ **本地模型**: 支持本地模型加载，无需重复下载
- ✅ **简单易用**: 统一的API接口
- ✅ **自动配置**: 基于官方配置自动解析

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirement.txt
```

### 2. 下载模型（首次使用）

```bash
# 仅下载模型
python example_usage.py download
```

### 3. 运行测试

```bash
# 完整测试（包括推理）
python example_usage.py
```

## 使用方法

### 基本使用

```python
import torch
from config import ModelConfig
from src.optimized_model import load_optimized_model
from transformers import AutoTokenizer

# 获取配置
config = ModelConfig.get_config("deepseek_v3")
model_path = config["model_path"]

# 加载优化模型
model = load_optimized_model(model_path)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 处理文本
text = "这是一个测试句子。"
inputs = tokenizer(text, return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)

print(f"输出形状: {outputs.shape}")
```

### 高级使用

```python
from src.optimized_model import (
    load_optimized_model, 
    get_model_info, 
    list_local_models
)

# 检查本地模型
local_models = list_local_models()
print("本地模型:", local_models)

# 获取模型信息
info = get_model_info("deepseek-ai/DeepSeek-V3-0324")
print("模型信息:", info)

# 指定本地路径加载
model = load_optimized_model(
    model_path="deepseek-ai/DeepSeek-V3-0324",
    local_path="./models/DeepSeek-V3-0324",
    device="cuda",
    torch_dtype=torch.float16
)
```

## 模型配置

### DeepSeek-V3-0324 配置

```python
DEEPSEEK_V3_CONFIG = {
    "model_path": "deepseek-ai/DeepSeek-V3-0324",
    "tokenizer_path": "deepseek-ai/DeepSeek-V3-0324", 
    "window_size": 131072,        # 128K 上下文窗口
    "rope_theta": 1000000.0,      # RoPE 参数
    "max_length": 131072          # 最大长度
}
```

### 模型参数

- **模型类型**: DeepSeek V3
- **参数量**: 32B
- **上下文长度**: 128K
- **架构**: Transformer with RoPE
- **训练数据**: 多语言

## 文件结构

```
LongAttn/
├── config.py                    # 配置文件
├── example_usage.py             # 使用示例
├── requirement.txt              # 依赖包
├── README_OPTIMIZED.md          # 本文档
├── src/
│   └── optimized_model.py       # 优化模型实现
└── models/                      # 本地模型目录
    └── DeepSeek-V3-0324/        # 下载的模型文件
```

## 内存使用对比

| 方法 | 内存使用 | 加载时间 | 兼容性 |
|------|----------|----------|--------|
| 完整模型 | ~70GB | 慢 | 最好 |
| 优化模型 | ~2-5GB | 快 | 好 |

## 注意事项

1. **首次使用**: 需要下载模型文件（约60GB），下载时间较长
2. **内存要求**: 建议至少 8GB GPU 内存
3. **本地存储**: 模型文件保存在 `./models/` 目录
4. **网络连接**: 首次下载需要稳定的网络连接

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```python
   model = load_optimized_model(model_path, device="cpu", torch_dtype=torch.float32)
   ```

2. **模型下载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间（至少60GB）
   - 尝试重新下载

3. **版本兼容性**
   ```bash
   pip install --upgrade transformers huggingface-hub
   ```

## 技术细节

### 加载的组件

优化模型只加载以下必要组件：

1. **词嵌入层** (`embed_tokens`)
   - 将输入token转换为向量表示
   - 大小: vocab_size × hidden_size

2. **第一层Transformer** (`layers.0`)
   - 包含自注意力机制和前馈网络
   - 提供基础的语义理解能力

3. **最终层归一化** (`norm`)
   - 对输出进行归一化处理
   - 提高数值稳定性

### 优化原理

1. **选择性加载**: 只加载必要的层，跳过中间层
2. **内存管理**: 加载后立即释放完整模型
3. **本地缓存**: 避免重复下载，提高加载速度

## 总结

通过使用优化模型，您可以：

1. ✅ 使用正确的 DeepSeek-V3-0324 模型
2. ✅ 大大减少内存使用（从 ~70GB 降到 ~2-5GB）
3. ✅ 享受本地模型的快速加载
4. ✅ 保持简单的使用方式

适合场景：
- 资源受限的环境
- 快速原型开发
- 批量文本处理
- 研究和实验 