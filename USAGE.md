# LongAttn 使用指南

## 项目概述

LongAttn 是一个基于token级别注意力机制的长上下文训练数据选择框架。该项目现在支持两种模型：

1. **LLaMA-3**: 32K上下文窗口
2. **DeepSeek V3**: 128K上下文窗口（默认）

## 环境要求

- Python 3.8+
- CUDA 11.8+
- 至少 80GB GPU 显存（用于DeepSeek V3 33B模型）

## 安装步骤

```bash
# 克隆项目
git clone https://github.com/Lyun0912-wu/LongAttn.git
cd LongAttn

# 安装依赖
pip install -r requirements.txt
```

## 数据格式

### 支持的输入格式

#### 格式1：单列JSON格式
```json
{"data_id": "123", "text": "这是要处理的文本内容"}
{"data_id": "456", "text": "另一段文本内容"}
```

#### 格式2：两列TSV格式（data_id + JSON）
```
123	{"text": "这是要处理的文本内容"}
456	{"text": "另一段文本内容"}
```

#### 格式3：两列TSV格式（data_id + 包含text字段的JSON）
```
sample_001	{"text": "文本内容", "other_field": "其他数据"}
sample_002	{"text": "更多文本", "metadata": {"source": "web"}}
```

### 输出数据格式
预处理后的数据格式：
```json
{"data_id": "123", "content": "这是要处理的文本内容"}
{"data_id": "456", "content": "另一段文本内容"}
```

### 关键要求
1. **必须包含的字段**：
   - `data_id`：数据的唯一标识符
   - `text`：需要处理的文本内容

2. **文件格式**：
   - 文本文件，每行一条记录
   - 如果是两列格式，使用tab分隔符（`\t`）

3. **JSON要求**：
   - 至少有一列必须是有效的JSON字符串
   - JSON中必须包含 `text` 字段
   - 如果JSON中有 `data_id` 字段，会优先使用JSON中的ID

## 使用步骤

### 1. 数据预处理

修改 `scripts/Preprocess.bash` 中的路径配置：

```bash
# 设置输入输出路径
INPUT_FOLDER="/path/to/your/input/data"     # 输入数据文件夹（支持多级目录）
OUTPUT_FOLDER="/path/to/processed/data"     # 预处理输出文件夹（保持目录结构）
```

运行预处理：
```bash
bash scripts/Preprocess.bash
```

### 2. 推理和过滤

修改 `scripts/inference_and_filter.bash` 中的路径配置：

```bash
# 设置目录路径
input_path="/path/to/processed/data"              # 预处理后的数据目录
output_inference_dir="/path/to/inference_results" # 推理结果目录
output_filtered_dir="/path/to/filtered_results"   # 最终过滤结果目录
batch_size=6                                      # 批处理大小
```

运行推理和过滤：
```bash
bash scripts/inference_and_filter.bash
```

## 模型配置

### 使用DeepSeek V3（默认）

当前配置已设置为使用DeepSeek V3 128K模型：
- 模型路径: `deepseek-ai/deepseek-coder-33b-instruct`
- 上下文窗口: 131,072 tokens (128K)
- RoPE theta: 1,000,000

### 切换到LLaMA-3

如果需要使用LLaMA-3模型，需要修改以下文件：

1. `src/0_pre-process.py`:
```python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
```

2. `src/1_inference_dp.py`:
```python
model_path = "meta-llama/Meta-Llama-3.1-70B"
config_kwargs = {
    "rope_theta": 2500000.0,
}
```

3. `scripts/Preprocess.bash`:
```bash
WINDOW_SIZE=32768  # 改为32K
```

## 参数说明

### 预处理参数
- `--input_folder`: 输入数据文件夹路径
- `--output_folder`: 输出文件夹路径
- `--max_workers`: 最大工作进程数（默认10）
- `--batch_size`: 批处理大小（默认100）
- `--window_size`: 滑动窗口大小（DeepSeek V3: 131072, LLaMA-3: 32768）
- `--sample_size`: 采样数量
- `--prefix`: 数据ID前缀

### 推理参数
- `input_file_path`: 输入文件路径
- `output_file_path`: 输出文件路径
- `batch_size`: 批处理大小（建议根据GPU显存调整）

## 性能优化建议

1. **GPU显存优化**:
   - 对于DeepSeek V3 33B模型，建议使用至少80GB显存
   - 可以通过调整batch_size来控制显存使用

2. **多GPU支持**:
   - 项目支持Accelerate分布式推理
   - 使用 `accelerate launch` 命令启动

3. **内存优化**:
   - 对于大数据集，建议分批处理
   - 使用多进程并行处理

## 故障排除

### 常见问题

1. **CUDA内存不足**:
   - 减少batch_size
   - 使用更小的模型或减少上下文长度

2. **模型下载失败**:
   - 检查网络连接
   - 确保有足够的磁盘空间
   - 对于LLaMA模型，需要Hugging Face访问权限

3. **文件路径错误**:
   - 确保所有路径都是绝对路径或正确的相对路径
   - 检查文件权限

### 日志文件

推理过程中的错误日志会保存在 `error.log` 文件中，可以查看该文件来诊断问题。

## 示例

### 完整处理流程示例

```bash
# 1. 设置环境变量（可选）
export CUDA_VISIBLE_DEVICES=0

# 2. 预处理数据
bash scripts/Preprocess.bash

# 3. 推理和过滤
bash scripts/inference_and_filter.bash

# 4. 检查结果
ls -la /path/to/final_filtered_data.jsonl
```

### 自定义配置示例

```python
# 使用配置文件
from config import ModelConfig

# 获取DeepSeek V3配置
config = ModelConfig.get_config("deepseek_v3")
print(f"Model path: {config['model_path']}")
print(f"Window size: {config['window_size']}")

# 获取LLaMA-3配置
config = ModelConfig.get_config("llama3")
print(f"Model path: {config['model_path']}")
print(f"Window size: {config['window_size']}")
``` 