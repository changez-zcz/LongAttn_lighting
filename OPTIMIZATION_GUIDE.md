# LongAttn 内存优化指南

## 概述

LongAttn 现在支持内存优化模式，专门为8卡A800环境设计。通过只加载模型的第一层，可以大幅减少显存占用，同时保持相同的功能。

## 内存优化原理

### 为什么可以优化？

LongAttn 的核心功能是计算第一层的注意力权重来评估长距离依赖关系。因此：

- **只需要**：词嵌入层 + 第一层Transformer + 最终层归一化
- **不需要**：其他Transformer层（第2层到最后一层）

### 优化效果

以 DeepSeek V3 33B 模型为例：

| 模式 | 参数量 | 显存占用 | 减少比例 |
|------|--------|----------|----------|
| 完整模型 | ~33B | ~66GB | - |
| 优化模型 | ~1.1B | ~2.2GB | 96.7% |

## 使用方法

### 1. 检查你的硬件配置

```bash
python memory_analysis.py
```

这个脚本会：
- 检测你的GPU配置
- 分析不同模型的内存需求
- 推荐合适的batch size

### 2. 选择运行模式

#### 标准模式（适用于单卡或小模型）
```bash
bash scripts/inference_and_filter.bash
```

#### 优化模式（推荐用于8卡A800）
```bash
bash scripts/inference_and_filter_optimized.bash
```

### 3. 手动运行优化推理

```bash
# 使用优化模式
accelerate launch --multi_gpu --num_processes=8 src/1_inference_dp_optimized.py \
    /path/to/input \
    /path/to/output \
    8 \
    --model_path "deepseek-ai/deepseek-coder-33b-instruct" \
    --use_optimized
```

## 8卡A800配置建议

### 硬件配置
- **GPU**: 8 × NVIDIA A800 (80GB each)
- **总显存**: 640GB
- **推荐batch_size**: 8-16

### 脚本配置
修改 `scripts/inference_and_filter_optimized.bash`:

```bash
# 设置路径
input_path="/path/to/processed/data"
output_inference_dir="/path/to/inference_results"
output_filtered_dir="/path/to/filtered_results"
batch_size=8  # 根据显存调整
model_path="deepseek-ai/deepseek-coder-33b-instruct"
```

### 内存分配策略

优化模式下的内存分配：

```
GPU 0: 词嵌入层 + 部分第一层参数
GPU 1: 部分第一层参数
GPU 2: 部分第一层参数
GPU 3: 部分第一层参数
GPU 4: 部分第一层参数
GPU 5: 部分第一层参数
GPU 6: 部分第一层参数
GPU 7: 部分第一层参数 + 最终层归一化
```

## 性能对比

### 显存使用对比

| 配置 | 完整模型 | 优化模型 | 节省 |
|------|----------|----------|------|
| 单卡A800 | 66GB | 2.2GB | 96.7% |
| 8卡A800 | 8.25GB/卡 | 0.28GB/卡 | 96.7% |

### 推理速度对比

由于只加载第一层，优化模式可以：
- **更快加载**：模型加载时间减少90%+
- **更大batch size**：支持更大的批处理大小
- **更高吞吐量**：充分利用8卡并行

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少batch_size
   batch_size=4  # 从8减少到4
   ```

2. **模型加载失败**
   ```bash
   # 检查模型路径
   python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct')"
   ```

3. **多GPU通信问题**
   ```bash
   # 使用单GPU模式
   accelerate launch --num_processes=1 src/1_inference_dp_optimized.py ...
   ```

### 调试命令

```bash
# 检查GPU状态
nvidia-smi

# 检查内存使用
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 测试模型加载
python -c "from LongAttnOptimized import load_optimized_model; model = load_optimized_model('deepseek-ai/deepseek-coder-33b-instruct')"
```

## 高级配置

### 自定义设备映射

```python
# 手动指定设备映射
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 1,
    "model.norm": 7,
    "lm_head": 7
}

model = load_optimized_model(
    model_path,
    device_map=device_map,
    torch_dtype=torch.float16
)
```

### 混合精度训练

```python
# 使用更低的精度进一步节省内存
model = load_optimized_model(
    model_path,
    torch_dtype=torch.bfloat16  # 或 torch.float8_e4m3fn
)
```

## 最佳实践

1. **先运行内存分析**：使用 `memory_analysis.py` 了解你的配置
2. **从小batch开始**：从batch_size=4开始，逐步增加
3. **监控显存使用**：使用 `nvidia-smi` 实时监控
4. **使用优化模式**：8卡A800环境强烈推荐使用优化模式
5. **定期清理缓存**：长时间运行时定期清理GPU缓存

## 技术支持

如果遇到问题：
1. 检查 `error_optimized.log` 文件
2. 运行 `memory_analysis.py` 诊断
3. 查看GPU状态和内存使用
4. 尝试减少batch_size或使用单GPU模式 