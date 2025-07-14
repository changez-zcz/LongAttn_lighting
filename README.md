# 🔥LongAttn ：Selecting Long-context Training Data via Token-level Attention
<div align="center">
  <img src="https://img.shields.io/badge/Self--Attention-black" alt="self-attention">
  <img src="https://img.shields.io/badge/Pre--trained-Data-red" alt="Data">
  <img src="https://img.shields.io/badge/Long--Context-red" alt="LongContext">
</div>

</div> 
<div align="center">
<a href="assets/LongAttn.pdf"><img src="assets/paper-page-xl.svg" alt="Paper page"></a>
<a href=""><img src="assets/dataset-on-hf-xl.svg" alt="Dataset on HF"></a>
<a href=""><img src="assets/model-on-hf-xl.svg" alt="Model on HF"></a>
</div>

## 👀Overview

This repository contains the code for the paper "LongAttn ：Selecting Long-context Training Data via Token-level Attention"

<p align="center">
<img src=assets/LongAttn.png width=700/>
</p>

In this work, we propose a novel token-level framework, LongAttn, which leverages the self-attention mechanism of LLMs to measure the long-range dependencies for the data. By calculating token-level dependency strength and distribution uniformity of token scores, LongAttn effectively quantifies **long-range dependencies**, enabling more accurate and efficient data selection. 

## 🔍Released Models & Datasetes
To facilitate future research in filtering long-context data, we release **LongAttn-8B** and **LongABC-32K**.
- **LongABC-32K-Raw** is a 32K-length long-context dataset obtained from the open-source **A**rxiv, **B**ook, and **C**ode data after preprocessing, with each category containing 12 billion tokens.
- **LongABC-32K** is the dataset we filtered using **LongAttn** based on LLaMA-3.1-70B from **LongABC-32K-Raw**, with each category containing 1.5 billion tokens.
- **LongAttn-8B** is the model we continual pre-trained on **LLaMA-3-8B** using **LongABC-32K**. Experiments have demonstrated that the model trained on 4.5B tokens from **LongABC-32K** outperforms the model trained on 18B tokens randomly selected from **LongABC-32K-Raw**.

The download links for the model and data are as follows:

| Model & Data | Link |
| --- | --- |
| **LongABC-32K** | [Coming soon]() |
| **LongAttn-8B** | [Coming soon]() |

## 🚀Quick Start
### Environment Setup

To use LongAttn, follow these steps to download the code and necessary dependencies:
```
git clone https://github.com/Lyun0912-wu/LongAttn.git
cd LongAttn
pip install -r requirements.txt
```

### Model Configuration
This project now supports both LLaMA-3 and DeepSeek V3 models:

- **LLaMA-3 Version**: Uses LLaMA-3.1-70B for inference with 32K context window
- **DeepSeek V3 Version**: Uses DeepSeek V3 128K for inference with 128K context window (default)

The current configuration uses DeepSeek V3 128K model. If you want to use LLaMA-3, modify the model paths in the source files.

### Memory Optimization
For 8-GPU A800 setups, we provide memory-optimized inference that only loads the first layer of the model:

- **Standard Mode**: Loads full model (requires more GPU memory)
- **Optimized Mode**: Loads only embeddings + first layer (recommended for 8-GPU setup)

Run memory analysis to check your setup:
```bash
python memory_analysis.py
```

### Pre-process
After that, the bash script `Preprocess.bash` will preprocess the pre-training data you have selected.
```
bash scripts/Preprocess.bash
```
The script performs the pipeline of Preprocess:
1. Parse input data in multiple formats (JSON, TSV) and extract text content with data IDs.
2. Convert data to standardized JSONL format for subsequent processing.
3. Preserve directory structure from input to output folders.
4. Support nested directories and multiple file formats.

### Filtering
Then, we use the script `inference_and_filter.bash` to filter the data.

```
bash scripts/inference_and_filter.bash
```

For 8-GPU A800 setups, use the optimized version:
```bash
bash scripts/inference_and_filter_optimized.bash
```

The script performs the pipeline of filtering:
1. Use the filtering model to obtain the **Attention map** for the labeled data.
2. Calculate the **$LDS_T$** indicator, and filter the data accordingly.
3. Process all files in the input directory while preserving directory structure.

Finally, we obtained the data filtered by the **LongAttn** framework, organized in the same directory structure as the input.

## 📑The Data Format

### Supported Input Formats

#### Format 1: Single Column JSON
```json
{"data_id": "123", "text": "This is the text content to be processed"}
{"data_id": "456", "text": "Another piece of text content"}
```

#### Format 2: Two Column TSV (data_id + JSON)
```
123	{"text": "This is the text content to be processed"}
456	{"text": "Another piece of text content"}
```

#### Format 3: Two Column TSV with Additional Fields
```
sample_001	{"text": "Text content", "other_field": "other data"}
sample_002	{"text": "More text", "metadata": {"source": "web"}}
```

### Data after Preprocessing
```json
{"data_id": "123", "content": "This is the text content to be processed"}
{"data_id": "456", "content": "Another piece of text content"}
```

### Key Requirements
1. **Required Fields**:
   - `data_id`: Unique identifier for the data
   - `text`: Text content to be processed

2. **File Format**:
   - Text files with one record per line
   - For two-column format, use tab separator (`\t`)

3. **JSON Requirements**:
   - At least one column must be valid JSON string
   - JSON must contain `text` field
   - If JSON has `data_id` field, it will be used instead of the first column

## 🔧Configuration
### Model Settings
- **DeepSeek V3 128K**: `deepseek-ai/deepseek-coder-33b-instruct`
- **LLaMA-3 32K**: `meta-llama/Meta-Llama-3.1-70B`

### Context Window Sizes
- **DeepSeek V3**: 131,072 tokens (128K)
- **LLaMA-3**: 32,768 tokens (32K)

### RoPE Configuration
- **DeepSeek V3**: `rope_theta=1000000.0`
- **LLaMA-3**: `rope_theta=2500000.0`

## 🌟 Citation
If you find this repo helpful, please cite our paper as follows:

```bibtex
@article{wu2025longattn,
  title={LongAttn: Selecting Long-context Training Data via Token-level Attention},
  author={Wu, Longyun and Zhu, Dawei and Zhao, Guangxiang and Yu, Zhuocheng and Ran, Junfeng and Wong, Xiangyu and Sun, Lin and Li, Sujian},
  journal={arXiv preprint arXiv:2502.16860},
  year={2025}
}
```