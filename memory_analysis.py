#!/usr/bin/env python3
"""
Memory usage analysis for LongAttn optimization
"""

import torch
import os
import json
from transformers import AutoConfig

def analyze_model_memory_usage(model_path, use_optimized=True):
    """Analyze memory usage for different model loading strategies"""
    
    print("=== LongAttn Memory Usage Analysis ===\n")
    
    # Load config
    config = AutoConfig.from_pretrained(model_path)
    
    print(f"Model: {model_path}")
    print(f"Original layers: {config.num_hidden_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Attention heads: {config.num_attention_heads}")
    
    # Calculate parameter counts
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    
    # Embedding parameters
    embedding_params = vocab_size * hidden_size
    
    # Per-layer parameters (attention + MLP)
    attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
    mlp_params = 3 * hidden_size * (4 * hidden_size)  # MLP layers
    layer_norm_params = 2 * hidden_size  # Two layer norms per layer
    per_layer_params = attention_params + mlp_params + layer_norm_params
    
    # Total parameters
    total_params_full = embedding_params + (num_layers * per_layer_params) + hidden_size  # + final norm
    
    # Optimized parameters (only first layer)
    total_params_optimized = embedding_params + per_layer_params + hidden_size
    
    # Memory usage (assuming float16)
    bytes_per_param = 2  # float16 = 2 bytes
    memory_full_gb = (total_params_full * bytes_per_param) / (1024**3)
    memory_optimized_gb = (total_params_optimized * bytes_per_param) / (1024**3)
    
    print(f"\n=== Parameter Count ===")
    print(f"Full model parameters: {total_params_full:,}")
    print(f"Optimized model parameters: {total_params_optimized:,}")
    print(f"Reduction: {((total_params_full - total_params_optimized) / total_params_full * 100):.1f}%")
    
    print(f"\n=== Memory Usage (float16) ===")
    print(f"Full model: {memory_full_gb:.2f} GB")
    print(f"Optimized model: {memory_optimized_gb:.2f} GB")
    print(f"Memory saved: {memory_full_gb - memory_optimized_gb:.2f} GB")
    
    # Multi-GPU analysis
    print(f"\n=== Multi-GPU Analysis ===")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        
        # A800 has 80GB memory
        gpu_memory_gb = 80
        total_gpu_memory = num_gpus * gpu_memory_gb
        
        print(f"Total GPU memory: {total_gpu_memory} GB")
        
        # Memory per GPU for optimized model
        memory_per_gpu_optimized = memory_optimized_gb / num_gpus
        memory_per_gpu_full = memory_full_gb / num_gpus
        
        print(f"Memory per GPU (optimized): {memory_per_gpu_optimized:.2f} GB")
        print(f"Memory per GPU (full): {memory_per_gpu_full:.2f} GB")
        
        # Check if it fits
        if memory_per_gpu_optimized <= gpu_memory_gb:
            print("✅ Optimized model fits in GPU memory")
        else:
            print("❌ Optimized model exceeds GPU memory")
            
        if memory_per_gpu_full <= gpu_memory_gb:
            print("✅ Full model fits in GPU memory")
        else:
            print("❌ Full model exceeds GPU memory")
        
        # Batch size recommendations
        print(f"\n=== Batch Size Recommendations ===")
        
        # Estimate memory for different batch sizes
        seq_length = 131072  # 128K tokens
        hidden_size = config.hidden_size
        
        for batch_size in [1, 2, 4, 8, 16, 32]:
            # Approximate memory for activations
            activation_memory = batch_size * seq_length * hidden_size * 2  # float16
            activation_memory_gb = activation_memory / (1024**3)
            
            total_memory_gb = memory_per_gpu_optimized + activation_memory_gb
            
            if total_memory_gb <= gpu_memory_gb * 0.8:  # Leave 20% buffer
                print(f"Batch size {batch_size}: {total_memory_gb:.2f} GB ✅")
            else:
                print(f"Batch size {batch_size}: {total_memory_gb:.2f} GB ❌")
    
    return {
        "full_params": total_params_full,
        "optimized_params": total_params_optimized,
        "full_memory_gb": memory_full_gb,
        "optimized_memory_gb": memory_optimized_gb,
        "reduction_percent": ((total_params_full - total_params_optimized) / total_params_full * 100)
    }

def check_gpu_setup():
    """Check current GPU setup"""
    print("=== GPU Setup Check ===\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {memory_gb:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Check if it's A800
        if "A800" in props.name:
            print(f"  ✅ A800 detected")
        elif "A100" in props.name:
            print(f"  ✅ A100 detected")
        else:
            print(f"  ⚠️  Other GPU type")

def main():
    """Main analysis function"""
    
    # Check GPU setup
    check_gpu_setup()
    print()
    
    # Analyze different models
    models = [
        "deepseek-ai/deepseek-coder-33b-instruct",
        "meta-llama/Meta-Llama-3.1-70B"
    ]
    
    results = {}
    
    for model_path in models:
        print(f"\n{'='*50}")
        try:
            result = analyze_model_memory_usage(model_path, use_optimized=True)
            results[model_path] = result
        except Exception as e:
            print(f"Error analyzing {model_path}: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("=== SUMMARY ===")
    for model_path, result in results.items():
        print(f"\n{model_path}:")
        print(f"  Memory reduction: {result['reduction_percent']:.1f}%")
        print(f"  Memory saved: {result['full_memory_gb'] - result['optimized_memory_gb']:.2f} GB")

if __name__ == "__main__":
    main() 