#!/usr/bin/env python3
"""
模型下载脚本
用于下载 DeepSeek-V3-0324 模型到本地
"""

import os
import sys
from config import ModelConfig
from src.optimized_model import download_model_to_local, get_model_info


def main():
    """主函数"""
    print("=" * 60)
    print("DeepSeek-V3-0324 模型下载工具")
    print("=" * 60)
    
    # 获取配置
    config = ModelConfig.get_config("deepseek_v3")
    model_path = config["model_path"]
    
    print(f"模型路径: {model_path}")
    print(f"模型名称: {model_path.split('/')[-1]}")
    
    # 显示模型信息
    print("\n=== 模型信息 ===")
    try:
        info = get_model_info(model_path)
        print(f"模型类型: {info['model_type']}")
        print(f"隐藏层大小: {info['hidden_size']}")
        print(f"注意力头数: {info['num_attention_heads']}")
        print(f"层数: {info['num_hidden_layers']}")
        print(f"词汇表大小: {info['vocab_size']}")
        print(f"最大位置编码: {info['max_position_embeddings']}")
        print(f"RoPE theta: {info['rope_theta']}")
        print(f"滑动窗口: {info['sliding_window']}")
    except Exception as e:
        print(f"获取模型信息失败: {e}")
        print("继续下载...")
    
    # 确认下载
    print(f"\n=== 下载确认 ===")
    print("⚠️  注意:")
    print("1. 模型文件大小约 60GB")
    print("2. 下载时间取决于网络速度")
    print("3. 需要确保有足够的磁盘空间")
    print("4. 下载过程中请保持网络连接稳定")
    
    # 检查磁盘空间
    try:
        import shutil
        total, used, free = shutil.disk_usage("./")
        free_gb = free // (1024**3)
        print(f"\n可用磁盘空间: {free_gb} GB")
        
        if free_gb < 70:
            print("⚠️  警告: 可用磁盘空间不足70GB，可能无法完成下载")
            response = input("是否继续下载? (y/N): ")
            if response.lower() != 'y':
                print("下载已取消")
                return
    except:
        print("无法检查磁盘空间，继续下载...")
    
    # 开始下载
    print(f"\n=== 开始下载 ===")
    print("正在下载模型文件...")
    print("这可能需要较长时间，请耐心等待...")
    
    try:
        local_path = download_model_to_local(model_path)
        
        print(f"\n✅ 下载完成!")
        print(f"本地路径: {local_path}")
        
        # 验证下载
        print(f"\n=== 验证下载 ===")
        if os.path.exists(local_path):
            # 计算文件大小
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            
            total_size_gb = total_size / (1024**3)
            print(f"文件数量: {file_count}")
            print(f"总大小: {total_size_gb:.2f} GB")
            
            # 检查关键文件
            key_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            missing_files = []
            for file in key_files:
                if not os.path.exists(os.path.join(local_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"⚠️  缺少关键文件: {missing_files}")
            else:
                print("✅ 所有关键文件都存在")
                
        else:
            print("❌ 下载验证失败: 本地路径不存在")
            
    except KeyboardInterrupt:
        print("\n❌ 下载被用户中断")
        print("您可以稍后重新运行此脚本，下载会从断点继续")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("请检查网络连接和磁盘空间")
        return
    
    print(f"\n=== 下载完成 ===")
    print("现在您可以使用以下命令运行优化模型:")
    print(f"python example_usage.py")


if __name__ == "__main__":
    main() 