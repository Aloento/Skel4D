#!/usr/bin/env python3
"""
Zero123Plus + StableKeypoints Quick Start Guide

这个脚本提供交互式的快速开始指南
"""

import sys
from pathlib import Path

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                Zero123Plus + StableKeypoints                  ║
║                     Quick Start Guide                         ║
╚══════════════════════════════════════════════════════════════╝
    """)

def print_project_structure():
    print("""
📁 Project Structure:
├── src/
│   ├── models/
│   │   └── zero123plus_stablekeypoints.py  # 核心模型实现
│   ├── data/                               # 数据处理模块
│   └── utils/                              # 工具函数
├── configs/
│   └── zero123plus_stablekeypoints.yaml    # 配置文件
├── train_zero123plus_sk.py                 # 训练脚本
├── inference_zero123plus_sk.py             # 推理脚本
├── test_integration.py                     # 集成测试
├── demo.py                                 # 演示脚本
└── setup.sh                               # 环境设置
    """)

def print_usage_options():
    print("""
🚀 Usage Options:

1. 🧪 Run Demo (No GPU required)
   python demo.py --num_probes 8

2. 🔧 Test Integration (Requires GPU & Zero123Plus access)
   python test_integration.py

3. 🎯 Train on Your Images
   python train_zero123plus_sk.py --image_dir /path/to/images

4. 🔍 Run Inference
   python inference_zero123plus_sk.py --probe_path runs/final_probes.safetensors --image_path test.jpg

5. ⚙️  Quick Setup
   ./setup.sh
    """)

def print_technical_details():
    print("""
🔬 Technical Details:

Core Idea:
• 将StableKeypoints的"可学习文本token"迁移到Zero123Plus的"条件探针"
• 在Zero123Plus条件编码中注入K个可学习探针向量
• 仅优化探针参数，冻结Zero123Plus主干网络

Key Components:
• LearnableProbes: 可学习探针向量 (仅训练这些参数)
• ProbeAttentionProcessor: 收集cross-attention注意力图
• Keypoint Extraction: 使用soft-argmax从注意力图提取2D坐标

Loss Functions:
• Compactness Loss: 促使注意力集中 (熵 + 方差)
• Diversity Loss: 确保不同探针关注不同区域 (余弦相似度)
    """)

def print_requirements():
    print("""
📋 Requirements:

Hardware:
• GPU with ≥8GB VRAM (recommended)
• CPU mode supported but slower

Software:
• Python ≥3.8
• PyTorch ≥2.0
• diffusers ≥0.29.0
• transformers ≥4.30.0

Access:
• Hugging Face account (for Zero123Plus model)
• Internet connection for model download
    """)

def interactive_guide():
    print_banner()
    
    while True:
        print("""
Choose an option:
1. 📁 View Project Structure
2. 🚀 View Usage Options  
3. 🔬 Technical Details
4. 📋 Requirements
5. 🧪 Run Demo Now
6. 🔧 Test Integration Now
7. 🚪 Exit
        """)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            print_project_structure()
        elif choice == '2':
            print_usage_options()
        elif choice == '3':
            print_technical_details()
        elif choice == '4':
            print_requirements()
        elif choice == '5':
            print("\n🧪 Running Demo...")
            import subprocess
            try:
                subprocess.run([sys.executable, "demo.py", "--num_probes", "6"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Demo failed: {e}")
            except FileNotFoundError:
                print("❌ demo.py not found. Make sure you're in the project root directory.")
        elif choice == '6':
            print("\n🔧 Running Integration Test...")
            import subprocess
            try:
                subprocess.run([sys.executable, "test_integration.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Integration test failed: {e}")
            except FileNotFoundError:
                print("❌ test_integration.py not found. Make sure you're in the project root directory.")
        elif choice == '7':
            print("\n👋 Goodbye! Happy researching!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-7.")
        
        input("\nPress Enter to continue...")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print_banner()
        print_project_structure()
        print_usage_options()
        print_technical_details()
        print_requirements()
        return
    
    # 检查是否在正确的目录
    if not Path("src/models/zero123plus_stablekeypoints.py").exists():
        print("❌ Error: Please run this script from the project root directory (Skel4D/)")
        print("   The core model file was not found.")
        return
    
    interactive_guide()

if __name__ == "__main__":
    main()
