#!/bin/bash

# SAM2 环境激活脚本
# 使用方法: source activate_sam2.sh

echo "🚀 激活SAM2环境..."

# 激活conda环境
source /home/tswcbyy20050413/anaconda3/etc/profile.d/conda.sh
conda activate sam2

# 设置环境变量
export SAM2_PROJECT_ROOT="/mnt/f/angment/sam2-main"
export PYTHONPATH="${SAM2_PROJECT_ROOT}:${PYTHONPATH}"

# 切换到项目目录
cd "${SAM2_PROJECT_ROOT}"

echo "✅ SAM2环境已激活!"
echo "📍 当前目录: $(pwd)"
echo "🐍 Python版本: $(python --version)"
echo "📦 PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"

echo ""
echo "📋 快速开始:"
echo "  测试安装:     python test_installation.py"
echo "  运行演示:     python sam2_demo.py"
echo "  启动Jupyter:  jupyter notebook notebooks/"
echo "  查看示例:     ls notebooks/"
echo ""
echo "🔧 模型配置:"
echo "  tiny:   sam2.1_hiera_tiny.pt     (最快, 149MB)"
echo "  small:  sam2.1_hiera_small.pt    (平衡, 176MB)"  
echo "  base+:  sam2.1_hiera_base_plus.pt (高质量, 309MB)"
echo "  large:  sam2.1_hiera_large.pt    (最佳, 857MB)"
echo ""
