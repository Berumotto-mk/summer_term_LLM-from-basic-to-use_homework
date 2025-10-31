# SAM2 环境配置完成 🎉

恭喜！您的SAM2环境已经成功配置完成。

## 🖥️ 系统配置概览

- **操作系统**: Linux (WSL/Ubuntu)
- **Python版本**: 3.10.18  
- **PyTorch版本**: 2.7.1+cpu
- **运行模式**: CPU版本
- **项目路径**: `/mnt/f/angment/sam2-main`
- **环境名称**: `sam2`

## 📦 已安装组件

### 核心依赖
- ✅ PyTorch 2.7.1 (CPU版本)
- ✅ TorchVision 0.22.1
- ✅ NumPy 2.1.2
- ✅ OpenCV 4.12.0
- ✅ SAM2 1.0 (最新版本)

### 模型权重
- ✅ sam2.1_hiera_tiny.pt (149 MB) - 最快
- ✅ sam2.1_hiera_small.pt (176 MB) - 平衡
- ✅ sam2.1_hiera_base_plus.pt (309 MB) - 高质量
- ✅ sam2.1_hiera_large.pt (857 MB) - 最佳效果

### 开发工具
- ✅ Jupyter Notebook
- ✅ Matplotlib (可视化)
- ✅ 完整的notebooks示例

## 🚀 快速开始

### 1. 激活环境
```bash
# 方法1: 使用我们的快捷脚本
source activate_sam2.sh

# 方法2: 手动激活
conda activate sam2
cd /mnt/f/angment/sam2-main
```

### 2. 验证安装
```bash
python test_installation.py
```

### 3. 运行演示
```bash
python sam2_demo.py
```

### 4. 启动Jupyter
```bash
jupyter notebook notebooks/
```

## 📚 使用示例

### 图像分割
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 加载模型
sam2_model = build_sam2("sam2/configs/sam2.1/sam2.1_hiera_t.yaml", 
                        "checkpoints/sam2.1_hiera_tiny.pt")
predictor = SAM2ImagePredictor(sam2_model)

# 设置图像
predictor.set_image(image_array)

# 点击分割
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True
)
```

### 视频分割  
```python
from sam2.sam2_video_predictor import SAM2VideoPredictor

# 加载视频预测器
predictor = SAM2VideoPredictor.from_pretrained("checkpoints/sam2.1_hiera_tiny.pt")

# 初始化视频状态
inference_state = predictor.init_state(video_path="path/to/video")

# 添加点击点
predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=input_points,
    labels=input_labels
)

# 传播到整个视频
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    # 处理每一帧的结果
    pass
```

## 📁 目录结构

```
/mnt/f/angment/sam2-main/
├── sam2/                    # SAM2核心代码
│   ├── configs/            # 模型配置文件
│   ├── modeling/           # 模型架构
│   └── utils/              # 工具函数
├── checkpoints/            # 预训练模型权重
├── notebooks/              # Jupyter示例
│   ├── images/            # 示例图片
│   └── *.ipynb           # 示例notebook
├── test_installation.py   # 安装验证脚本
├── sam2_demo.py           # 快速演示脚本
└── activate_sam2.sh       # 环境激活脚本
```

## 🔧 模型选择指南

| 模型 | 大小 | 速度 | 精度 | 推荐用途 |
|------|------|------|------|----------|
| tiny | 149MB | 最快 | 良好 | 快速测试、实时应用 |
| small | 176MB | 快 | 好 | 平衡性能和速度 |
| base+ | 309MB | 中等 | 很好 | 高质量分割 |
| large | 857MB | 慢 | 最佳 | 最高精度要求 |

## 📝 注意事项

1. **CPU模式**: 当前配置为CPU版本，适合开发和测试。如需GPU加速，请：
   - 安装NVIDIA驱动和CUDA
   - 重新安装GPU版本的PyTorch
   - 设置环境变量跳过CUDA扩展构建

2. **内存要求**: 
   - tiny模型: 最少4GB内存
   - large模型: 建议16GB+内存

3. **性能优化**:
   - 使用较小的输入图像尺寸
   - 选择合适的模型大小
   - 考虑批处理多个预测

## 🐛 故障排除

### 常见问题

1. **导入错误**
```bash
# 确保环境已激活
conda activate sam2
export PYTHONPATH="/mnt/f/angment/sam2-main:$PYTHONPATH"
```

2. **模型加载失败**
```bash
# 检查模型文件
ls -la checkpoints/
# 重新下载模型权重
cd checkpoints && bash download_ckpts.sh
```

3. **内存不足**
```python
# 使用更小的模型
model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"  # tiny版本
```

## 📖 更多资源

- [SAM2 官方论文](https://arxiv.org/abs/2408.00714)
- [SAM2 GitHub仓库](https://github.com/facebookresearch/sam2)
- [示例代码](./notebooks/)
- [模型配置文档](./sam2/configs/)

## 🎯 下一步

1. 浏览 `notebooks/` 目录中的示例
2. 尝试在您自己的图像上运行分割
3. 探索视频分割功能
4. 根据需要调整模型配置

---

**环境配置完成时间**: $(date)  
**配置者**: GitHub Copilot  
**项目版本**: SAM2 v1.0
