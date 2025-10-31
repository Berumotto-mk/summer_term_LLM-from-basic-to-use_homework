# SAM2 应用套件运行报告

## 📋 基本信息
- **生成时间**: 2025-07-31 09:02:22
- **应用总数**: 5

## 🚀 应用列表

### 基础分割演示 (basic)
- **状态**: ✅ 可用
- **脚本**: `simple_demo.py`
- **描述**: 单点点击分割，生成多种可视化

### 交互式分割 (interactive)
- **状态**: ✅ 可用
- **脚本**: `interactive_segmentation.py`
- **描述**: 多点交互式分割，支持前景/背景点

### 批量处理 (batch)
- **状态**: ✅ 可用
- **脚本**: `batch_segmentation.py`
- **描述**: 批量处理多个图像，多种策略

### 视频分割 (video)
- **状态**: ✅ 可用
- **脚本**: `video_segmentation.py`
- **描述**: 视频对象追踪和分割

### 自动掩码生成 (auto)
- **状态**: ✅ 可用
- **脚本**: `auto_mask_generation.py`
- **描述**: 全图自动分割，无需用户输入

## 💡 使用指南

### 单独运行应用
```bash
python sam2_suite.py <应用名称>
```

### 运行所有应用
```bash
python sam2_suite.py --all
```

### 查看应用列表
```bash
python sam2_suite.py --list
```

## 📂 文件结构
```
sam2_demos/
├── simple_demo.py           # 基础分割演示
├── interactive_segmentation.py  # 交互式分割
├── batch_segmentation.py    # 批量处理
├── video_segmentation.py    # 视频分割
├── auto_mask_generation.py  # 自动掩码生成
└── sam2_suite.py           # 应用集成中心
```

## 🎯 应用场景
- **医疗影像**: 器官和病变分割
- **自动驾驶**: 道路场景理解
- **工业检测**: 缺陷检测和质量控制
- **内容创作**: 视频编辑和对象移除
- **科研分析**: 显微镜图像分析
