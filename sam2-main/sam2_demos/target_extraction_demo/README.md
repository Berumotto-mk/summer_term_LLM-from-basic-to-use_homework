# 目标提取功能演示

## 🎯 功能描述
目标提取功能可以从视频中精确提取指定的目标对象，生成带透明背景的目标图像。

## ✨ 主要特性
- 精确的目标分割和提取
- 透明背景处理
- 多帧一致性保证
- 质量评估和报告生成

## 🔧 使用方法
```python
segmenter = AdvancedVideoSegmenter()
extracted_targets, masks = segmenter.target_extraction(
    frame_paths=frame_list,
    click_point=[x, y],
    output_dir="output",
    obj_name="target_object"
)
```

## 💡 应用场景
- 视频素材提取
- 对象分析研究
- 内容创作辅助
- 动画制作素材
