# 背景替换功能演示

## 🎨 功能描述
背景替换功能可以将视频中的背景替换为任意指定的新背景，保持前景对象不变。

## ✨ 主要特性
- 精确的前景/背景分离
- 支持任意背景图像
- 边缘优化处理
- 多帧时序一致性

## 🔧 使用方法
```python
segmenter = AdvancedVideoSegmenter()
replaced_frames, masks = segmenter.background_replacement(
    frame_paths=frame_list,
    click_point=[x, y],  # 点击前景对象
    new_background_path="new_bg.jpg",
    output_dir="output"
)
```

## 💡 应用场景
- 虚拟背景会议
- 影视后期制作
- 直播背景替换
- 创意视频制作
- 产品展示视频
