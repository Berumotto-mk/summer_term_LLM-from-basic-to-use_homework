# 目标跟踪功能演示

## 🔍 功能描述
目标跟踪功能支持同时跟踪多个目标对象，分析其运动轨迹、速度变化等运动特征。

## ✨ 主要特性
- 多目标同时跟踪
- 实时轨迹分析
- 运动特征统计
- 遮挡恢复能力
- 自动质量评估

## 🔧 使用方法
```python
segmenter = AdvancedVideoSegmenter()
tracked_frames, results, motion_analysis = segmenter.object_tracking(
    frame_paths=frame_list,
    click_points=[(x1, y1), (x2, y2), ...],  # 多个目标点
    output_dir="output",
    track_names=["target1", "target2", ...]
)
```

## 📊 分析指标
- **轨迹完整性**: 跟踪成功的帧数比例
- **运动速度**: 平均速度和最大速度
- **运动距离**: 总移动距离
- **置信度**: 跟踪质量评分
- **目标稳定性**: 大小变化率

## 💡 应用场景
- 运动分析研究
- 安防监控系统
- 体育比赛分析
- 交通流量监测
- 行为模式研究
