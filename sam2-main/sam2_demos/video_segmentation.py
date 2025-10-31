#!/usr/bin/env python3
"""
SAM2 视频分割应用
支持视频中的对象追踪和分割
"""

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

class VideoSegmenter:
    def __init__(self, model_path="checkpoints/sam2.1_hiera_tiny.pt"):
        """初始化视频分割器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 初始化SAM2视频分割器 (设备: {self.device})")
        
        # 加载视频预测器
        self.predictor = SAM2VideoPredictor.from_pretrained(
            model_id="facebook/sam2-hiera-tiny",
            device=self.device
        )
        print("✅ 视频模型加载完成")
    
    def extract_frames(self, video_path, output_dir, max_frames=30):
        """从视频提取帧"""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 视频信息: {total_frames} 帧, {fps:.1f} FPS")
        
        # 计算采样间隔
        frame_interval = max(1, total_frames // max_frames)
        extracted_frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_path = os.path.join(output_dir, f"frame_{len(extracted_frames):04d}.jpg")
                Image.fromarray(frame_rgb).save(frame_path)
                extracted_frames.append(frame_path)
                
                if len(extracted_frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"✅ 提取了 {len(extracted_frames)} 帧")
        return extracted_frames, fps
    
    def segment_video_object(self, frame_paths, click_point, output_dir):
        """对视频中的对象进行分割和追踪"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎯 开始视频对象分割...")
        print(f"   点击位置: {click_point}")
        print(f"   帧数: {len(frame_paths)}")
        
        # 初始化推理状态
        inference_state = self.predictor.init_state(video_path=os.path.dirname(frame_paths[0]))
        
        # 在第一帧添加点击点
        ann_frame_idx = 0  # 在第一帧添加注释
        ann_obj_id = 1     # 对象ID
        
        # 读取第一帧
        first_frame = np.array(Image.open(frame_paths[0]))
        
        # 添加点击点
        points = np.array([click_point], dtype=np.float32)
        labels = np.array([1], np.int32)  # 1表示前景点
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        
        print(f"✅ 在第一帧添加了追踪点")
        
        # 传播到所有帧
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        print(f"✅ 完成视频传播，处理了 {len(video_segments)} 帧")
        
        # 可视化结果
        self._visualize_video_results(frame_paths, video_segments, click_point, output_dir)
        
        return video_segments
    
    def _visualize_video_results(self, frame_paths, video_segments, click_point, output_dir):
        """可视化视频分割结果"""
        print("🎨 生成视频分割可视化...")
        
        # 创建结果帧
        result_frames = []
        
        for i, frame_path in enumerate(frame_paths):
            frame = np.array(Image.open(frame_path))
            
            if i in video_segments:
                masks = video_segments[i]
                
                # 创建可视化
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 原始帧
                axes[0].imshow(frame)
                if i == 0:  # 在第一帧显示点击点
                    axes[0].plot(click_point[0], click_point[1], 'ro', markersize=10)
                axes[0].set_title(f"Frame {i}")
                axes[0].axis('off')
                
                # 分割结果
                axes[1].imshow(frame)
                for obj_id, mask in masks.items():
                    axes[1].imshow(mask, alpha=0.5, cmap='jet')
                axes[1].set_title(f"Segmentation")
                axes[1].axis('off')
                
                # 纯mask
                for obj_id, mask in masks.items():
                    axes[2].imshow(mask, cmap='gray')
                axes[2].set_title("Mask")
                axes[2].axis('off')
                
                # 保存帧结果
                frame_output = os.path.join(output_dir, f"result_frame_{i:04d}.png")
                plt.tight_layout()
                plt.savefig(frame_output, dpi=100, bbox_inches='tight')
                plt.close()
                
                result_frames.append(frame_output)
        
        print(f"✅ 生成了 {len(result_frames)} 个结果帧")
        
        # 创建汇总图
        self._create_video_summary(result_frames[:12], output_dir)  # 最多显示12帧
    
    def _create_video_summary(self, frame_paths, output_dir):
        """创建视频分割汇总图"""
        if not frame_paths:
            return
        
        num_frames = len(frame_paths)
        cols = min(4, num_frames)
        rows = (num_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, frame_path in enumerate(frame_paths):
            if i < len(axes):
                img = Image.open(frame_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Frame {i}")
                axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(frame_paths), len(axes)):
            axes[i].axis('off')
        
        summary_path = os.path.join(output_dir, "video_segmentation_summary.png")
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 视频汇总图已保存: {summary_path}")

def demo_video_segmentation():
    """演示视频分割（使用模拟数据）"""
    print("🚀 SAM2 视频分割演示")
    print("=" * 50)
    
    # 检查是否有视频文件
    video_path = "notebooks/videos/bedroom.mp4"
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        print("📝 创建模拟视频分割演示...")
        demo_simulated_video_segmentation()
        return
    
    try:
        # 初始化视频分割器
        segmenter = VideoSegmenter()
        
        # 提取视频帧
        frames_dir = "sam2_demos/video_frames"
        frame_paths, fps = segmenter.extract_frames(video_path, frames_dir, max_frames=20)
        
        # 定义点击点（视频中心）
        first_frame = Image.open(frame_paths[0])
        width, height = first_frame.size
        click_point = [width // 2, height // 2]
        
        # 执行视频分割
        output_dir = "sam2_demos/video_results"
        video_segments = segmenter.segment_video_object(frame_paths, click_point, output_dir)
        
        print("✅ 视频分割演示完成!")
        
    except Exception as e:
        print(f"❌ 视频分割演示失败: {e}")
        print("📝 尝试模拟演示...")
        demo_simulated_video_segmentation()

def demo_simulated_video_segmentation():
    """创建模拟的视频分割演示"""
    print("🎬 创建模拟视频分割演示...")
    
    output_dir = "sam2_demos/simulated_video"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用静态图片模拟视频帧
    image_path = "notebooks/images/cars.jpg"
    if not os.path.exists(image_path):
        print(f"❌ 基础图片不存在: {image_path}")
        return
    
    # 读取基础图片
    base_image = Image.open(image_path)
    width, height = base_image.size
    
    # 创建模拟的"视频帧"（添加不同的变换）
    frames = []
    for i in range(8):
        # 应用轻微的变换模拟运动
        angle = i * 2  # 轻微旋转
        scale = 1.0 + i * 0.01  # 轻微缩放
        
        # 创建变换后的图片
        transformed = base_image.rotate(angle, expand=False)
        # 轻微调整亮度
        enhanced = Image.eval(transformed, lambda x: min(255, int(x * (0.9 + i * 0.02))))
        
        frame_path = os.path.join(output_dir, f"simulated_frame_{i:04d}.jpg")
        enhanced.save(frame_path)
        frames.append(frame_path)
    
    # 创建模拟的分割结果展示
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, frame_path in enumerate(frames):
        row = i // 4
        col = i % 4
        
        frame = Image.open(frame_path)
        axes[row, col].imshow(frame)
        
        # 模拟添加分割效果
        if i == 0:
            # 在第一帧显示点击点
            axes[row, col].plot(width//2, height//2, 'ro', markersize=8)
            axes[row, col].set_title(f"Frame {i} (Click Point)")
        else:
            # 在其他帧显示模拟的分割区域
            circle = plt.Circle((width//2, height//2), 50 + i*10, 
                              color='red', fill=False, linewidth=3, alpha=0.7)
            axes[row, col].add_patch(circle)
            axes[row, col].set_title(f"Frame {i} (Tracked)")
        
        axes[row, col].axis('off')
    
    plt.suptitle("SAM2 Video Segmentation Simulation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, "simulated_video_segmentation.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 模拟视频分割演示已保存: {summary_path}")
    
    # 创建说明文档
    readme_content = """# 模拟视频分割演示

## 📝 说明
由于缺少实际视频文件，此演示展示了SAM2视频分割的工作原理：

## 🎯 视频分割流程
1. **初始标注**: 在第一帧点击目标对象
2. **自动传播**: SAM2自动追踪对象到后续帧
3. **一致性维护**: 保持分割的时序一致性

## 🔧 实际使用
要使用真实视频分割功能：
1. 将视频文件放入 `notebooks/videos/` 目录
2. 运行 `python video_segmentation.py`
3. SAM2将自动提取帧并进行分割

## 💡 应用场景
- 视频对象移除
- 运动分析
- 视频编辑
- 行为追踪
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📋 说明文档已创建: {readme_path}")

if __name__ == "__main__":
    demo_video_segmentation()
