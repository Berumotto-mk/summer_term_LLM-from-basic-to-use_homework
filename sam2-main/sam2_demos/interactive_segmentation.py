#!/usr/bin/env python3
"""
SAM2 交互式多点分割应用
支持多个点击点进行精确分割
"""

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class InteractiveSegmenter:
    def __init__(self, model_path="checkpoints/sam2.1_hiera_tiny.pt"):
        """初始化交互式分割器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 初始化SAM2模型 (设备: {self.device})")
        
        # 加载模型
        sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", model_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("✅ 模型加载完成")
        
    def load_image(self, image_path):
        """加载图片"""
        self.image = Image.open(image_path).convert("RGB")
        self.image_array = np.array(self.image)
        self.predictor.set_image(self.image_array)
        print(f"✅ 图片加载完成: {self.image_array.shape}")
        return self.image_array
    
    def segment_with_points(self, points, labels, output_dir="sam2_demos"):
        """
        使用多个点进行分割
        points: 点击点坐标 [[x1,y1], [x2,y2], ...]
        labels: 点标签 [1=前景, 0=背景]
        """
        print(f"🎯 使用 {len(points)} 个点进行分割...")
        
        input_points = np.array(points)
        input_labels = np.array(labels)
        
        # 执行预测
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        print(f"✅ 分割完成! 生成 {len(masks)} 个候选mask")
        print(f"   得分: {[f'{s:.3f}' for s in scores]}")
        
        # 可视化结果
        self._visualize_multipoint_result(input_points, input_labels, masks, scores, output_dir)
        
        return masks, scores
    
    def _visualize_multipoint_result(self, points, labels, masks, scores, output_dir):
        """可视化多点分割结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建详细的可视化
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 原图 + 所有点击点
        plt.subplot(2, 3, 1)
        plt.imshow(self.image_array)
        self._plot_points(points, labels)
        plt.title("Input Image + Click Points", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 2-4. 显示所有候选mask
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.subplot(2, 3, 2 + i)
            plt.imshow(self.image_array)
            plt.imshow(mask, alpha=0.6, cmap='jet')
            self._plot_points(points, labels)
            plt.title(f"Candidate {i+1} (Score: {score:.3f})", fontsize=12)
            plt.axis('off')
        
        # 5. 最佳结果的mask
        best_idx = np.argmax(scores)
        plt.subplot(2, 3, 5)
        plt.imshow(masks[best_idx], cmap='gray')
        plt.title(f"Best Mask (Score: {scores[best_idx]:.3f})", fontsize=12)
        plt.axis('off')
        
        # 6. 轮廓版本
        plt.subplot(2, 3, 6)
        plt.imshow(self.image_array)
        best_mask = masks[best_idx]
        # 绘制轮廓
        contours = self._get_mask_contours(best_mask)
        for contour in contours:
            plt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)
        self._plot_points(points, labels)
        plt.title("Contour Result", fontsize=12)
        plt.axis('off')
        
        # 保存结果
        output_path = os.path.join(output_dir, "multipoint_segmentation_result.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 多点分割结果已保存: {output_path}")
        return output_path
    
    def _plot_points(self, points, labels):
        """绘制点击点"""
        for point, label in zip(points, labels):
            color = 'green' if label == 1 else 'red'
            marker = 'o' if label == 1 else 'x'
            plt.plot(point[0], point[1], color=color, marker=marker, 
                    markersize=10, markeredgecolor='white', linewidth=2)
    
    def _get_mask_contours(self, mask):
        """获取mask轮廓"""
        import cv2
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [contour.squeeze() for contour in contours if len(contour) > 3]

def demo_multipoint_segmentation():
    """演示多点分割"""
    print("🚀 SAM2 多点交互式分割演示")
    print("=" * 50)
    
    # 初始化分割器
    segmenter = InteractiveSegmenter()
    
    # 加载图片
    image_path = "notebooks/images/groceries.jpg"
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return
    
    image_array = segmenter.load_image(image_path)
    height, width = image_array.shape[:2]
    
    # 定义多个测试场景
    scenarios = [
        {
            "name": "精确分割 - 前景+背景点",
            "points": [[width//3, height//3], [width*2//3, height*2//3], [width//6, height//6]],
            "labels": [1, 1, 0],  # 前两个是前景点，最后一个是背景点
            "description": "使用前景和背景点进行精确分割"
        },
        {
            "name": "多前景点分割",
            "points": [[width//4, height//2], [width//2, height//2], [width*3//4, height//2]],
            "labels": [1, 1, 1],  # 全部是前景点
            "description": "使用多个前景点确保完整分割"
        },
        {
            "name": "区域排除分割",
            "points": [[width//2, height//2], [width//8, height//8], [width*7//8, height*7//8]],
            "labels": [1, 0, 0],  # 一个前景点，两个背景点
            "description": "通过背景点排除不需要的区域"
        }
    ]
    
    # 执行所有场景
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 场景 {i}: {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   点位置: {scenario['points']}")
        print(f"   点标签: {scenario['labels']}")
        
        masks, scores = segmenter.segment_with_points(
            scenario['points'], 
            scenario['labels'],
            f"sam2_demos/scenario_{i}"
        )
        
        print(f"   最佳得分: {scores.max():.3f}")

if __name__ == "__main__":
    demo_multipoint_segmentation()
