#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
SAM医学图像分割演示 - 模拟医学影像中的器官分割
应用场景：医学影像分析、器官检测、病灶分割
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from common_tools import show_mask, show_box, show_points, show_anns
import json
import os

class MedicalSegmentationDemo:
    def __init__(self, model_path="models/sam_vit_h_4b8939.pth", device="cpu"):
        self.sam_checkpoint = model_path
        self.model_type = "vit_h"
        self.device = device
        self.sam = None
        self.predictor = None
        self.mask_generator = None
        self.load_model()
    
    def load_model(self):
        """加载SAM模型"""
        print("正在加载医学分割模型...")
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        # 配置医学图像专用的掩码生成器
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=16,  # 医学图像通常需要更精细的分割
            pred_iou_thresh=0.90,  # 提高阈值以获得更高质量的掩码
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200,  # 过滤掉过小的区域
        )
        print("医学分割模型加载完成！")
    
    def segment_organ_with_points(self, image_path, organ_points, background_points=None):
        """
        使用点提示进行器官分割
        organ_points: 器官区域的点坐标列表
        background_points: 背景区域的点坐标列表
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"开始分割器官 - 图像尺寸: {image.shape}")
        self.predictor.set_image(image)
        
        # 组合前景和背景点
        all_points = []
        all_labels = []
        
        for point in organ_points:
            all_points.append(point)
            all_labels.append(1)  # 前景点
            
        if background_points:
            for point in background_points:
                all_points.append(point)
                all_labels.append(0)  # 背景点
        
        input_points = np.array(all_points)
        input_labels = np.array(all_labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        return image, masks, scores, input_points, input_labels
    
    def segment_with_bounding_box(self, image_path, bbox):
        """使用边界框进行器官分割"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image)
        
        input_box = np.array(bbox)  # [x1, y1, x2, y2]
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        return image, masks, scores, input_box
    
    def automatic_organ_detection(self, image_path):
        """自动检测图像中的所有潜在器官区域"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print("正在进行自动器官检测...")
        masks = self.mask_generator.generate(image)
        
        # 根据面积和形状筛选可能的器官区域
        organ_candidates = []
        for mask_data in masks:
            area = mask_data['area']
            stability_score = mask_data['stability_score']
            
            # 简单的器官区域筛选规则
            if area > 1000 and stability_score > 0.95:
                organ_candidates.append(mask_data)
        
        print(f"检测到 {len(organ_candidates)} 个可能的器官区域")
        return image, organ_candidates
    
    def calculate_organ_metrics(self, mask):
        """计算器官的基本几何指标"""
        if len(mask.shape) > 2:
            mask = mask[0] if mask.shape[0] == 1 else mask
        
        # 计算面积
        area = np.sum(mask)
        
        # 计算周长（边界像素数）
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0
        
        # 计算圆形度 (4π*面积/周长²)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 计算边界框
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            bbox = [np.min(x_indices), np.min(y_indices), 
                   np.max(x_indices), np.max(y_indices)]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = width / height if height > 0 else 0
        else:
            bbox = [0, 0, 0, 0]
            aspect_ratio = 0
        
        return {
            'area': int(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'bbox': bbox
        }
    
    def save_segmentation_results(self, image, masks, scores, output_dir, prefix="medical"):
        """保存分割结果和指标"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # 保存掩码图像
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            plt.title(f"Medical Segmentation - Mask {i+1}, Score: {score:.3f}")
            plt.axis('off')
            
            mask_filename = f"{prefix}_mask_{i+1}_score_{score:.3f}.png"
            plt.savefig(os.path.join(output_dir, mask_filename), dpi=150, bbox_inches='tight')
            plt.close()
            
            # 计算并保存指标
            metrics = self.calculate_organ_metrics(mask)
            metrics['score'] = float(score)
            metrics['mask_file'] = mask_filename
            results.append(metrics)
        
        # 保存JSON报告
        report_file = os.path.join(output_dir, f"{prefix}_analysis_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"分割结果已保存到: {output_dir}")
        return results

def demo_heart_segmentation():
    """演示心脏分割"""
    demo = MedicalSegmentationDemo()
    
    # 模拟心脏区域的点（实际使用时需要根据真实医学图像调整）
    heart_points = [[300, 250], [320, 270], [280, 280]]
    background_points = [[100, 100], [500, 100], [100, 400]]
    
    print("\n=== 心脏分割演示 ===")
    image, masks, scores, points, labels = demo.segment_organ_with_points(
        'images/truck.jpg',  # 演示用，实际应使用医学图像
        heart_points, 
        background_points
    )
    
    # 可视化结果
    best_mask_idx = np.argmax(scores)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    show_points(points, labels, plt.gca())
    plt.title("输入点标注")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    show_mask(masks[best_mask_idx], plt.gca())
    show_points(points, labels, plt.gca())
    plt.title(f"最佳分割结果 (Score: {scores[best_mask_idx]:.3f})")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(masks[best_mask_idx], cmap='gray')
    plt.title("分割掩码")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('heart_segmentation_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 计算心脏指标
    metrics = demo.calculate_organ_metrics(masks[best_mask_idx])
    print(f"心脏分割指标:")
    print(f"  面积: {metrics['area']} 像素")
    print(f"  周长: {metrics['perimeter']:.2f} 像素")
    print(f"  圆形度: {metrics['circularity']:.3f}")
    print(f"  长宽比: {metrics['aspect_ratio']:.3f}")

def demo_lung_detection():
    """演示肺部检测"""
    demo = MedicalSegmentationDemo()
    
    print("\n=== 肺部自动检测演示 ===")
    image, organ_candidates = demo.automatic_organ_detection('images/dog.jpg')
    
    # 可视化检测结果
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    
    # 显示检测到的器官候选区域
    for i, candidate in enumerate(organ_candidates[:5]):  # 只显示前5个
        mask = candidate['segmentation']
        show_mask(mask, plt.gca(), random_color=True)
        
        # 在区域中心添加标签
        y_center = int(candidate['bbox'][1] + candidate['bbox'][3] / 2)
        x_center = int(candidate['bbox'][0] + candidate['bbox'][2] / 2)
        plt.text(x_center, y_center, f"R{i+1}", fontsize=12, color='white', 
                weight='bold', ha='center', va='center')
    
    plt.title(f"自动检测到 {len(organ_candidates)} 个潜在器官区域")
    plt.axis('off')
    plt.savefig('lung_detection_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 输出检测统计
    print(f"检测统计:")
    for i, candidate in enumerate(organ_candidates[:5]):
        print(f"  区域 {i+1}: 面积={candidate['area']}, 稳定性={candidate['stability_score']:.3f}")

if __name__ == '__main__':
    print("=== SAM医学图像分割演示 ===")
    print("注意：此演示使用普通图像模拟医学图像分割，实际应用需要真实的医学影像数据")
    
    try:
        # 演示心脏分割
        demo_heart_segmentation()
        
        # 演示肺部检测  
        demo_lung_detection()
        
        print("\n✅ 医学分割演示完成！")
        print("💡 在实际医学应用中，建议:")
        print("   1. 使用专门的医学图像数据集训练模型")
        print("   2. 结合医学专业知识调整分割参数")
        print("   3. 添加更多的形态学后处理步骤")
        print("   4. 集成医学影像标准(如DICOM)支持")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确保模型文件和图像文件存在")
