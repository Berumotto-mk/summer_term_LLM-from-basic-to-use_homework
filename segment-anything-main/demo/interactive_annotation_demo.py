#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
SAM交互式标注工具 - 用于快速创建训练数据集
应用场景：数据标注、半自动标注、标注质量控制、数据集创建
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from common_tools import show_mask, show_points, show_box
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class InteractiveAnnotationTool:
    def __init__(self, model_path="models/sam_vit_h_4b8939.pth", device="cpu"):
        self.sam_checkpoint = model_path
        self.model_type = "vit_h"
        self.device = device
        self.sam = None
        self.predictor = None
        
        # 标注状态
        self.current_image = None
        self.current_image_path = None
        self.current_masks = []
        self.annotation_points = []
        self.annotation_labels = []
        self.annotation_history = []
        self.current_class = "object"
        
        # 类别定义
        self.classes = ["background", "object", "person", "vehicle", "animal", "building"]
        self.class_colors = {
            "background": [128, 128, 128],
            "object": [255, 0, 0],
            "person": [0, 255, 0], 
            "vehicle": [0, 0, 255],
            "animal": [255, 255, 0],
            "building": [255, 0, 255]
        }
        
        self.load_model()
    
    def load_model(self):
        """加载SAM模型"""
        print("正在加载交互式标注模型...")
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        print("交互式标注模型加载完成！")
    
    def set_image(self, image_path):
        """设置当前标注图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        self.current_image_path = image_path
        self.current_image = cv2.imread(image_path)
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        print(f"设置图像: {image_path}, 尺寸: {self.current_image.shape}")
        self.predictor.set_image(self.current_image)
        
        # 重置标注状态
        self.current_masks = []
        self.annotation_points = []
        self.annotation_labels = []
        self.annotation_history = []
    
    def add_positive_point(self, x, y):
        """添加正向点（前景点）"""
        self.annotation_points.append([x, y])
        self.annotation_labels.append(1)
        return self.update_mask()
    
    def add_negative_point(self, x, y):
        """添加负向点（背景点）"""
        self.annotation_points.append([x, y])
        self.annotation_labels.append(0)
        return self.update_mask()
    
    def add_bounding_box(self, x1, y1, x2, y2):
        """添加边界框"""
        bbox = np.array([x1, y1, x2, y2])
        
        point_coords = np.array(self.annotation_points) if self.annotation_points else None
        point_labels = np.array(self.annotation_labels) if self.annotation_labels else None
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=bbox[None, :],
            multimask_output=True,
        )
        
        # 选择最佳掩码
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        return best_mask, best_score
    
    def update_mask(self):
        """更新当前掩码"""
        if not self.annotation_points:
            return None, 0
        
        point_coords = np.array(self.annotation_points)
        point_labels = np.array(self.annotation_labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        
        # 选择最佳掩码
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        return best_mask, best_score
    
    def undo_last_point(self):
        """撤销最后一个点"""
        if self.annotation_points:
            self.annotation_points.pop()
            self.annotation_labels.pop()
            return self.update_mask()
        return None, 0
    
    def clear_all_points(self):
        """清除所有点"""
        self.annotation_points = []
        self.annotation_labels = []
        return None, 0
    
    def save_annotation(self, mask, class_name, output_dir):
        """保存当前标注"""
        if self.current_image is None or mask is None:
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成唯一的标注ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotation_id = f"{os.path.basename(self.current_image_path).split('.')[0]}_{timestamp}"
        
        # 保存掩码
        mask_file = os.path.join(output_dir, f"{annotation_id}_mask.png")
        cv2.imwrite(mask_file, (mask * 255).astype(np.uint8))
        
        # 创建标注记录
        annotation_record = {
            'annotation_id': annotation_id,
            'image_path': self.current_image_path,
            'class_name': class_name,
            'mask_file': mask_file,
            'points': self.annotation_points.copy(),
            'labels': self.annotation_labels.copy(),
            'timestamp': timestamp,
            'image_size': self.current_image.shape[:2],
            'mask_area': int(np.sum(mask))
        }
        
        # 保存标注记录
        record_file = os.path.join(output_dir, f"{annotation_id}_annotation.json")
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(annotation_record, f, indent=2, ensure_ascii=False)
        
        # 添加到历史记录
        self.annotation_history.append(annotation_record)
        
        print(f"标注已保存: {annotation_id}")
        return True
    
    def load_annotation_project(self, project_dir):
        """加载标注项目"""
        if not os.path.exists(project_dir):
            return []
        
        annotations = []
        for file in os.listdir(project_dir):
            if file.endswith('_annotation.json'):
                try:
                    with open(os.path.join(project_dir, file), 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                        annotations.append(annotation)
                except Exception as e:
                    print(f"加载标注文件失败 {file}: {e}")
        
        return sorted(annotations, key=lambda x: x['timestamp'])
    
    def create_annotation_statistics(self, annotations):
        """创建标注统计"""
        if not annotations:
            return {}
        
        stats = {
            'total_annotations': len(annotations),
            'class_distribution': {},
            'total_mask_area': 0,
            'average_mask_area': 0,
            'images_annotated': len(set(ann['image_path'] for ann in annotations))
        }
        
        # 类别分布统计
        for annotation in annotations:
            class_name = annotation['class_name']
            mask_area = annotation['mask_area']
            
            if class_name not in stats['class_distribution']:
                stats['class_distribution'][class_name] = {
                    'count': 0,
                    'total_area': 0,
                    'average_area': 0
                }
            
            stats['class_distribution'][class_name]['count'] += 1
            stats['class_distribution'][class_name]['total_area'] += mask_area
            stats['total_mask_area'] += mask_area
        
        # 计算平均值
        stats['average_mask_area'] = stats['total_mask_area'] / stats['total_annotations']
        
        for class_name in stats['class_distribution']:
            class_stats = stats['class_distribution'][class_name]
            class_stats['average_area'] = class_stats['total_area'] / class_stats['count']
        
        return stats
    
    def visualize_annotation_progress(self, annotations, output_dir):
        """可视化标注进度"""
        if not annotations:
            print("没有标注数据可视化")
            return
        
        stats = self.create_annotation_statistics(annotations)
        
        # 创建进度图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 类别分布
        class_names = list(stats['class_distribution'].keys())
        class_counts = [stats['class_distribution'][name]['count'] for name in class_names]
        
        ax1.pie(class_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
        ax1.set_title('类别分布')
        
        # 面积分布
        class_areas = [stats['class_distribution'][name]['total_area'] for name in class_names]
        ax2.bar(class_names, class_areas, color=['red', 'green', 'blue', 'orange', 'purple'][:len(class_names)])
        ax2.set_title('类别面积分布')
        ax2.set_ylabel('总面积 (像素)')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # 标注时间线
        timestamps = [ann['timestamp'] for ann in annotations]
        cumulative_counts = list(range(1, len(annotations) + 1))
        ax3.plot(cumulative_counts, marker='o')
        ax3.set_title('标注进度时间线')
        ax3.set_xlabel('标注顺序')
        ax3.set_ylabel('累计标注数量')
        ax3.grid(True, alpha=0.3)
        
        # 平均掩码大小
        mask_sizes = [ann['mask_area'] for ann in annotations]
        ax4.hist(mask_sizes, bins=20, alpha=0.7, color='skyblue')
        ax4.axvline(stats['average_mask_area'], color='red', linestyle='--', 
                   label=f'平均值: {stats["average_mask_area"]:.0f}')
        ax4.set_title('掩码面积分布')
        ax4.set_xlabel('面积 (像素)')
        ax4.set_ylabel('频次')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'annotation_statistics.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # 输出统计摘要
        print("\n📊 标注统计摘要:")
        print(f"  总标注数: {stats['total_annotations']}")
        print(f"  标注图像数: {stats['images_annotated']}")
        print(f"  平均掩码面积: {stats['average_mask_area']:.0f} 像素")
        print(f"  类别分布:")
        for class_name, class_stats in stats['class_distribution'].items():
            print(f"    {class_name}: {class_stats['count']} 个 (平均面积: {class_stats['average_area']:.0f})")

def demo_interactive_annotation():
    """演示交互式标注功能"""
    tool = InteractiveAnnotationTool()
    
    print("\n=== 交互式标注工具演示 ===")
    
    # 设置图像
    image_path = 'images/truck.jpg'
    if not os.path.exists(image_path):
        print(f"演示图像 {image_path} 不存在")
        return
    
    tool.set_image(image_path)
    
    # 模拟标注过程
    print("模拟交互式标注过程...")
    
    # 添加正向点
    mask1, score1 = tool.add_positive_point(400, 300)
    print(f"添加正向点 (400, 300), 得分: {score1:.3f}")
    
    # 添加负向点
    mask2, score2 = tool.add_negative_point(100, 100)
    print(f"添加负向点 (100, 100), 得分: {score2:.3f}")
    
    # 再添加一个正向点
    mask3, score3 = tool.add_positive_point(450, 350)
    print(f"添加正向点 (450, 350), 得分: {score3:.3f}")
    
    # 可视化标注过程
    if mask3 is not None:
        plt.figure(figsize=(15, 5))
        
        # 原图
        plt.subplot(1, 3, 1)
        plt.imshow(tool.current_image)
        plt.title("原始图像")
        plt.axis('off')
        
        # 标注点
        plt.subplot(1, 3, 2)
        plt.imshow(tool.current_image)
        points = np.array(tool.annotation_points)
        labels = np.array(tool.annotation_labels)
        show_points(points, labels, plt.gca())
        plt.title("标注点")
        plt.axis('off')
        
        # 分割结果
        plt.subplot(1, 3, 3)
        plt.imshow(tool.current_image)
        show_mask(mask3, plt.gca())
        show_points(points, labels, plt.gca())
        plt.title(f"分割结果 (得分: {score3:.3f})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('interactive_annotation_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 保存标注
    output_dir = "annotation_project_demo"
    success = tool.save_annotation(mask3, "vehicle", output_dir)
    if success:
        print(f"标注已保存到: {output_dir}")
    
    # 测试撤销功能
    print("\n测试撤销功能...")
    mask_undo, score_undo = tool.undo_last_point()
    print(f"撤销后得分: {score_undo:.3f}")
    
    # 清除所有点
    tool.clear_all_points()
    print("已清除所有标注点")

def demo_annotation_quality_control():
    """演示标注质量控制"""
    tool = InteractiveAnnotationTool()
    
    print("\n=== 标注质量控制演示 ===")
    
    # 模拟多个标注记录
    mock_annotations = [
        {
            'annotation_id': 'truck_001',
            'image_path': 'images/truck.jpg',
            'class_name': 'vehicle',
            'mask_area': 15000,
            'timestamp': '20240731_120000'
        },
        {
            'annotation_id': 'truck_002', 
            'image_path': 'images/truck.jpg',
            'class_name': 'vehicle',
            'mask_area': 18000,
            'timestamp': '20240731_120100'
        },
        {
            'annotation_id': 'dog_001',
            'image_path': 'images/dog.jpg',
            'class_name': 'animal',
            'mask_area': 12000,
            'timestamp': '20240731_120200'
        },
        {
            'annotation_id': 'person_001',
            'image_path': 'images/person.jpg',
            'class_name': 'person',
            'mask_area': 8000,
            'timestamp': '20240731_120300'
        }
    ]
    
    # 创建统计分析
    stats = tool.create_annotation_statistics(mock_annotations)
    
    print("📊 质量控制统计:")
    print(f"  总标注数: {stats['total_annotations']}")
    print(f"  标注图像数: {stats['images_annotated']}")
    print(f"  平均掩码面积: {stats['average_mask_area']:.0f} 像素")
    
    # 质量检查
    quality_issues = []
    
    # 检查异常大小的掩码
    for annotation in mock_annotations:
        area = annotation['mask_area']
        if area < 1000:
            quality_issues.append(f"掩码过小: {annotation['annotation_id']} ({area} 像素)")
        elif area > 50000:
            quality_issues.append(f"掩码过大: {annotation['annotation_id']} ({area} 像素)")
    
    # 检查类别不平衡
    class_counts = stats['class_distribution']
    total_annotations = stats['total_annotations']
    
    for class_name, class_stats in class_counts.items():
        percentage = (class_stats['count'] / total_annotations) * 100
        if percentage < 10:
            quality_issues.append(f"类别样本不足: {class_name} (仅 {percentage:.1f}%)")
        elif percentage > 70:
            quality_issues.append(f"类别样本过多: {class_name} ({percentage:.1f}%)")
    
    print(f"\n⚠️  质量问题检查:")
    if quality_issues:
        for issue in quality_issues:
            print(f"  - {issue}")
    else:
        print("  未发现明显质量问题")
    
    # 可视化质量分析
    tool.visualize_annotation_progress(mock_annotations, "annotation_quality_demo")

def demo_batch_annotation_workflow():
    """演示批量标注工作流"""
    print("\n=== 批量标注工作流演示 ===")
    
    # 模拟批量标注场景
    image_list = [
        'images/truck.jpg',
        'images/dog.jpg', 
        'images/groceries.jpg'
    ]
    
    # 检查可用图像
    available_images = [img for img in image_list if os.path.exists(img)]
    
    if not available_images:
        print("没有可用的演示图像")
        return
    
    tool = InteractiveAnnotationTool()
    annotation_workflow = []
    
    for i, image_path in enumerate(available_images):
        print(f"\n处理图像 {i+1}/{len(available_images)}: {image_path}")
        
        try:
            tool.set_image(image_path)
            
            # 模拟自动标注建议（简化版）
            h, w = tool.current_image.shape[:2]
            suggested_points = [
                [w//2, h//2],  # 中心点
                [w//4, h//4],  # 左上区域
                [3*w//4, 3*h//4]  # 右下区域
            ]
            
            workflow_step = {
                'image_path': image_path,
                'image_size': (h, w),
                'suggested_points': suggested_points,
                'annotation_status': 'pending'
            }
            
            # 对每个建议点进行分割
            for j, point in enumerate(suggested_points):
                mask, score = tool.add_positive_point(point[0], point[1])
                if score > 0.8:  # 高质量分割
                    workflow_step[f'mask_{j}'] = {
                        'point': point,
                        'score': score,
                        'area': int(np.sum(mask)) if mask is not None else 0,
                        'quality': 'high' if score > 0.9 else 'medium'
                    }
            
            tool.clear_all_points()  # 清理准备下一个图像
            workflow_step['annotation_status'] = 'completed'
            annotation_workflow.append(workflow_step)
            
        except Exception as e:
            print(f"处理图像时出错: {e}")
            workflow_step['annotation_status'] = 'failed'
            workflow_step['error'] = str(e)
            annotation_workflow.append(workflow_step)
    
    # 输出工作流总结
    print(f"\n📋 批量标注工作流总结:")
    completed = sum(1 for step in annotation_workflow if step['annotation_status'] == 'completed')
    print(f"  处理图像总数: {len(annotation_workflow)}")
    print(f"  成功处理: {completed}")
    print(f"  处理失败: {len(annotation_workflow) - completed}")
    
    # 保存工作流记录
    workflow_file = "batch_annotation_workflow.json"
    with open(workflow_file, 'w', encoding='utf-8') as f:
        json.dump(annotation_workflow, f, indent=2, ensure_ascii=False)
    print(f"  工作流记录已保存: {workflow_file}")

if __name__ == '__main__':
    print("=== SAM交互式标注工具演示 ===")
    
    try:
        # 演示交互式标注
        demo_interactive_annotation()
        
        # 演示质量控制
        demo_annotation_quality_control()
        
        # 演示批量工作流
        demo_batch_annotation_workflow()
        
        print("\n✅ 交互式标注工具演示完成！")
        print("💡 在实际标注项目中，建议:")
        print("   1. 建立标注规范和质量标准")
        print("   2. 定期进行标注质量检查")
        print("   3. 使用多人标注验证重要样本")
        print("   4. 建立标注进度跟踪系统")
        print("   5. 导出为常见的数据集格式(COCO, YOLO等)")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确保模型文件和图像文件存在")
