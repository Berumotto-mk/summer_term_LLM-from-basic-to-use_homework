#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
SAM农业应用演示 - 农作物和病虫害检测分割
应用场景：精准农业、作物监测、病虫害识别、农业自动化
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from common_tools import show_mask, show_box, show_points, show_anns
import json
import os
from collections import defaultdict

class AgriculturalSegmentationSystem:
    def __init__(self, model_path="models/sam_vit_h_4b8939.pth", device="cpu"):
        self.sam_checkpoint = model_path
        self.model_type = "vit_h"
        self.device = device
        self.sam = None
        self.predictor = None
        self.mask_generator = None
        self.load_model()
        
        # 作物类型定义（基于颜色和形状特征的简单分类）
        self.crop_types = {
            'wheat': {'color_range': ([20, 50, 50], [30, 255, 255]), 'min_area': 500},
            'corn': {'color_range': ([15, 50, 50], [25, 255, 255]), 'min_area': 800},
            'tomato': {'color_range': ([0, 50, 50], [10, 255, 255]), 'min_area': 300},
            'lettuce': {'color_range': ([35, 50, 50], [85, 255, 255]), 'min_area': 200}
        }
    
    def load_model(self):
        """加载SAM模型"""
        print("正在加载农业分割模型...")
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        # 农业专用掩码生成器配置
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=20,  # 农田需要密集采样
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=150,  # 适合小作物
        )
        print("农业分割模型加载完成！")
    
    def detect_crops_in_field(self, image_path):
        """检测农田中的作物"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print("正在检测农田中的作物...")
        masks = self.mask_generator.generate(image_rgb)
        
        # 过滤和分类作物
        crops = []
        for mask_data in masks:
            area = mask_data['area']
            if area > 100:  # 过滤小区域
                crop_info = self.classify_crop(image_rgb, mask_data)
                if crop_info:
                    crops.append(crop_info)
        
        print(f"检测到 {len(crops)} 个作物区域")
        return image_rgb, crops
    
    def classify_crop(self, image, mask_data):
        """简单的作物分类（基于颜色特征）"""
        mask = mask_data['segmentation']
        
        # 提取掩码区域的平均颜色
        masked_image = image.copy()
        masked_image[~mask] = 0
        
        # 转换为HSV色彩空间
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
        
        # 计算平均色调
        mask_pixels = hsv_image[mask]
        if len(mask_pixels) == 0:
            return None
            
        avg_hue = np.mean(mask_pixels[:, 0])
        avg_saturation = np.mean(mask_pixels[:, 1])
        avg_value = np.mean(mask_pixels[:, 2])
        
        # 简单分类规则
        crop_type = "unknown"
        if 20 <= avg_hue <= 30 and avg_saturation > 100:  # 黄绿色
            crop_type = "wheat"
        elif 35 <= avg_hue <= 85 and avg_saturation > 80:  # 绿色
            crop_type = "lettuce" if mask_data['area'] < 1000 else "corn"
        elif avg_hue <= 15 or avg_hue >= 165:  # 红色系
            crop_type = "tomato"
        
        return {
            'type': crop_type,
            'area': mask_data['area'],
            'bbox': mask_data['bbox'],
            'mask': mask,
            'avg_color_hsv': [avg_hue, avg_saturation, avg_value],
            'stability_score': mask_data['stability_score']
        }
    
    def detect_plant_diseases(self, image_path, disease_points):
        """检测植物病害"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image_rgb)
        
        disease_regions = []
        
        for i, point in enumerate(disease_points):
            print(f"分析可疑病害区域 {i+1}/{len(disease_points)}")
            
            input_point = np.array([point])
            input_label = np.array([1])
            
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # 分析病害特征
            disease_analysis = self.analyze_disease_features(image_rgb, best_mask)
            disease_analysis.update({
                'region_id': i + 1,
                'point': point,
                'mask': best_mask,
                'score': float(scores[best_mask_idx])
            })
            
            disease_regions.append(disease_analysis)
        
        return image_rgb, disease_regions
    
    def analyze_disease_features(self, image, mask):
        """分析病害特征"""
        # 提取病害区域
        masked_region = image[mask]
        
        if len(masked_region) == 0:
            return {'severity': 'unknown', 'type': 'unknown'}
        
        # 颜色分析
        avg_color = np.mean(masked_region, axis=0)
        
        # 纹理分析（简化版）
        gray_region = cv2.cvtColor(masked_region.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY)
        texture_variance = np.var(gray_region)
        
        # 简单的病害分类
        disease_type = "unknown"
        severity = "mild"
        
        # 基于颜色的病害分类
        if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:  # 红色主导
            disease_type = "rust"
            severity = "moderate" if avg_color[0] > 150 else "mild"
        elif avg_color[1] < 100:  # 黄化
            disease_type = "yellowing"
            severity = "severe" if avg_color[1] < 50 else "moderate"
        elif np.all(avg_color < 80):  # 黑斑
            disease_type = "blight"
            severity = "severe"
        
        return {
            'type': disease_type,
            'severity': severity,
            'avg_color': avg_color.tolist(),
            'texture_variance': float(texture_variance),
            'area': int(np.sum(mask))
        }
    
    def calculate_field_statistics(self, crops):
        """计算农田统计信息"""
        stats = defaultdict(list)
        
        for crop in crops:
            crop_type = crop['type']
            stats[crop_type].append(crop['area'])
        
        field_stats = {}
        total_area = 0
        
        for crop_type, areas in stats.items():
            count = len(areas)
            total_crop_area = sum(areas)
            avg_area = total_crop_area / count if count > 0 else 0
            total_area += total_crop_area
            
            field_stats[crop_type] = {
                'count': count,
                'total_area': total_crop_area,
                'average_area': avg_area,
                'percentage': 0  # 稍后计算
            }
        
        # 计算百分比
        for crop_type in field_stats:
            field_stats[crop_type]['percentage'] = (
                field_stats[crop_type]['total_area'] / total_area * 100 
                if total_area > 0 else 0
            )
        
        return field_stats
    
    def generate_field_report(self, image_path, crops, diseases, output_dir):
        """生成农田分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算统计信息
        field_stats = self.calculate_field_statistics(crops)
        
        # 生成报告数据
        report = {
            'image_path': image_path,
            'analysis_date': '2024-07-31',  # 实际应用中使用当前日期
            'total_crops_detected': len(crops),
            'total_diseases_detected': len(diseases),
            'crop_statistics': field_stats,
            'disease_summary': self.summarize_diseases(diseases),
            'recommendations': self.generate_recommendations(field_stats, diseases)
        }
        
        # 保存JSON报告
        report_file = os.path.join(output_dir, 'field_analysis_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可视化报告
        self.create_visual_report(crops, diseases, field_stats, output_dir)
        
        print(f"农田分析报告已保存到: {output_dir}")
        return report
    
    def summarize_diseases(self, diseases):
        """汇总病害信息"""
        if not diseases:
            return {'total': 0, 'types': {}, 'severity_distribution': {}}
        
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for disease in diseases:
            type_counts[disease['type']] += 1
            severity_counts[disease['severity']] += 1
        
        return {
            'total': len(diseases),
            'types': dict(type_counts),
            'severity_distribution': dict(severity_counts)
        }
    
    def generate_recommendations(self, field_stats, diseases):
        """生成农业建议"""
        recommendations = []
        
        # 基于作物分布的建议
        if 'unknown' in field_stats and field_stats['unknown']['count'] > 0:
            recommendations.append("建议对未识别的作物区域进行人工核查")
        
        # 基于病害的建议
        severe_diseases = [d for d in diseases if d.get('severity') == 'severe']
        if severe_diseases:
            recommendations.append(f"发现 {len(severe_diseases)} 个严重病害区域，建议立即处理")
        
        moderate_diseases = [d for d in diseases if d.get('severity') == 'moderate']
        if moderate_diseases:
            recommendations.append(f"发现 {len(moderate_diseases)} 个中等病害区域，建议加强监测")
        
        # 基于作物密度的建议
        total_crops = sum(stats['count'] for stats in field_stats.values())
        if total_crops < 10:
            recommendations.append("作物密度较低，考虑补种或调整种植策略")
        
        return recommendations
    
    def create_visual_report(self, crops, diseases, field_stats, output_dir):
        """创建可视化报告"""
        # 作物分布饼图
        if field_stats:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 作物类型分布
            crop_names = list(field_stats.keys())
            crop_counts = [field_stats[name]['count'] for name in crop_names]
            
            ax1.pie(crop_counts, labels=crop_names, autopct='%1.1f%%', startangle=90)
            ax1.set_title('作物类型分布')
            
            # 作物面积分布
            crop_areas = [field_stats[name]['total_area'] for name in crop_names]
            ax2.bar(crop_names, crop_areas, color=['green', 'yellow', 'red', 'blue'][:len(crop_names)])
            ax2.set_title('作物面积分布')
            ax2.set_ylabel('总面积 (像素)')
            
            # 病害统计
            if diseases:
                disease_types = defaultdict(int)
                for disease in diseases:
                    disease_types[disease['type']] += 1
                
                ax3.bar(disease_types.keys(), disease_types.values(), color='red', alpha=0.7)
                ax3.set_title('病害类型统计')
                ax3.set_ylabel('数量')
            else:
                ax3.text(0.5, 0.5, '未检测到病害', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('病害类型统计')
            
            # 作物健康度
            if diseases:
                severity_counts = defaultdict(int)
                for disease in diseases:
                    severity_counts[disease['severity']] += 1
                
                ax4.bar(severity_counts.keys(), severity_counts.values(), 
                       color=['green', 'orange', 'red'][:len(severity_counts)])
                ax4.set_title('病害严重程度分布')
                ax4.set_ylabel('数量')
            else:
                ax4.text(0.5, 0.5, '作物健康', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('病害严重程度分布')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'field_statistics.png'), dpi=150, bbox_inches='tight')
            plt.show()

def demo_crop_detection():
    """演示作物检测"""
    system = AgriculturalSegmentationSystem()
    
    print("\n=== 农作物检测演示 ===")
    
    # 使用dog.jpg作为演示（模拟农田图像）
    image_path = 'images/dog.jpg'
    if not os.path.exists(image_path):
        print(f"演示图像 {image_path} 不存在")
        return
    
    image, crops = system.detect_crops_in_field(image_path)
    
    # 可视化检测结果
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    
    # 显示检测到的作物
    colors = {'wheat': 'yellow', 'corn': 'green', 'tomato': 'red', 'lettuce': 'lightgreen', 'unknown': 'gray'}
    
    for i, crop in enumerate(crops[:10]):  # 只显示前10个
        mask = crop['mask']
        crop_type = crop['type']
        color = colors.get(crop_type, 'white')
        
        # 显示掩码
        show_mask(mask, plt.gca(), random_color=False)
        
        # 添加标签
        bbox = crop['bbox']
        center_x = bbox[0] + bbox[2] // 2
        center_y = bbox[1] + bbox[3] // 2
        
        plt.text(center_x, center_y, f"{crop_type}\n{crop['area']}px", 
                fontsize=10, color='white', weight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    plt.title(f"农作物检测结果 - 共检测到 {len(crops)} 个作物区域")
    plt.axis('off')
    plt.savefig('crop_detection_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 输出统计信息
    stats = system.calculate_field_statistics(crops)
    print(f"\n📊 作物统计:")
    for crop_type, info in stats.items():
        print(f"  {crop_type}: {info['count']} 个, 总面积: {info['total_area']} 像素 ({info['percentage']:.1f}%)")

def demo_disease_detection():
    """演示病害检测"""
    system = AgriculturalSegmentationSystem()
    
    print("\n=== 植物病害检测演示 ===")
    
    # 定义可疑病害点（在实际应用中可能通过颜色异常自动检测）
    disease_points = [
        [200, 150],  # 可疑病害点1
        [350, 250],  # 可疑病害点2
        [150, 300],  # 可疑病害点3
    ]
    
    image_path = 'images/truck.jpg'  # 使用truck.jpg作为演示
    if not os.path.exists(image_path):
        print(f"演示图像 {image_path} 不存在")
        return
    
    image, diseases = system.detect_plant_diseases(image_path, disease_points)
    
    # 可视化病害检测结果
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    
    # 显示病害区域
    severity_colors = {'mild': 'yellow', 'moderate': 'orange', 'severe': 'red'}
    
    for disease in diseases:
        mask = disease['mask']
        point = disease['point']
        severity = disease['severity']
        disease_type = disease['type']
        
        # 显示掩码
        show_mask(mask, plt.gca(), random_color=False)
        
        # 显示检测点
        show_points(np.array([point]), np.array([1]), plt.gca())
        
        # 添加标签
        plt.text(point[0], point[1] - 30, f"{disease_type}\n{severity}", 
                fontsize=10, color='white', weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor=severity_colors.get(severity, 'gray'), alpha=0.8))
    
    plt.title(f"植物病害检测结果 - 共检测到 {len(diseases)} 个病害区域")
    plt.axis('off')
    plt.savefig('disease_detection_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 输出病害分析
    print(f"\n🦠 病害分析:")
    for i, disease in enumerate(diseases, 1):
        print(f"  病害 {i}: {disease['type']} ({disease['severity']})")
        print(f"    面积: {disease['area']} 像素")
        print(f"    平均颜色: RGB{[int(c) for c in disease['avg_color']]}")

if __name__ == '__main__':
    print("=== SAM农业应用演示 ===")
    print("注意：此演示使用普通图像模拟农业场景，实际应用需要农田和作物的真实图像")
    
    try:
        # 演示作物检测
        demo_crop_detection()
        
        # 演示病害检测
        demo_disease_detection()
        
        print("\n✅ 农业应用演示完成！")
        print("💡 在实际农业应用中，建议:")
        print("   1. 使用无人机或卫星图像获取农田数据")
        print("   2. 建立作物和病害的专业数据库")
        print("   3. 结合多光谱成像技术")
        print("   4. 集成GPS定位信息用于精准农业")
        print("   5. 建立时序监测系统跟踪作物生长")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确保模型文件和图像文件存在")
