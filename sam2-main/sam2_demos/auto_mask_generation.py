#!/usr/bin/env python3
"""
SAM2 自动掩码生成应用
支持全图自动分割，无需用户输入
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
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import json
from datetime import datetime

class AutoMaskGenerator:
    def __init__(self, model_path="checkpoints/sam2.1_hiera_tiny.pt"):
        """初始化自动掩码生成器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 初始化SAM2自动掩码生成器 (设备: {self.device})")
        
        # 构建SAM2模型
        sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", model_path, device=self.device)
        
        # 创建自动掩码生成器
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,        # 每边采样点数
            pred_iou_thresh=0.8,       # IoU阈值
            stability_score_thresh=0.9, # 稳定性分数阈值
            crop_n_layers=1,           # 裁剪层数
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # 最小掩码区域面积
        )
        print("✅ 自动掩码生成器初始化完成")
    
    def generate_masks(self, image_path):
        """生成图像的所有掩码"""
        print(f"🎯 生成图像掩码: {image_path}")
        
        # 读取图像
        image = Image.open(image_path)
        image_np = np.array(image)
        
        print(f"📏 图像尺寸: {image_np.shape}")
        
        # 生成掩码
        print("⚡ 执行自动掩码生成...")
        masks = self.mask_generator.generate(image_np)
        
        print(f"✅ 生成了 {len(masks)} 个掩码")
        
        # 按稳定性分数排序
        masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
        
        return image_np, masks
    
    def analyze_masks(self, masks):
        """分析掩码统计信息"""
        stats = {
            'total_masks': len(masks),
            'areas': [mask['area'] for mask in masks],
            'stability_scores': [mask['stability_score'] for mask in masks],
            'predicted_ious': [mask['predicted_iou'] for mask in masks]
        }
        
        stats['area_stats'] = {
            'min': min(stats['areas']) if stats['areas'] else 0,
            'max': max(stats['areas']) if stats['areas'] else 0,
            'mean': np.mean(stats['areas']) if stats['areas'] else 0,
            'median': np.median(stats['areas']) if stats['areas'] else 0
        }
        
        stats['score_stats'] = {
            'min_stability': min(stats['stability_scores']) if stats['stability_scores'] else 0,
            'max_stability': max(stats['stability_scores']) if stats['stability_scores'] else 0,
            'mean_stability': np.mean(stats['stability_scores']) if stats['stability_scores'] else 0,
            'min_iou': min(stats['predicted_ious']) if stats['predicted_ious'] else 0,
            'max_iou': max(stats['predicted_ious']) if stats['predicted_ious'] else 0,
            'mean_iou': np.mean(stats['predicted_ious']) if stats['predicted_ious'] else 0
        }
        
        return stats
    
    def visualize_all_masks(self, image, masks, output_path):
        """可视化所有掩码"""
        print("🎨 生成所有掩码可视化...")
        
        # 创建掩码叠加图
        mask_overlay = np.zeros_like(image)
        
        # 为每个掩码分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            color = (colors[i][:3] * 255).astype(np.uint8)
            mask_overlay[mask] = color
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # 掩码叠加
        axes[0, 1].imshow(image)
        axes[0, 1].imshow(mask_overlay, alpha=0.6)
        axes[0, 1].set_title(f"All Masks Overlay ({len(masks)} masks)")
        axes[0, 1].axis('off')
        
        # 纯掩码图
        axes[1, 0].imshow(mask_overlay)
        axes[1, 0].set_title("Masks Only")
        axes[1, 0].axis('off')
        
        # 统计信息
        stats = self.analyze_masks(masks)
        stats_text = f"""Mask Statistics:
Total Masks: {stats['total_masks']}

Area Statistics:
Min: {stats['area_stats']['min']:.0f}
Max: {stats['area_stats']['max']:.0f}
Mean: {stats['area_stats']['mean']:.0f}
Median: {stats['area_stats']['median']:.0f}

Quality Scores:
Stability: {stats['score_stats']['mean_stability']:.3f}
IoU: {stats['score_stats']['mean_iou']:.3f}"""
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Statistics")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 所有掩码可视化已保存: {output_path}")
        return stats
    
    def visualize_top_masks(self, image, masks, output_path, top_n=12):
        """可视化质量最高的掩码"""
        print(f"🎨 生成前{top_n}个最佳掩码可视化...")
        
        top_masks = masks[:top_n]
        
        # 计算网格布局
        cols = 4
        rows = (len(top_masks) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, mask_data in enumerate(top_masks):
            if i < len(axes):
                mask = mask_data['segmentation']
                area = mask_data['area']
                stability = mask_data['stability_score']
                
                # 显示原图和掩码
                axes[i].imshow(image)
                axes[i].imshow(mask, alpha=0.7, cmap='jet')
                axes[i].set_title(f"Mask {i+1}\nArea: {area}, Score: {stability:.3f}")
                axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(top_masks), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Top {len(top_masks)} Masks by Stability Score", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 顶级掩码可视化已保存: {output_path}")
    
    def create_mask_analysis(self, image, masks, output_dir):
        """创建详细的掩码分析"""
        print("📊 创建详细掩码分析...")
        
        stats = self.analyze_masks(masks)
        
        # 创建分析图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 面积分布直方图
        axes[0, 0].hist(stats['areas'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title("Mask Area Distribution")
        axes[0, 0].set_xlabel("Area (pixels)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 稳定性分数分布
        axes[0, 1].hist(stats['stability_scores'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title("Stability Score Distribution")
        axes[0, 1].set_xlabel("Stability Score")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU分数分布
        axes[0, 2].hist(stats['predicted_ious'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
        axes[0, 2].set_title("Predicted IoU Distribution")
        axes[0, 2].set_xlabel("Predicted IoU")
        axes[0, 2].set_ylabel("Count")
        axes[0, 2].grid(True, alpha=0.3)
        
        # 面积vs稳定性散点图
        axes[1, 0].scatter(stats['areas'], stats['stability_scores'], alpha=0.6, c='purple')
        axes[1, 0].set_title("Area vs Stability Score")
        axes[1, 0].set_xlabel("Area (pixels)")
        axes[1, 0].set_ylabel("Stability Score")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 面积vs IoU散点图
        axes[1, 1].scatter(stats['areas'], stats['predicted_ious'], alpha=0.6, c='orange')
        axes[1, 1].set_title("Area vs Predicted IoU")
        axes[1, 1].set_xlabel("Area (pixels)")
        axes[1, 1].set_ylabel("Predicted IoU")
        axes[1, 1].grid(True, alpha=0.3)
        
        # 稳定性vs IoU散点图
        axes[1, 2].scatter(stats['stability_scores'], stats['predicted_ious'], alpha=0.6, c='teal')
        axes[1, 2].set_title("Stability Score vs Predicted IoU")
        axes[1, 2].set_xlabel("Stability Score")
        axes[1, 2].set_ylabel("Predicted IoU")
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        analysis_path = os.path.join(output_dir, "mask_analysis.png")
        plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 掩码分析图表已保存: {analysis_path}")
        
        # 保存统计数据到JSON
        stats_path = os.path.join(output_dir, "mask_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"📋 统计数据已保存: {stats_path}")
        
        return stats

def demo_auto_mask_generation():
    """演示自动掩码生成"""
    print("🚀 SAM2 自动掩码生成演示")
    print("=" * 50)
    
    try:
        # 初始化自动掩码生成器
        generator = AutoMaskGenerator()
        
        # 测试图像
        test_images = [
            "notebooks/images/cars.jpg",
            "notebooks/images/groceries.jpg",
            "notebooks/images/truck.jpg"
        ]
        
        for image_path in test_images:
            if not os.path.exists(image_path):
                print(f"⚠️  图像不存在: {image_path}")
                continue
            
            print(f"\n🖼️  处理图像: {image_path}")
            
            # 生成掩码
            image, masks = generator.generate_masks(image_path)
            
            # 创建输出目录
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = f"sam2_demos/auto_masks_{image_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 可视化所有掩码
            all_masks_path = os.path.join(output_dir, "all_masks.png")
            stats = generator.visualize_all_masks(image, masks, all_masks_path)
            
            # 可视化顶级掩码
            top_masks_path = os.path.join(output_dir, "top_masks.png")
            generator.visualize_top_masks(image, masks, top_masks_path)
            
            # 创建详细分析
            generator.create_mask_analysis(image, masks, output_dir)
            
            # 创建报告
            create_auto_mask_report(image_path, stats, output_dir)
        
        print("\n✅ 自动掩码生成演示完成!")
        
    except Exception as e:
        print(f"❌ 自动掩码生成演示失败: {e}")
        import traceback
        traceback.print_exc()

def create_auto_mask_report(image_path, stats, output_dir):
    """创建自动掩码生成报告"""
    report_content = f"""# SAM2 自动掩码生成报告

## 📋 基本信息
- **图像路径**: `{image_path}`
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总掩码数**: {stats['total_masks']}

## 📊 掩码统计

### 面积统计
- **最小面积**: {stats['area_stats']['min']:.0f} 像素
- **最大面积**: {stats['area_stats']['max']:.0f} 像素
- **平均面积**: {stats['area_stats']['mean']:.0f} 像素
- **中位面积**: {stats['area_stats']['median']:.0f} 像素

### 质量评分
- **平均稳定性分数**: {stats['score_stats']['mean_stability']:.3f}
- **稳定性分数范围**: {stats['score_stats']['min_stability']:.3f} - {stats['score_stats']['max_stability']:.3f}
- **平均预测IoU**: {stats['score_stats']['mean_iou']:.3f}
- **预测IoU范围**: {stats['score_stats']['min_iou']:.3f} - {stats['score_stats']['max_iou']:.3f}

## 🎯 生成文件
1. `all_masks.png` - 所有掩码可视化
2. `top_masks.png` - 质量最高的掩码
3. `mask_analysis.png` - 详细统计分析
4. `mask_statistics.json` - 原始统计数据

## 💡 应用建议
- 高稳定性分数(>0.9)的掩码适合精确应用
- 大面积掩码通常对应主要对象
- 可根据面积筛选感兴趣的对象
- 结合IoU分数评估掩码质量

## 🔧 参数调整建议
- 增加`points_per_side`获得更多掩码
- 提高`pred_iou_thresh`获得更高质量掩码
- 调整`min_mask_region_area`过滤小对象
"""
    
    report_path = os.path.join(output_dir, "auto_mask_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📋 自动掩码报告已保存: {report_path}")

if __name__ == "__main__":
    demo_auto_mask_generation()
