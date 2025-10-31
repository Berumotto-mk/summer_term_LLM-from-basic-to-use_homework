#!/usr/bin/env python3
"""
SAM2 批量图像分割应用
自动处理多张图片，适用于数据集处理
"""

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class BatchSegmenter:
    def __init__(self, model_path="checkpoints/sam2.1_hiera_tiny.pt"):
        """初始化批量分割器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 初始化批量分割器 (设备: {self.device})")
        
        # 加载模型
        sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", model_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("✅ 模型加载完成")
        
    def process_image_batch(self, image_dir, output_dir, strategy="center_click"):
        """
        批量处理图片
        strategy: "center_click", "auto_grid", "adaptive"
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ 在 {image_dir} 中未找到图片文件")
            return
        
        print(f"📁 找到 {len(image_files)} 张图片")
        
        results = {}
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n🖼️  处理图片 {i}/{len(image_files)}: {image_path.name}")
            
            try:
                result = self._process_single_image(image_path, output_dir, strategy)
                results[str(image_path)] = result
                print(f"✅ 完成处理: {image_path.name}")
                
            except Exception as e:
                print(f"❌ 处理失败 {image_path.name}: {e}")
                results[str(image_path)] = {"error": str(e)}
        
        # 保存处理结果
        results_path = os.path.join(output_dir, "batch_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ 批量处理完成! 结果保存在: {output_dir}")
        print(f"📊 处理结果: {results_path}")
        
        # 生成汇总报告
        self._generate_summary_report(results, output_dir)
        
        return results
    
    def _process_single_image(self, image_path, output_dir, strategy):
        """处理单张图片"""
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        self.predictor.set_image(image_array)
        
        height, width = image_array.shape[:2]
        
        # 根据策略生成点击点
        if strategy == "center_click":
            points = [[width//2, height//2]]
            labels = [1]
        elif strategy == "auto_grid":
            # 网格采样
            points = [
                [width//4, height//4],
                [width//2, height//2], 
                [width*3//4, height*3//4]
            ]
            labels = [1, 1, 1]
        elif strategy == "adaptive":
            # 自适应采样（基于图片大小）
            num_points = min(5, max(2, (width * height) // 500000))
            points = []
            labels = []
            for i in range(num_points):
                x = int(width * (0.2 + 0.6 * i / (num_points - 1)))
                y = int(height * (0.3 + 0.4 * i / (num_points - 1)))
                points.append([x, y])
                labels.append(1)
        
        # 执行分割
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=True,
        )
        
        # 选择最佳mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        # 保存结果
        output_name = f"{image_path.stem}_segmented.png"
        output_path = os.path.join(output_dir, output_name)
        
        self._save_segmentation_result(
            image_array, best_mask, points, labels, 
            output_path, best_score
        )
        
        return {
            "input_image": str(image_path),
            "output_image": output_path,
            "strategy": strategy,
            "points": points,
            "labels": labels,
            "best_score": float(best_score),
            "all_scores": [float(s) for s in scores],
            "image_size": [width, height]
        }
    
    def _save_segmentation_result(self, image, mask, points, labels, output_path, score):
        """保存分割结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图 + 点击点
        axes[0].imshow(image)
        for point, label in zip(points, labels):
            color = 'green' if label == 1 else 'red'
            marker = 'o' if label == 1 else 'x'
            axes[0].plot(point[0], point[1], color=color, marker=marker, 
                        markersize=8, markeredgecolor='white', linewidth=2)
        axes[0].set_title("Input + Points")
        axes[0].axis('off')
        
        # 分割结果
        axes[1].imshow(image)
        axes[1].imshow(mask, alpha=0.5, cmap='jet')
        axes[1].set_title(f"Segmentation (Score: {score:.3f})")
        axes[1].axis('off')
        
        # 纯mask
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title("Mask Only")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, results, output_dir):
        """生成汇总报告"""
        successful_results = [r for r in results.values() if "error" not in r]
        
        if not successful_results:
            return
        
        # 统计信息
        total_images = len(results)
        successful_images = len(successful_results)
        avg_score = np.mean([r["best_score"] for r in successful_results])
        
        strategies = {}
        for r in successful_results:
            strategy = r["strategy"]
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(r["best_score"])
        
        # 生成报告
        report_content = f"""# SAM2 批量分割报告

## 📊 处理统计
- **总图片数**: {total_images}
- **成功处理**: {successful_images}
- **失败数量**: {total_images - successful_images}
- **成功率**: {successful_images/total_images*100:.1f}%

## 🎯 分割质量
- **平均得分**: {avg_score:.3f}
- **最高得分**: {max([r["best_score"] for r in successful_results]):.3f}
- **最低得分**: {min([r["best_score"] for r in successful_results]):.3f}

## 🔧 策略分析
"""
        
        for strategy, scores in strategies.items():
            report_content += f"""
### {strategy}
- 图片数量: {len(scores)}
- 平均得分: {np.mean(scores):.3f}
- 得分范围: {min(scores):.3f} - {max(scores):.3f}
"""
        
        report_content += f"""
## 📁 输出文件
所有分割结果已保存在: `{output_dir}`

## 💡 使用建议
- 高得分(>0.5): 分割质量优秀
- 中等得分(0.3-0.5): 分割质量良好
- 低得分(<0.3): 建议调整点击策略或使用更大模型
"""
        
        report_path = os.path.join(output_dir, "batch_summary_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📋 汇总报告已生成: {report_path}")

def demo_batch_processing():
    """演示批量处理"""
    print("🚀 SAM2 批量图像分割演示")
    print("=" * 50)
    
    # 初始化批量分割器
    segmenter = BatchSegmenter()
    
    # 处理notebooks/images目录下的所有图片
    input_dir = "notebooks/images"
    output_dir = "sam2_demos/batch_results"
    
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 使用不同策略处理
    strategies = ["center_click", "auto_grid", "adaptive"]
    
    for strategy in strategies:
        print(f"\n🎯 使用策略: {strategy}")
        strategy_output_dir = os.path.join(output_dir, strategy)
        
        results = segmenter.process_image_batch(
            input_dir, 
            strategy_output_dir, 
            strategy
        )
        
        if results:
            successful = len([r for r in results.values() if "error" not in r])
            print(f"✅ 策略 {strategy} 完成: {successful}/{len(results)} 张图片成功处理")

if __name__ == "__main__":
    demo_batch_processing()
