#!/usr/bin/env python3
"""
SAM2 快速使用示例
演示如何使用SAM2进行图像分割
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# 导入SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def main():
    print("🚀 SAM2 图像分割示例")
    print("=" * 50)
    
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 使用设备: {device}")
    
    # 2. 加载模型
    print("📦 加载SAM2模型...")
    
    # 选择模型配置和权重
    model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"  # 使用tiny模型，速度最快
    sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    
    if not os.path.exists(sam2_checkpoint):
        print(f"❌ 模型权重文件未找到: {sam2_checkpoint}")
        print("请先运行 cd checkpoints && bash download_ckpts.sh 下载模型权重")
        return
    
    try:
        # 构建SAM2模型
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        print("✅ SAM2模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 3. 检查示例图片
    print("\n🖼️  检查示例图片...")
    image_paths = [
        "notebooks/images/cars.jpg",
        "notebooks/images/truck.jpg", 
        "notebooks/images/groceries.jpg"
    ]
    
    available_images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            available_images.append(img_path)
            print(f"✅ 找到图片: {img_path}")
    
    if not available_images:
        print("❌ 未找到示例图片")
        print("请确保 notebooks/images/ 目录下有图片文件")
        return
    
    # 4. 加载并处理图片
    image_path = available_images[0]
    print(f"\n🔄 处理图片: {image_path}")
    
    try:
        # 加载图片
        image = Image.open(image_path)
        image = image.convert("RGB")
        image_array = np.array(image)
        
        print(f"📐 图片尺寸: {image_array.shape}")
        
        # 设置图片到预测器
        predictor.set_image(image_array)
        print("✅ 图片预处理完成!")
        
        # 5. 演示点击分割
        print("\n🎯 执行点击分割...")
        
        # 假设点击图片中心位置
        height, width = image_array.shape[:2]
        input_point = np.array([[width//2, height//2]])  # 点击中心
        input_label = np.array([1])  # 1表示前景点
        
        # 执行预测
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        print(f"✅ 分割完成!")
        print(f"   生成了 {len(masks)} 个候选mask")
        print(f"   最佳mask得分: {scores.max():.3f}")
        
        # 6. 保存结果
        print("\n💾 保存分割结果...")
        
        # 选择得分最高的mask
        best_mask = masks[scores.argmax()]
        
        # 创建可视化结果
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 原图
        axes[0].imshow(image_array)
        axes[0].plot(input_point[0, 0], input_point[0, 1], 'ro', markersize=10)
        axes[0].set_title("原图 + 点击点")
        axes[0].axis('off')
        
        # 分割结果
        axes[1].imshow(image_array)
        axes[1].imshow(best_mask, alpha=0.5, cmap='jet')
        axes[1].plot(input_point[0, 0], input_point[0, 1], 'ro', markersize=10)
        axes[1].set_title("分割结果")
        axes[1].axis('off')
        
        # 保存结果
        output_path = "sam2_demo_result.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 结果已保存到: {output_path}")
        
        print("\n" + "=" * 50)
        print("🎉 SAM2 演示完成!")
        print("\n📋 更多用法:")
        print("1. 查看 notebooks/ 目录下的Jupyter示例")
        print("2. 尝试不同的模型大小 (tiny/small/base_plus/large)")
        print("3. 使用多个点或框进行更精确的分割")
        print("4. 尝试视频分割功能")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
