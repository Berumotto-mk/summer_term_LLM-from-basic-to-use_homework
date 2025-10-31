#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
快速版SAM - 使用较小模型和优化设置
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 避免GUI阻塞
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from common_tools import show_mask, show_box, show_points

if __name__ == '__main__':
    # 使用较小的模型
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"  # 如果有vit_b或vit_l模型会更快
    model_type = "vit_h" 
    device = "cpu"
    image_path = 'images/truck.jpg'
    
    print("加载图像...")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 缩小图像以提高速度（可选）
    original_shape = image.shape
    scale_factor = 0.5  # 缩小到一半
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    image_resized = cv2.resize(image, (new_width, new_height))
    print(f"原始尺寸: {original_shape[:2]}, 缩放后: {image_resized.shape[:2]}")
    
    print("加载SAM模型...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("SAM模型加载成功")

    print("开始图像编码（预计30-60秒）...")
    predictor = SamPredictor(sam)
    predictor.set_image(image_resized)  # 使用缩放后的图像
    print("图像编码完成！")

    # 调整点坐标到缩放后的图像
    input_point = np.array([[int(500*scale_factor), int(375*scale_factor)], 
                           [int(1125*scale_factor), int(625*scale_factor)]])
    input_label = np.array([1, 0])

    print("生成掩码...")
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    print(f"成功生成 {len(masks)} 个掩码")
    
    # 保存结果而不是显示
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(12, 8))
        plt.imshow(image_resized)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=16)
        plt.axis('off')
        filename = f'result_mask_{i+1}_score_{score:.3f}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {filename}")
    
    print("所有结果已保存！")
