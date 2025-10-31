#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
简化版SAM测试脚本 - 用于诊断性能问题
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI阻塞
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

def main():
    print("=== SAM性能测试开始 ===")
    start_time = time.time()
    
    # 1. 检查图像
    print("1. 加载图像...")
    image_path = 'images/truck.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"   图像尺寸: {image.shape}")
    print(f"   图像加载耗时: {time.time() - start_time:.2f}秒")
    
    # 2. 检查模型文件
    checkpoint_time = time.time()
    print("2. 检查模型文件...")
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    import os
    if not os.path.exists(sam_checkpoint):
        print(f"错误：模型文件不存在 {sam_checkpoint}")
        return
    file_size = os.path.getsize(sam_checkpoint) / (1024*1024*1024)  # GB
    print(f"   模型文件大小: {file_size:.2f} GB")
    
    # 3. 加载模型
    print("3. 加载SAM模型...")
    model_type = "vit_h"
    device = "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print(f"   模型加载耗时: {time.time() - checkpoint_time:.2f}秒")
    
    # 4. 图像编码（最耗时的步骤）
    encoding_time = time.time()
    print("4. 图像编码（这是最耗时的步骤）...")
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    print(f"   图像编码耗时: {time.time() - encoding_time:.2f}秒")
    
    # 5. 快速预测测试
    prediction_time = time.time()
    print("5. 预测测试...")
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    print(f"   预测耗时: {time.time() - prediction_time:.2f}秒")
    print(f"   生成 {len(masks)} 个掩码，最高分数: {max(scores):.3f}")
    
    total_time = time.time() - start_time
    print(f"\n=== 总耗时: {total_time:.2f}秒 ===")
    print("测试完成！如果图像编码步骤超过2分钟，说明CPU性能较慢")

if __name__ == '__main__':
    main()
