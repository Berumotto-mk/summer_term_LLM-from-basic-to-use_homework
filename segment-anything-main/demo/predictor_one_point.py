#!/usr/bin/env python
# -*- coding: UTF-8 -*-
  
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from common_tools import show_mask, show_box, show_points

"""
-----------------------------------------------代码解释-----------------------------------------------
库的导入：导入了处理图像和可视化所需的库，如 numpy、torch、matplotlib、cv2，以及 Segment Anything 模型相关的库。

模型和设备设置：指定了 SAM 模型的检查点路径、模型类型和使用的计算设备（GPU）。

图像处理：读取图像并将其从 BGR 色彩空间转换为 RGB 色彩空间，以便在 Matplotlib 中正确显示。

模型加载：通过 sam_model_registry 加载指定类型的 SAM 模型，并将其移动到 GPU。

初始化预测器：创建 SamPredictor 实例，并通过 set_image 方法设置图像，生成必要的嵌入。

输入点设置：定义一个输入点的坐标和标签，用于指导模型生成对应的掩码。

显示输入点：在原始图像上显示输入点，帮助用户直观地了解输入点的位置。

掩码预测：使用 predict 方法进行掩码预测，生成多个掩码及其对应的得分。

掩码显示：通过循环遍历生成的掩码，将每个掩码与原始图像叠加显示，并显示掩码的质量得分。
"""

if __name__ == '__main__':
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"                       # 训练好的sam模型
    model_type = "vit_h"                                                 # 模型类型，使用 ViT-H
    device = "cpu"                                                      # 使用 CPU 进行计算
    image_path = 'images/truck.jpg'                                      # 图像路径
    
    print("开始加载模型...")  # 添加进度提示
    image = cv2.imread(image_path)  # 读取图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                       # 将图像从BGR转换为RGB
    print(f"图像加载成功，尺寸: {image.shape}")

    # 加载SAM模型并将其移动到指定设备
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("SAM 模型加载成功")

    # 初始化 SamPredictor 并设置图像
    print("开始图像编码（这可能需要1-2分钟）...")
    predictor = SamPredictor(sam)
    predictor.set_image(image)                         # 设置图像，生成必要的嵌入
    print("图像编码完成！")
    # 给一个坐标点，帮助生成mask区域
    # input_point = np.array([[500, 375]])             # 输入点坐标
    # input_label = np.array([1])                      # 输入点标签，1表示前景点

    input_point = np.array([[500, 375], [1125, 625]])  # 输入点坐标
    input_label = np.array([1, 0])                     # 输入点标签，1表示前景点, 0表示背景点

    # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask 可以将之前的掩码输入给模型进行再次的预测，也就是cascade的概念
    # 显示图像并在图像上标记输入点
    print("显示输入点图像...")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.title("输入点位置")
    plt.show()  # 恢复图像显示
    print("请关闭图像窗口继续...")

    # 使用输入点进行预测，生成掩码
    print("开始生成掩码...")
    masks, scores, logits = predictor.predict(
        point_coords=input_point,                # 输入点坐标
        point_labels=input_label,                # 输入点标签
        multimask_output=True,                   # 是否输出多个掩码
    )
    print(f"掩码生成完成！生成了 {len(masks)} 个候选掩码")

    # 展示生成的掩码
    for i, (mask, score) in enumerate(zip(masks, scores)):
        print(f"显示掩码 {i + 1}，分数: {score:.3f}")
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()  # 显示图像
        print(f"请关闭掩码 {i + 1} 窗口继续...")
    
    print("所有掩码显示完成！")
