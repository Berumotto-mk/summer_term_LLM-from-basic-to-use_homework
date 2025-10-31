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

使用边界框作为提示进行预测：使用边界框 (input_box) 作为提示，而不是使用点提示。predictor.predict 方法接受该边界框，并生成相应的掩码。设置 multimask_output=False 表示只需要一个掩码输出。

掩码预测：使用 predict 方法进行掩码预测，生成多个掩码及其对应的得分。

掩码显示：通过循环遍历生成的掩码，将每个掩码与原始图像叠加显示，并显示掩码的质量得分。
"""

if __name__ == '__main__':
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"                       # 训练好的sam模型
    model_type = "vit_h"                                                 # 模型类型，使用 ViT-H
    device = "cpu"                                                       # 使用 CPU 进行计算
    image_path = 'images/truck.jpg'                                      # 图像路径
    
    print("开始加载模型...")
    image = cv2.imread(image_path)  # 读取图像
    print("开始加载模型...")
    image = cv2.imread(image_path)  # 读取图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                       # 将图像从BGR转换为RGB
    print(f"图像加载成功，尺寸: {image.shape}")

    # 加载SAM模型并将其移动到指定设备
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("SAM 模型加载成功")

    # 初始化 SamPredictor 并设置图像
    print("开始图像编码...")
    predictor = SamPredictor(sam)
    predictor.set_image(image)             # 设置图像，生成必要的嵌入
    print("图像编码完成！")
    print("图像编码完成！")
    # 给一个坐标点，帮助生成mask区域
    input_point = np.array([[500, 375]])   # 输入点坐标
    input_label = np.array([1])            # 输入点标签，1表示前景点

    # 使用边界框作为sam网络输入的prompt
    print("使用边界框进行分割...")
    input_box = np.array([425, 600, 700, 875])  # [x1, y1, x2, y2] 格式
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    print(f"边界框分割完成！掩码分数: {scores[0]:.3f}")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.title(f"Bounding Box Segmentation (Score: {scores[0]:.3f})")
    plt.axis('off')
    plt.show()
