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

使用边界框和点同时作为提示词进行预测。

掩码预测：使用 predict 方法进行掩码预测，生成多个掩码及其对应的得分。

掩码显示：通过循环遍历生成的掩码，将每个掩码与原始图像叠加显示，并显示掩码的质量得分。
"""

if __name__ == '__main__':
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"                       # 训练好的sam模型
    model_type = "vit_h"                                                 # 模型类型，使用 ViT-H
    device = "cuda"                                                      # 使用 GPU 进行计算
    image_path = 'images/truck.jpg'                                      # 图像路径
    image = cv2.imread(image_path)  # 读取图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                       # 将图像从BGR转换为RGB

    # 加载SAM模型并将其移动到指定设备
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("SAM 模型加载成功")

    # 初始化 SamPredictor 并设置图像
    predictor = SamPredictor(sam)
    predictor.set_image(image)             # 设置图像，生成必要的嵌入
    # 给一个坐标点，帮助生成mask区域
    input_box = np.array([425, 600, 700, 875])
    input_point = np.array([[575, 750]])
    input_label = np.array([0])

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()
