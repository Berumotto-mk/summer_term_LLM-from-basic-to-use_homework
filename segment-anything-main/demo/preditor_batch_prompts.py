#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from common_tools import show_mask, show_box, show_points
import torch

"""
-----------------------------------------------代码解释-----------------------------------------------
利用SAM模型进行批量处理多个图像，并生成多个对象的掩码。
通过可视化结果，可以更直观地看到模型的预测效果。
这种批处理方式在处理大规模数据集或需要同时对多个图像进行推理时非常有用。
实测非常卡
"""

from segment_anything.utils.transforms import ResizeLongestSide


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()


if __name__ == '__main__':
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"  # 训练好的sam模型
    model_type = "vit_h"  # 模型类型，使用 ViT-H
    device = "cuda"  # 使用 GPU 进行计算
    image_path = 'images/truck.jpg'  # 图像路径
    image = cv2.imread(image_path)  # 读取图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB

    # 加载SAM模型并将其移动到指定设备
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("SAM 模型加载成功")

    # 初始化 SamPredictor 并设置图像
    predictor = SamPredictor(sam)
    predictor.set_image(image)  # 设置图像，生成必要的嵌入

    # 给一系列的内容
    image1 = image  # truck.jpg from above
    image1_boxes = torch.tensor([
        [75, 275, 1725, 850],
        [425, 600, 700, 875],
        [1375, 550, 1650, 800],
        [1240, 675, 1400, 750],
    ], device=sam.device)

    image2 = cv2.imread('images/groceries.jpg')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2_boxes = torch.tensor([
        [450, 170, 520, 350],
        [350, 190, 450, 350],
        [500, 170, 580, 350],
        [580, 170, 640, 350],
    ], device=sam.device)

    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    # 参数解释
    # image: 以 PyTorch 张量形式输入的图像，格式为 CHW。
    # original_size: 输入到 SAM 之前的图像大小，格式为 (H, W)。
    # point_coords: 批处理的点提示坐标。
    # point_labels: 批处理的点提示标签。
    # boxes: 批处理的输入框。
    # mask_inputs: 批处理的输入掩码。
    batched_input = [
        {
            'image': prepare_image(image1, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(image1_boxes, image1.shape[:2]),
            'original_size': image1.shape[:2]
        },
        {
            'image': prepare_image(image2, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(image2_boxes, image2.shape[:2]),
            'original_size': image2.shape[:2]
        }
    ]

    batched_output = sam(batched_input, multimask_output=False)

    batched_output[0].keys()

    fig, ax = plt.subplots(1, 2, figsize=(20, 20))

    ax[0].imshow(image1)
    for mask in batched_output[0]['masks']:
        show_mask(mask.cpu().numpy(), ax[0], random_color=True)
    for box in image1_boxes:
        show_box(box.cpu().numpy(), ax[0])
    ax[0].axis('off')

    ax[1].imshow(image2)
    for mask in batched_output[1]['masks']:
        show_mask(mask.cpu().numpy(), ax[1], random_color=True)
    for box in image2_boxes:
        show_box(box.cpu().numpy(), ax[1])
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()
