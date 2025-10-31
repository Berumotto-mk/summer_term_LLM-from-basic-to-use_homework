#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
from common_tools import show_anns


if __name__ == '__main__':
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"  # 训练好的sam模型
    model_type = "vit_h"  # 模型类型，使用 ViT-H
    device = "cuda"  # 使用 GPU 进行计算
    # image_path = 'images/dog.jpg'
    image_path = 'images/person/2007_000762.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("step1: 图像加载成功")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("step2: 模型加载成功")
    # mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=2,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    masks = mask_generator.generate(image)
    print(len(masks))
    print(masks[0].keys())
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()
