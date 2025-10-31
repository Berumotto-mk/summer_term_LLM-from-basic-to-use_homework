#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：segment-anything-main 
@File    ：42_common_tools.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：通用的sam工具
@Date    ：2024/9/3 14:16 
'''

import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    '''
    展示mask
    '''
    # 颜色选择
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    # 获取掩码的高度和宽度
    h, w = mask.shape[-2:]
    # 通过将掩码与颜色组合，生成带有颜色的掩码图像
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # 在给定的轴（ax）上显示带颜色的掩码图像
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    '''
    展示坐标点
    '''
    # 获取标签为 1 的正样本点坐标
    pos_points = coords[labels == 1]
    # 获取标签为 0 的负样本点坐标
    neg_points = coords[labels == 0]
    # 绘制正样本点，颜色为绿色，形状为星号
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    # 绘制负样本点，颜色为红色，形状为星号
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    '''
    展示边界框
    '''
    # 获取框的左上角坐标
    x0, y0 = box[0], box[1]
    # 计算框的宽度和高度
    w, h = box[2] - box[0], box[3] - box[1]
    # 在给定的轴（ax）上添加一个矩形框，颜色为绿色，边缘宽度为
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
