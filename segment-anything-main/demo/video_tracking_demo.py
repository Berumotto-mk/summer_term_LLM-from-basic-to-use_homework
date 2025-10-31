#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
SAM视频分割演示 - 视频中物体的实时分割跟踪
应用场景：视频监控、运动分析、自动驾驶、视频编辑
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from common_tools import show_mask, show_points
import os
import time
import json

class VideoSegmentationTracker:
    def __init__(self, model_path="models/sam_vit_h_4b8939.pth", device="cpu"):
        self.sam_checkpoint = model_path
        self.model_type = "vit_h"
        self.device = device
        self.sam = None
        self.predictor = None
        self.tracking_history = []
        self.load_model()
    
    def load_model(self):
        """加载SAM模型"""
        print("正在加载视频分割模型...")
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        print("视频分割模型加载完成！")
    
    def track_object_in_images(self, image_paths, initial_point, output_dir="video_tracking_results"):
        """
        在图像序列中跟踪物体
        模拟视频跟踪的效果
        """
        os.makedirs(output_dir, exist_ok=True)
        tracking_data = []
        
        print(f"开始跟踪物体，共 {len(image_paths)} 帧图像")
        
        for frame_idx, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                print(f"警告：图像 {image_path} 不存在，跳过")
                continue
                
            print(f"处理第 {frame_idx + 1}/{len(image_paths)} 帧: {image_path}")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 设置图像
            start_time = time.time()
            self.predictor.set_image(image_rgb)
            
            # 使用点进行分割
            if frame_idx == 0:
                # 第一帧使用初始点
                current_point = initial_point
            else:
                # 后续帧使用前一帧的质心作为跟踪点
                if tracking_data:
                    last_mask = tracking_data[-1]['mask']
                    current_point = self.calculate_centroid(last_mask)
                else:
                    current_point = initial_point
            
            input_point = np.array([current_point])
            input_label = np.array([1])
            
            # 预测掩码
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            
            # 选择最佳掩码
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            
            processing_time = time.time() - start_time
            
            # 计算物体属性
            bbox = self.calculate_bbox(best_mask)
            area = np.sum(best_mask)
            centroid = self.calculate_centroid(best_mask)
            
            # 保存跟踪数据
            frame_data = {
                'frame_idx': frame_idx,
                'image_path': image_path,
                'point': current_point,
                'mask': best_mask,
                'score': float(best_score),
                'bbox': bbox,
                'area': int(area),
                'centroid': centroid,
                'processing_time': processing_time
            }
            tracking_data.append(frame_data)
            
            # 保存可视化结果
            self.save_tracking_frame(image_rgb, best_mask, current_point, 
                                   frame_idx, best_score, output_dir)
            
            print(f"  分割得分: {best_score:.3f}, 处理时间: {processing_time:.2f}秒")
        
        # 保存跟踪报告
        self.save_tracking_report(tracking_data, output_dir)
        return tracking_data
    
    def calculate_centroid(self, mask):
        """计算掩码的质心"""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            centroid_x = int(np.mean(x_indices))
            centroid_y = int(np.mean(y_indices))
            return [centroid_x, centroid_y]
        return [0, 0]
    
    def calculate_bbox(self, mask):
        """计算掩码的边界框"""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            return [int(np.min(x_indices)), int(np.min(y_indices)),
                   int(np.max(x_indices)), int(np.max(y_indices))]
        return [0, 0, 0, 0]
    
    def save_tracking_frame(self, image, mask, point, frame_idx, score, output_dir):
        """保存单帧跟踪结果"""
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(np.array([point]), np.array([1]), plt.gca())
        
        # 添加信息文本
        plt.text(10, 30, f"Frame: {frame_idx + 1}", fontsize=12, color='white', 
                weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        plt.text(10, 60, f"Score: {score:.3f}", fontsize=12, color='white', 
                weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        plt.title(f"Object Tracking - Frame {frame_idx + 1}")
        plt.axis('off')
        
        filename = f"frame_{frame_idx:04d}_tracking.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_tracking_report(self, tracking_data, output_dir):
        """保存跟踪分析报告"""
        if not tracking_data:
            return
        
        # 准备报告数据
        report_data = {
            'total_frames': len(tracking_data),
            'average_score': float(np.mean([d['score'] for d in tracking_data])),
            'average_area': float(np.mean([d['area'] for d in tracking_data])),
            'average_processing_time': float(np.mean([d['processing_time'] for d in tracking_data])),
            'frames': []
        }
        
        for data in tracking_data:
            frame_info = {
                'frame_idx': data['frame_idx'],
                'image_path': data['image_path'],
                'point': data['point'],
                'score': data['score'],
                'bbox': data['bbox'],
                'area': data['area'],
                'centroid': data['centroid'],
                'processing_time': data['processing_time']
            }
            report_data['frames'].append(frame_info)
        
        # 保存JSON报告
        report_file = os.path.join(output_dir, "tracking_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # 生成跟踪轨迹图
        self.plot_tracking_trajectory(tracking_data, output_dir)
        
        print(f"跟踪报告已保存到: {report_file}")
    
    def plot_tracking_trajectory(self, tracking_data, output_dir):
        """绘制跟踪轨迹图"""
        if len(tracking_data) < 2:
            return
        
        # 提取轨迹数据
        centroids = [data['centroid'] for data in tracking_data]
        scores = [data['score'] for data in tracking_data]
        areas = [data['area'] for data in tracking_data]
        frames = [data['frame_idx'] for data in tracking_data]
        
        x_coords = [c[0] for c in centroids]
        y_coords = [c[1] for c in centroids]
        
        # 创建轨迹图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 轨迹图
        ax1.plot(x_coords, y_coords, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('X 坐标')
        ax1.set_ylabel('Y 坐标')
        ax1.set_title('物体跟踪轨迹')
        ax1.grid(True, alpha=0.3)
        
        # 分割得分随时间变化
        ax2.plot(frames, scores, 'g-o', linewidth=2)
        ax2.set_xlabel('帧数')
        ax2.set_ylabel('分割得分')
        ax2.set_title('分割质量随时间变化')
        ax2.grid(True, alpha=0.3)
        
        # 物体面积随时间变化
        ax3.plot(frames, areas, 'r-o', linewidth=2)
        ax3.set_xlabel('帧数')
        ax3.set_ylabel('物体面积 (像素)')
        ax3.set_title('物体大小随时间变化')
        ax3.grid(True, alpha=0.3)
        
        # X坐标随时间变化
        ax4.plot(frames, x_coords, 'c-o', linewidth=2, label='X坐标')
        ax4.plot(frames, y_coords, 'm-o', linewidth=2, label='Y坐标')
        ax4.set_xlabel('帧数')
        ax4.set_ylabel('坐标')
        ax4.set_title('物体位置随时间变化')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tracking_analysis.png'), dpi=150, bbox_inches='tight')
        plt.show()

def demo_multi_object_tracking():
    """演示多物体跟踪"""
    tracker = VideoSegmentationTracker()
    
    # 模拟视频序列（使用多张图像）
    image_sequence = [
        'images/truck.jpg',
        'images/dog.jpg',
        'images/groceries.jpg'
    ]
    
    # 检查图像是否存在
    available_images = [img for img in image_sequence if os.path.exists(img)]
    if not available_images:
        print("警告：找不到演示图像，请确保images文件夹中有图像文件")
        return
    
    print(f"\n=== 多物体跟踪演示 ===")
    print(f"使用 {len(available_images)} 张图像模拟视频跟踪")
    
    # 定义初始跟踪点
    initial_point = [400, 300]  # 图像中心附近
    
    # 执行跟踪
    tracking_results = tracker.track_object_in_images(
        available_images, 
        initial_point,
        "video_tracking_demo"
    )
    
    # 输出跟踪统计
    if tracking_results:
        total_frames = len(tracking_results)
        avg_score = np.mean([r['score'] for r in tracking_results])
        avg_area = np.mean([r['area'] for r in tracking_results])
        
        print(f"\n📊 跟踪统计:")
        print(f"  总帧数: {total_frames}")
        print(f"  平均分割得分: {avg_score:.3f}")
        print(f"  平均物体面积: {avg_area:.0f} 像素")
        
        # 显示轨迹变化
        centroids = [r['centroid'] for r in tracking_results]
        print(f"  轨迹点: {centroids}")

def demo_object_motion_analysis():
    """演示物体运动分析"""
    print(f"\n=== 物体运动分析演示 ===")
    
    # 模拟运动数据（实际应用中从视频跟踪获得）
    motion_data = [
        {'frame': 0, 'x': 100, 'y': 100, 'area': 1500},
        {'frame': 1, 'x': 120, 'y': 105, 'area': 1520},
        {'frame': 2, 'x': 140, 'y': 110, 'area': 1480},
        {'frame': 3, 'x': 160, 'y': 120, 'area': 1550},
        {'frame': 4, 'x': 185, 'y': 125, 'area': 1600},
    ]
    
    # 分析运动特征
    print("分析物体运动特征:")
    
    for i in range(1, len(motion_data)):
        prev = motion_data[i-1]
        curr = motion_data[i]
        
        # 计算位移
        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        displacement = np.sqrt(dx**2 + dy**2)
        
        # 计算速度（像素/帧）
        velocity = displacement
        
        # 计算面积变化率
        area_change = (curr['area'] - prev['area']) / prev['area'] * 100
        
        print(f"  帧 {prev['frame']} -> {curr['frame']}:")
        print(f"    位移: {displacement:.2f} 像素")
        print(f"    速度: {velocity:.2f} 像素/帧") 
        print(f"    面积变化: {area_change:+.1f}%")
    
    # 绘制运动轨迹
    x_coords = [d['x'] for d in motion_data]
    y_coords = [d['y'] for d in motion_data]
    areas = [d['area'] for d in motion_data]
    frames = [d['frame'] for d in motion_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 运动轨迹
    ax1.plot(x_coords, y_coords, 'bo-', linewidth=2, markersize=8)
    for i, (x, y, frame) in enumerate(zip(x_coords, y_coords, frames)):
        ax1.annotate(f'F{frame}', (x, y), xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.set_title('物体运动轨迹')
    ax1.grid(True, alpha=0.3)
    
    # 面积变化
    ax2.plot(frames, areas, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('帧数')
    ax2.set_ylabel('物体面积')
    ax2.set_title('物体大小变化')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('motion_analysis_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("=== SAM视频分割与跟踪演示 ===")
    
    try:
        # 演示多物体跟踪
        demo_multi_object_tracking()
        
        # 演示运动分析
        demo_object_motion_analysis()
        
        print("\n✅ 视频分割演示完成！")
        print("💡 在实际视频应用中，建议:")
        print("   1. 使用连续的视频帧而非静态图像")
        print("   2. 实现更智能的跟踪点更新策略")
        print("   3. 添加多目标跟踪功能")
        print("   4. 优化处理速度以实现实时跟踪")
        print("   5. 集成Kalman滤波等预测算法")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确保模型文件和图像文件存在")
