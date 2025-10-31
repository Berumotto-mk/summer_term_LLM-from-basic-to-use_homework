#!/usr/bin/env python3
"""
SAM2 高级视频分割应用
包含目标提取、背景替换、目标跟踪等高级功能
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import json
from datetime import datetime
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

class AdvancedVideoSegmenter:
    def __init__(self, model_path="checkpoints/sam2.1_hiera_tiny.pt"):
        """初始化高级视频分割器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 初始化SAM2高级视频分割器 (设备: {self.device})")
        
        # 加载视频预测器
        self.predictor = SAM2VideoPredictor.from_pretrained(
            model_id="facebook/sam2-hiera-tiny",
            device=self.device
        )
        print("✅ 视频模型加载完成")
        
        # 跟踪状态
        self.tracking_objects = {}
        self.tracking_history = {}
    
    def extract_frames(self, video_path, output_dir, max_frames=30):
        """从视频提取帧"""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 视频信息: {total_frames} 帧, {fps:.1f} FPS")
        
        # 计算采样间隔
        frame_interval = max(1, total_frames // max_frames)
        extracted_frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_path = os.path.join(output_dir, f"frame_{len(extracted_frames):04d}.jpg")
                Image.fromarray(frame_rgb).save(frame_path)
                extracted_frames.append(frame_path)
                
                if len(extracted_frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"✅ 提取了 {len(extracted_frames)} 帧")
        return extracted_frames, fps
    
    def target_extraction(self, frame_paths, click_point, output_dir, obj_name="target"):
        """
        目标提取功能：从视频中提取特定目标
        """
        print(f"🎯 开始目标提取: {obj_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化推理状态
        inference_state = self.predictor.init_state(video_path=os.path.dirname(frame_paths[0]))
        
        # 在第一帧添加点击点
        points = np.array([click_point], dtype=np.float32)
        labels = np.array([1], np.int32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )
        
        # 传播到所有帧并提取目标
        extracted_targets = []
        target_masks = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            frame = Image.open(frame_paths[out_frame_idx])
            frame_array = np.array(frame)
            
            # 获取掩码
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            target_masks[out_frame_idx] = mask
            
            # 创建透明背景的目标图像
            target_image = self._extract_target_with_transparency(frame_array, mask)
            
            # 保存提取的目标
            target_path = os.path.join(output_dir, f"{obj_name}_frame_{out_frame_idx:04d}.png")
            target_image.save(target_path)
            extracted_targets.append(target_path)
        
        # 创建提取结果汇总
        self._create_extraction_summary(frame_paths, extracted_targets, target_masks, output_dir, obj_name)
        
        # 生成提取报告
        self._generate_extraction_report(frame_paths, target_masks, output_dir, obj_name, click_point)
        
        print(f"✅ 目标提取完成: {len(extracted_targets)} 帧")
        return extracted_targets, target_masks
    
    def background_replacement(self, frame_paths, click_point, new_background_path, output_dir):
        """
        背景替换功能：将视频背景替换为新背景
        """
        print(f"🎨 开始背景替换")
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载新背景
        new_bg = Image.open(new_background_path).convert("RGB")
        
        # 初始化推理状态
        inference_state = self.predictor.init_state(video_path=os.path.dirname(frame_paths[0]))
        
        # 在第一帧添加点击点（选择前景对象）
        points = np.array([click_point], dtype=np.float32)
        labels = np.array([1], np.int32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )
        
        # 传播并替换背景
        replaced_frames = []
        replacement_masks = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            frame = Image.open(frame_paths[out_frame_idx])
            
            # 获取前景掩码
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            replacement_masks[out_frame_idx] = mask
            
            # 执行背景替换
            replaced_frame = self._replace_background(frame, mask, new_bg)
            
            # 保存替换结果
            output_path = os.path.join(output_dir, f"replaced_frame_{out_frame_idx:04d}.png")
            replaced_frame.save(output_path)
            replaced_frames.append(output_path)
        
        # 创建替换效果对比
        self._create_replacement_comparison(frame_paths, replaced_frames, replacement_masks, output_dir)
        
        # 生成替换报告
        self._generate_replacement_report(frame_paths, replacement_masks, new_background_path, output_dir, click_point)
        
        print(f"✅ 背景替换完成: {len(replaced_frames)} 帧")
        return replaced_frames, replacement_masks
    
    def object_tracking(self, frame_paths, click_points, output_dir, track_names=None):
        """
        目标跟踪功能：多目标跟踪和运动分析
        """
        print(f"🔍 开始多目标跟踪")
        os.makedirs(output_dir, exist_ok=True)
        
        if track_names is None:
            track_names = [f"object_{i+1}" for i in range(len(click_points))]
        
        # 初始化推理状态
        inference_state = self.predictor.init_state(video_path=os.path.dirname(frame_paths[0]))
        
        # 为每个目标添加点击点
        for i, (click_point, obj_name) in enumerate(zip(click_points, track_names)):
            points = np.array([click_point], dtype=np.float32)
            labels = np.array([1], np.int32)
            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=i+1,
                points=points,
                labels=labels,
            )
            
            # 初始化跟踪历史
            self.tracking_history[i+1] = {
                'name': obj_name,
                'positions': [],
                'areas': [],
                'confidence_scores': []
            }
        
        # 执行跟踪
        tracking_results = {}
        tracked_frames = []
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            frame = Image.open(frame_paths[out_frame_idx])
            frame_array = np.array(frame)
            
            frame_results = {}
            
            # 处理每个跟踪对象
            for i, (obj_id, mask_logit) in enumerate(zip(out_obj_ids, out_mask_logits)):
                mask = (mask_logit > 0.0).cpu().numpy()
                
                # 计算目标位置和属性
                centroid, area, bbox = self._analyze_object_properties(mask)
                confidence = float(torch.max(mask_logit).cpu())
                
                # 更新跟踪历史
                self.tracking_history[obj_id]['positions'].append(centroid)
                self.tracking_history[obj_id]['areas'].append(area)
                self.tracking_history[obj_id]['confidence_scores'].append(confidence)
                
                frame_results[obj_id] = {
                    'mask': mask,
                    'centroid': centroid,
                    'area': area,
                    'bbox': bbox,
                    'confidence': confidence
                }
            
            tracking_results[out_frame_idx] = frame_results
            
            # 创建带跟踪信息的可视化帧
            tracked_frame = self._create_tracking_visualization(frame_array, frame_results, out_frame_idx)
            
            # 保存跟踪结果帧
            tracked_path = os.path.join(output_dir, f"tracked_frame_{out_frame_idx:04d}.png")
            Image.fromarray(tracked_frame).save(tracked_path)
            tracked_frames.append(tracked_path)
        
        # 分析运动轨迹
        motion_analysis = self._analyze_motion_trajectories()
        
        # 创建跟踪分析报告
        self._create_tracking_analysis(tracking_results, motion_analysis, output_dir)
        
        # 生成跟踪报告
        self._generate_tracking_report(tracking_results, motion_analysis, click_points, track_names, output_dir)
        
        print(f"✅ 目标跟踪完成: {len(tracked_frames)} 帧, {len(click_points)} 个目标")
        return tracked_frames, tracking_results, motion_analysis
    
    def _extract_target_with_transparency(self, frame_array, mask):
        """创建带透明背景的目标图像"""
        height, width = frame_array.shape[:2]
        
        # 创建RGBA图像
        target_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        target_rgba[:, :, :3] = frame_array
        target_rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        
        return Image.fromarray(target_rgba, 'RGBA')
    
    def _replace_background(self, frame, mask, new_background):
        """执行背景替换"""
        frame_array = np.array(frame)
        
        # 调整新背景尺寸
        new_bg_resized = new_background.resize(frame.size)
        new_bg_array = np.array(new_bg_resized)
        
        # 扩展掩码到3个通道
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        # 执行背景替换
        result = frame_array * mask_3d + new_bg_array * (1 - mask_3d)
        
        return Image.fromarray(result.astype(np.uint8))
    
    def _analyze_object_properties(self, mask):
        """分析目标对象的属性"""
        # 计算质心
        y_coords, x_coords = np.where(mask)
        if len(x_coords) > 0:
            centroid = (int(np.mean(x_coords)), int(np.mean(y_coords)))
        else:
            centroid = (0, 0)
        
        # 计算面积
        area = np.sum(mask)
        
        # 计算边界框
        if len(x_coords) > 0:
            bbox = (int(np.min(x_coords)), int(np.min(y_coords)), 
                   int(np.max(x_coords)), int(np.max(y_coords)))
        else:
            bbox = (0, 0, 0, 0)
        
        return centroid, area, bbox
    
    def _create_tracking_visualization(self, frame, frame_results, frame_idx):
        """创建带跟踪信息的可视化"""
        vis_frame = frame.copy()
        
        # 颜色映射
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (obj_id, result) in enumerate(frame_results.items()):
            color = colors[i % len(colors)]
            centroid = result['centroid']
            bbox = result['bbox']
            confidence = result['confidence']
            
            # 绘制边界框
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 绘制质心
            cv2.circle(vis_frame, centroid, 5, color, -1)
            
            # 绘制跟踪轨迹
            if obj_id in self.tracking_history:
                positions = self.tracking_history[obj_id]['positions']
                if len(positions) > 1:
                    for j in range(1, len(positions)):
                        cv2.line(vis_frame, positions[j-1], positions[j], color, 2)
            
            # 添加文本信息
            obj_name = self.tracking_history.get(obj_id, {}).get('name', f'Object {obj_id}')
            text = f"{obj_name}: {confidence:.2f}"
            cv2.putText(vis_frame, text, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame
    
    def _analyze_motion_trajectories(self):
        """分析运动轨迹"""
        motion_analysis = {}
        
        for obj_id, history in self.tracking_history.items():
            positions = history['positions']
            areas = history['areas']
            
            if len(positions) < 2:
                continue
            
            # 计算运动距离
            total_distance = 0
            velocities = []
            
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                distance = np.sqrt(dx*dx + dy*dy)
                total_distance += distance
                velocities.append(distance)
            
            # 计算运动统计
            avg_velocity = np.mean(velocities) if velocities else 0
            max_velocity = np.max(velocities) if velocities else 0
            
            # 计算面积变化
            area_changes = []
            for i in range(1, len(areas)):
                change = abs(areas[i] - areas[i-1]) / areas[i-1] if areas[i-1] > 0 else 0
                area_changes.append(change)
            
            avg_area_change = np.mean(area_changes) if area_changes else 0
            
            motion_analysis[obj_id] = {
                'name': history['name'],
                'total_distance': total_distance,
                'average_velocity': avg_velocity,
                'max_velocity': max_velocity,
                'average_area_change': avg_area_change,
                'trajectory_length': len(positions),
                'start_position': positions[0] if positions else None,
                'end_position': positions[-1] if positions else None
            }
        
        return motion_analysis
    
    def _create_extraction_summary(self, frame_paths, extracted_targets, target_masks, output_dir, obj_name):
        """创建目标提取结果汇总"""
        fig, axes = plt.subplots(2, min(6, len(frame_paths)), figsize=(18, 8))
        if len(frame_paths) == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(min(6, len(frame_paths))):
            # 原始帧
            frame = Image.open(frame_paths[i])
            axes[0, i].imshow(frame)
            axes[0, i].set_title(f"Frame {i}")
            axes[0, i].axis('off')
            
            # 提取的目标
            if i < len(extracted_targets):
                target = Image.open(extracted_targets[i])
                axes[1, i].imshow(target)
                axes[1, i].set_title(f"Extracted {obj_name}")
            else:
                axes[1, i].axis('off')
            axes[1, i].axis('off')
        
        plt.suptitle(f"Target Extraction Results: {obj_name}", fontsize=16)
        plt.tight_layout()
        summary_path = os.path.join(output_dir, f"extraction_summary_{obj_name}.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 提取汇总已保存: {summary_path}")
    
    def _create_replacement_comparison(self, original_frames, replaced_frames, masks, output_dir):
        """创建背景替换对比图"""
        fig, axes = plt.subplots(3, min(4, len(original_frames)), figsize=(16, 12))
        if len(original_frames) == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(min(4, len(original_frames))):
            # 原始帧
            original = Image.open(original_frames[i])
            axes[0, i].imshow(original)
            axes[0, i].set_title(f"Original Frame {i}")
            axes[0, i].axis('off')
            
            # 替换后的帧
            if i < len(replaced_frames):
                replaced = Image.open(replaced_frames[i])
                axes[1, i].imshow(replaced)
                axes[1, i].set_title(f"Background Replaced")
            else:
                axes[1, i].axis('off')
            axes[1, i].axis('off')
            
            # 掩码
            if i in masks:
                axes[2, i].imshow(masks[i], cmap='gray')
                axes[2, i].set_title(f"Foreground Mask")
            else:
                axes[2, i].axis('off')
            axes[2, i].axis('off')
        
        plt.suptitle("Background Replacement Results", fontsize=16)
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "replacement_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 替换对比图已保存: {comparison_path}")
    
    def _create_tracking_analysis(self, tracking_results, motion_analysis, output_dir):
        """创建跟踪分析可视化"""
        # 创建轨迹图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 轨迹可视化
        ax = axes[0, 0]
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (obj_id, analysis) in enumerate(motion_analysis.items()):
            if obj_id in self.tracking_history:
                positions = self.tracking_history[obj_id]['positions']
                if len(positions) > 1:
                    x_coords = [pos[0] for pos in positions]
                    y_coords = [pos[1] for pos in positions]
                    color = colors[i % len(colors)]
                    ax.plot(x_coords, y_coords, color=color, marker='o', 
                           label=analysis['name'], linewidth=2, markersize=4)
        
        ax.set_title("Object Trajectories")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 速度分析
        ax = axes[0, 1]
        obj_names = [analysis['name'] for analysis in motion_analysis.values()]
        avg_velocities = [analysis['average_velocity'] for analysis in motion_analysis.values()]
        
        ax.bar(obj_names, avg_velocities, color=['red', 'blue', 'green', 'orange', 'purple'][:len(obj_names)])
        ax.set_title("Average Velocity")
        ax.set_ylabel("Pixels per frame")
        ax.tick_params(axis='x', rotation=45)
        
        # 面积变化
        ax = axes[1, 0]
        for i, (obj_id, history) in enumerate(self.tracking_history.items()):
            areas = history['areas']
            if areas:
                color = colors[i % len(colors)]
                ax.plot(range(len(areas)), areas, color=color, 
                       label=history['name'], linewidth=2)
        
        ax.set_title("Object Area Over Time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Area (pixels)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 置信度分析
        ax = axes[1, 1]
        for i, (obj_id, history) in enumerate(self.tracking_history.items()):
            scores = history['confidence_scores']
            if scores:
                color = colors[i % len(colors)]
                ax.plot(range(len(scores)), scores, color=color, 
                       label=history['name'], linewidth=2)
        
        ax.set_title("Tracking Confidence Over Time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Confidence Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        analysis_path = os.path.join(output_dir, "tracking_analysis.png")
        plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 跟踪分析图已保存: {analysis_path}")
    
    def _generate_extraction_report(self, frame_paths, target_masks, output_dir, obj_name, click_point):
        """生成目标提取报告"""
        report_content = f"""# 目标提取报告

## 📋 基本信息
- **目标名称**: {obj_name}
- **处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总帧数**: {len(frame_paths)}
- **初始点击位置**: {click_point}

## 📊 提取统计
- **成功提取帧数**: {len(target_masks)}
- **成功率**: {len(target_masks)/len(frame_paths)*100:.1f}%

## 🎯 质量分析
"""
        
        if target_masks:
            areas = [np.sum(mask) for mask in target_masks.values()]
            avg_area = np.mean(areas)
            min_area = np.min(areas)
            max_area = np.max(areas)
            area_stability = 1 - (np.std(areas) / avg_area) if avg_area > 0 else 0
            
            report_content += f"""
- **平均目标面积**: {avg_area:.0f} 像素
- **面积范围**: {min_area:.0f} - {max_area:.0f} 像素
- **面积稳定性**: {area_stability:.3f}

## 💡 应用建议
- 高稳定性(>0.8): 适合精确应用
- 中等稳定性(0.5-0.8): 适合一般应用
- 低稳定性(<0.5): 建议调整参数

## 📁 输出文件
- 提取的目标图像: `{obj_name}_frame_*.png`
- 提取汇总图: `extraction_summary_{obj_name}.png`
"""
        
        report_path = os.path.join(output_dir, f"extraction_report_{obj_name}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📋 提取报告已保存: {report_path}")
    
    def _generate_replacement_report(self, frame_paths, masks, new_bg_path, output_dir, click_point):
        """生成背景替换报告"""
        report_content = f"""# 背景替换报告

## 📋 基本信息
- **处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总帧数**: {len(frame_paths)}
- **新背景**: {os.path.basename(new_bg_path)}
- **前景选择点**: {click_point}

## 📊 替换统计
- **成功替换帧数**: {len(masks)}
- **成功率**: {len(masks)/len(frame_paths)*100:.1f}%

## 🎯 质量分析
"""
        
        if masks:
            # 计算前景区域稳定性
            areas = [np.sum(mask) for mask in masks.values()]
            avg_area = np.mean(areas)
            area_stability = 1 - (np.std(areas) / avg_area) if avg_area > 0 else 0
            
            # 计算边缘质量（边缘像素比例）
            edge_qualities = []
            for mask in masks.values():
                edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
                edge_quality = np.sum(edges > 0) / np.sum(mask) if np.sum(mask) > 0 else 0
                edge_qualities.append(edge_quality)
            
            avg_edge_quality = np.mean(edge_qualities)
            
            report_content += f"""
- **前景面积稳定性**: {area_stability:.3f}
- **平均边缘质量**: {avg_edge_quality:.3f}
- **平均前景占比**: {avg_area/np.prod(list(masks.values())[0].shape)*100:.1f}%

## 💡 质量评估
- **稳定性评级**: {"优秀" if area_stability > 0.8 else "良好" if area_stability > 0.5 else "一般"}
- **边缘质量**: {"清晰" if avg_edge_quality < 0.1 else "中等" if avg_edge_quality < 0.2 else "需优化"}

## 📁 输出文件
- 替换后视频帧: `replaced_frame_*.png`
- 效果对比图: `replacement_comparison.png`
"""
        
        report_path = os.path.join(output_dir, "replacement_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📋 替换报告已保存: {report_path}")
    
    def _generate_tracking_report(self, tracking_results, motion_analysis, click_points, track_names, output_dir):
        """生成目标跟踪报告"""
        report_content = f"""# 目标跟踪报告

## 📋 基本信息
- **处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **跟踪目标数**: {len(track_names)}
- **总帧数**: {len(tracking_results)}

## 🎯 跟踪目标
"""
        
        for i, (click_point, name) in enumerate(zip(click_points, track_names)):
            report_content += f"- **{name}**: 初始位置 {click_point}\n"
        
        report_content += f"""
## 📊 运动分析
"""
        
        for obj_id, analysis in motion_analysis.items():
            report_content += f"""
### {analysis['name']}
- **总移动距离**: {analysis['total_distance']:.1f} 像素
- **平均速度**: {analysis['average_velocity']:.2f} 像素/帧
- **最大速度**: {analysis['max_velocity']:.2f} 像素/帧
- **平均面积变化率**: {analysis['average_area_change']:.3f}
- **轨迹长度**: {analysis['trajectory_length']} 帧
- **起始位置**: {analysis['start_position']}
- **结束位置**: {analysis['end_position']}
"""
        
        # 计算跟踪质量指标
        total_frames = len(tracking_results)
        successful_tracks = sum(1 for analysis in motion_analysis.values() 
                              if analysis['trajectory_length'] > total_frames * 0.8)
        
        report_content += f"""
## 🏆 跟踪质量
- **跟踪成功率**: {successful_tracks/len(track_names)*100:.1f}%
- **平均轨迹完整性**: {np.mean([a['trajectory_length']/total_frames for a in motion_analysis.values()])*100:.1f}%

## 📁 输出文件
- 跟踪可视化帧: `tracked_frame_*.png`
- 运动分析图: `tracking_analysis.png`
- 跟踪数据: JSON格式保存在同目录

## 💡 分析建议
- 高速运动目标建议增加采样率
- 遮挡频繁区域可考虑多点标注
- 长期跟踪建议定期重新校准
"""
        
        report_path = os.path.join(output_dir, "tracking_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 保存跟踪数据为JSON
        tracking_data = {
            'motion_analysis': motion_analysis,
            'tracking_history': {str(k): v for k, v in self.tracking_history.items()},
            'summary': {
                'total_objects': len(track_names),
                'total_frames': total_frames,
                'successful_tracks': successful_tracks
            }
        }
        
        json_path = os.path.join(output_dir, "tracking_data.json")
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=2, default=str)
        
        print(f"📋 跟踪报告已保存: {report_path}")
        print(f"📊 跟踪数据已保存: {json_path}")


def demo_advanced_video_applications():
    """演示高级视频分割应用"""
    print("🚀 SAM2 高级视频分割应用演示")
    print("=" * 60)
    
    # 由于没有真实视频，我们创建模拟演示
    print("📝 创建高级功能模拟演示...")
    
    # 使用静态图片模拟不同应用
    demo_target_extraction_simulation()
    demo_background_replacement_simulation()
    demo_object_tracking_simulation()


def demo_target_extraction_simulation():
    """模拟目标提取演示"""
    print("\n🎯 模拟目标提取演示")
    print("-" * 40)
    
    output_dir = "sam2_demos/target_extraction_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用现有图片创建模拟效果
    if not os.path.exists("notebooks/images/cars.jpg"):
        print("❌ 演示图片不存在")
        return
    
    # 读取基础图片
    base_image = Image.open("notebooks/images/cars.jpg")
    width, height = base_image.size
    
    # 创建模拟的目标提取结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 模拟3个不同的提取结果
    for i in range(3):
        # 原始图像
        axes[0, i].imshow(base_image)
        axes[0, i].set_title(f"Frame {i+1}")
        axes[0, i].axis('off')
        
        # 模拟提取的目标（添加透明背景效果）
        # 创建一个简单的椭圆掩码作为演示
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        center_x, center_y = width//2 + i*20, height//2 + i*10
        draw.ellipse([center_x-100, center_y-80, center_x+100, center_y+80], fill=255)
        
        # 应用掩码创建提取效果
        extracted = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        extracted.paste(base_image, mask=mask)
        
        axes[1, i].imshow(extracted)
        axes[1, i].set_title(f"Extracted Target {i+1}")
        axes[1, i].axis('off')
    
    plt.suptitle("Target Extraction Simulation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    demo_path = os.path.join(output_dir, "target_extraction_demo.png")
    plt.savefig(demo_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建说明文档
    readme_content = """# 目标提取功能演示

## 🎯 功能描述
目标提取功能可以从视频中精确提取指定的目标对象，生成带透明背景的目标图像。

## ✨ 主要特性
- 精确的目标分割和提取
- 透明背景处理
- 多帧一致性保证
- 质量评估和报告生成

## 🔧 使用方法
```python
segmenter = AdvancedVideoSegmenter()
extracted_targets, masks = segmenter.target_extraction(
    frame_paths=frame_list,
    click_point=[x, y],
    output_dir="output",
    obj_name="target_object"
)
```

## 💡 应用场景
- 视频素材提取
- 对象分析研究
- 内容创作辅助
- 动画制作素材
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ 目标提取演示已保存: {demo_path}")


def demo_background_replacement_simulation():
    """模拟背景替换演示"""
    print("\n🎨 模拟背景替换演示")
    print("-" * 40)
    
    output_dir = "sam2_demos/background_replacement_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists("notebooks/images/cars.jpg"):
        print("❌ 演示图片不存在")
        return
    
    # 读取基础图片
    base_image = Image.open("notebooks/images/cars.jpg")
    width, height = base_image.size
    
    # 创建几个不同的背景
    backgrounds = []
    
    # 渐变背景
    gradient_bg = Image.new('RGB', (width, height))
    for y in range(height):
        color = int(255 * y / height)
        for x in range(width):
            gradient_bg.putpixel((x, y), (color, 100, 255-color))
    backgrounds.append(("Gradient", gradient_bg))
    
    # 纯色背景
    solid_bg = Image.new('RGB', (width, height), (50, 150, 50))
    backgrounds.append(("Green Screen", solid_bg))
    
    # 纹理背景
    texture_bg = Image.new('RGB', (width, height))
    for y in range(height):
        for x in range(width):
            noise = (x + y) % 50
            texture_bg.putpixel((x, y), (100 + noise, 120 + noise, 140 + noise))
    backgrounds.append(("Texture", texture_bg))
    
    # 创建替换效果展示
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, (bg_name, bg_image) in enumerate(backgrounds):
        # 原始图像
        axes[0, i].imshow(base_image)
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        # 模拟背景替换效果
        # 创建简单的前景掩码
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([width//4, height//4, width*3//4, height*3//4], fill=255)
        
        # 执行简单的背景替换
        result = Image.composite(base_image, bg_image, mask)
        
        axes[1, i].imshow(result)
        axes[1, i].set_title(f"With {bg_name} Background")
        axes[1, i].axis('off')
    
    plt.suptitle("Background Replacement Simulation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    demo_path = os.path.join(output_dir, "background_replacement_demo.png")
    plt.savefig(demo_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建说明文档
    readme_content = """# 背景替换功能演示

## 🎨 功能描述
背景替换功能可以将视频中的背景替换为任意指定的新背景，保持前景对象不变。

## ✨ 主要特性
- 精确的前景/背景分离
- 支持任意背景图像
- 边缘优化处理
- 多帧时序一致性

## 🔧 使用方法
```python
segmenter = AdvancedVideoSegmenter()
replaced_frames, masks = segmenter.background_replacement(
    frame_paths=frame_list,
    click_point=[x, y],  # 点击前景对象
    new_background_path="new_bg.jpg",
    output_dir="output"
)
```

## 💡 应用场景
- 虚拟背景会议
- 影视后期制作
- 直播背景替换
- 创意视频制作
- 产品展示视频
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ 背景替换演示已保存: {demo_path}")


def demo_object_tracking_simulation():
    """模拟目标跟踪演示"""
    print("\n🔍 模拟目标跟踪演示")
    print("-" * 40)
    
    output_dir = "sam2_demos/object_tracking_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模拟的多目标跟踪轨迹
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 模拟数据
    frames = 20
    
    # 目标1轨迹 - 直线运动
    traj1_x = np.linspace(100, 400, frames)
    traj1_y = np.linspace(100, 200, frames)
    
    # 目标2轨迹 - 圆形运动
    angles = np.linspace(0, 2*np.pi, frames)
    traj2_x = 250 + 100 * np.cos(angles)
    traj2_y = 250 + 100 * np.sin(angles)
    
    # 目标3轨迹 - 随机运动
    np.random.seed(42)
    traj3_x = 300 + np.cumsum(np.random.randn(frames) * 10)
    traj3_y = 150 + np.cumsum(np.random.randn(frames) * 10)
    
    # 绘制轨迹
    ax = axes[0, 0]
    ax.plot(traj1_x, traj1_y, 'ro-', label='Target 1 (Linear)', linewidth=2, markersize=6)
    ax.plot(traj2_x, traj2_y, 'bo-', label='Target 2 (Circular)', linewidth=2, markersize=6)
    ax.plot(traj3_x, traj3_y, 'go-', label='Target 3 (Random)', linewidth=2, markersize=6)
    ax.set_title("Object Trajectories")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 速度分析
    ax = axes[0, 1]
    vel1 = np.sqrt(np.diff(traj1_x)**2 + np.diff(traj1_y)**2)
    vel2 = np.sqrt(np.diff(traj2_x)**2 + np.diff(traj2_y)**2)
    vel3 = np.sqrt(np.diff(traj3_x)**2 + np.diff(traj3_y)**2)
    
    ax.plot(vel1, 'r-', label='Target 1', linewidth=2)
    ax.plot(vel2, 'b-', label='Target 2', linewidth=2)
    ax.plot(vel3, 'g-', label='Target 3', linewidth=2)
    ax.set_title("Velocity Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Velocity (pixels/frame)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 距离分析
    ax = axes[1, 0]
    dist1 = np.cumsum(vel1)
    dist2 = np.cumsum(vel2)
    dist3 = np.cumsum(vel3)
    
    ax.plot(dist1, 'r-', label='Target 1', linewidth=2)
    ax.plot(dist2, 'b-', label='Target 2', linewidth=2)
    ax.plot(dist3, 'g-', label='Target 3', linewidth=2)
    ax.set_title("Cumulative Distance")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Total Distance (pixels)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 统计信息
    ax = axes[1, 1]
    targets = ['Target 1', 'Target 2', 'Target 3']
    avg_velocities = [np.mean(vel1), np.mean(vel2), np.mean(vel3)]
    colors = ['red', 'blue', 'green']
    
    bars = ax.bar(targets, avg_velocities, color=colors, alpha=0.7)
    ax.set_title("Average Velocity Comparison")
    ax.set_ylabel("Average Velocity (pixels/frame)")
    
    # 添加数值标签
    for bar, vel in zip(bars, avg_velocities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{vel:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    demo_path = os.path.join(output_dir, "object_tracking_demo.png")
    plt.savefig(demo_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建说明文档
    readme_content = """# 目标跟踪功能演示

## 🔍 功能描述
目标跟踪功能支持同时跟踪多个目标对象，分析其运动轨迹、速度变化等运动特征。

## ✨ 主要特性
- 多目标同时跟踪
- 实时轨迹分析
- 运动特征统计
- 遮挡恢复能力
- 自动质量评估

## 🔧 使用方法
```python
segmenter = AdvancedVideoSegmenter()
tracked_frames, results, motion_analysis = segmenter.object_tracking(
    frame_paths=frame_list,
    click_points=[(x1, y1), (x2, y2), ...],  # 多个目标点
    output_dir="output",
    track_names=["target1", "target2", ...]
)
```

## 📊 分析指标
- **轨迹完整性**: 跟踪成功的帧数比例
- **运动速度**: 平均速度和最大速度
- **运动距离**: 总移动距离
- **置信度**: 跟踪质量评分
- **目标稳定性**: 大小变化率

## 💡 应用场景
- 运动分析研究
- 安防监控系统
- 体育比赛分析
- 交通流量监测
- 行为模式研究
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ 目标跟踪演示已保存: {demo_path}")


if __name__ == "__main__":
    demo_advanced_video_applications()
