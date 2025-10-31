#!/usr/bin/env python3
"""
SAM2 真实视频处理应用 - 简化版
专注于核心功能，避免复杂依赖
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import json
from datetime import datetime

class SimpleVideoProcessor:
    def __init__(self):
        """初始化简单视频处理器"""
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        print("🎥 SAM2 简化视频处理器已初始化")
    
    def create_demo_video(self, output_path="sam2_demos/demo_video.mp4", duration=3, fps=5):
        """创建一个简单的演示视频"""
        print(f"🎬 创建演示视频: {output_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 视频参数
        width, height = 320, 240
        total_frames = duration * fps
        
        try:
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 生成视频帧
            for frame_idx in range(total_frames):
                # 创建背景
                frame = np.ones((height, width, 3), dtype=np.uint8) * 64
                
                # 添加移动的圆形目标
                center_x = int(50 + (frame_idx / total_frames) * (width - 100))
                center_y = int(height // 2 + 30 * np.sin(frame_idx * 0.5))
                cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), -1)
                
                # 添加静态的方形目标
                rect_x, rect_y = width - 60, 60
                cv2.rectangle(frame, (rect_x - 15, rect_y - 15), 
                             (rect_x + 15, rect_y + 15), (255, 0, 0), -1)
                
                # 添加帧号文本
                cv2.putText(frame, f"F{frame_idx+1:02d}", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                out.write(frame)
            
            out.release()
            print(f"✅ 演示视频已创建: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 创建视频失败: {e}")
            return None
    
    def extract_video_frames(self, video_path, output_dir, max_frames=10):
        """从视频中提取帧"""
        print(f"📹 从视频提取帧: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"📊 视频信息: {total_frames} 帧, {fps:.1f} FPS")
            
            frame_paths = []
            frame_interval = max(1, total_frames // max_frames)
            
            frame_idx = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    frame_path = os.path.join(output_dir, f"frame_{extracted_count:03d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    extracted_count += 1
                
                frame_idx += 1
            
            cap.release()
            print(f"✅ 成功提取 {len(frame_paths)} 帧")
            return frame_paths
            
        except Exception as e:
            print(f"❌ 提取帧失败: {e}")
            return []
    
    def analyze_video_motion(self, frame_paths):
        """分析视频中的运动"""
        print("🔍 分析视频运动...")
        
        if len(frame_paths) < 2:
            print("❌ 需要至少2帧进行运动分析")
            return {}
        
        try:
            motion_data = {
                "total_frames": len(frame_paths),
                "motion_detected": [],
                "object_positions": []
            }
            
            prev_frame = None
            
            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(frame_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # 计算帧差
                    diff = cv2.absdiff(prev_frame, gray)
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    
                    # 检测运动
                    motion_pixels = np.sum(thresh > 0)
                    motion_ratio = motion_pixels / (thresh.shape[0] * thresh.shape[1])
                    
                    motion_data["motion_detected"].append({
                        "frame": i,
                        "motion_ratio": float(motion_ratio),
                        "motion_pixels": int(motion_pixels)
                    })
                
                # 简单的绿色对象检测 (演示用)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                green_mask = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))
                
                # 找到绿色对象的质心
                moments = cv2.moments(green_mask)
                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    motion_data["object_positions"].append({
                        "frame": i,
                        "x": cx,
                        "y": cy
                    })
                
                prev_frame = gray
            
            print(f"✅ 运动分析完成: {len(motion_data['motion_detected'])} 帧运动数据")
            return motion_data
            
        except Exception as e:
            print(f"❌ 运动分析失败: {e}")
            return {}
    
    def create_motion_visualization(self, motion_data, output_path):
        """创建运动分析可视化"""
        print(f"📊 创建运动可视化: {output_path}")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 运动强度图
            if motion_data.get("motion_detected"):
                frames = [d["frame"] for d in motion_data["motion_detected"]]
                ratios = [d["motion_ratio"] for d in motion_data["motion_detected"]]
                
                ax1.plot(frames, ratios, 'b-o', markersize=4)
                ax1.set_title("视频运动强度分析", fontsize=14)
                ax1.set_xlabel("帧号")
                ax1.set_ylabel("运动比例")
                ax1.grid(True, alpha=0.3)
            
            # 对象轨迹图
            if motion_data.get("object_positions"):
                x_pos = [p["x"] for p in motion_data["object_positions"]]
                y_pos = [p["y"] for p in motion_data["object_positions"]]
                frames = [p["frame"] for p in motion_data["object_positions"]]
                
                scatter = ax2.scatter(x_pos, y_pos, c=frames, cmap='viridis', s=50)
                ax2.plot(x_pos, y_pos, 'r-', alpha=0.5, linewidth=2)
                ax2.set_title("绿色目标运动轨迹", fontsize=14)
                ax2.set_xlabel("X 坐标")
                ax2.set_ylabel("Y 坐标")
                plt.colorbar(scatter, ax=ax2, label="帧号")
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 运动可视化已保存: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 创建可视化失败: {e}")
            return False
    
    def demo_simple_video_processing(self):
        """演示简单视频处理功能"""
        print("🚀 简单视频处理演示")
        print("=" * 50)
        
        try:
            # 1. 创建演示视频
            video_path = self.create_demo_video()
            if not video_path:
                print("❌ 无法创建演示视频")
                return
            
            # 2. 提取视频帧
            frames_dir = "sam2_demos/simple_video_frames"
            frame_paths = self.extract_video_frames(video_path, frames_dir, max_frames=8)
            
            if not frame_paths:
                print("❌ 无法提取视频帧")
                return
            
            # 3. 分析运动
            motion_data = self.analyze_video_motion(frame_paths)
            
            # 4. 创建可视化
            viz_path = "sam2_demos/simple_video_motion_analysis.png"
            self.create_motion_visualization(motion_data, viz_path)
            
            # 5. 保存分析报告
            self.create_analysis_report(video_path, frame_paths, motion_data)
            
            print("\n✅ 简单视频处理演示完成!")
            print("📁 查看结果:")
            print(f"  - 演示视频: {video_path}")
            print(f"  - 提取帧: {frames_dir}")
            print(f"  - 运动分析: {viz_path}")
            print(f"  - 分析报告: sam2_demos/simple_video_analysis_report.md")
            
        except Exception as e:
            print(f"❌ 演示失败: {e}")
            import traceback
            traceback.print_exc()
    
    def create_analysis_report(self, video_path, frame_paths, motion_data):
        """创建分析报告"""
        report_content = f"""# SAM2 简单视频处理分析报告

## 📋 基本信息
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **源视频**: {os.path.basename(video_path)}
- **提取帧数**: {len(frame_paths)}

## 🎥 视频分析结果

### 运动检测
- **检测帧数**: {len(motion_data.get('motion_detected', []))}
- **平均运动强度**: {np.mean([d['motion_ratio'] for d in motion_data.get('motion_detected', [])]) if motion_data.get('motion_detected') else 0:.3f}

### 目标跟踪
- **跟踪帧数**: {len(motion_data.get('object_positions', []))}
- **轨迹完整性**: {len(motion_data.get('object_positions', []))/len(frame_paths)*100:.1f}%

## 🔧 技术特点
- **轻量级处理**: 使用OpenCV进行基础视频分析
- **运动检测**: 基于帧差的运动检测算法
- **目标跟踪**: 基于颜色的简单目标检测
- **可视化分析**: matplotlib生成分析图表

## 💡 应用场景
- **监控分析**: 基础运动检测和目标跟踪
- **视频预处理**: 为SAM2分割做准备
- **教学演示**: 计算机视觉基础概念展示
- **快速原型**: 视频分析应用原型开发

## 🚀 扩展方向
- **集成SAM2**: 添加精确的实例分割
- **多目标跟踪**: 支持多个目标同时跟踪
- **背景建模**: 更复杂的背景分离技术
- **特征跟踪**: 基于特征点的目标跟踪

这个简化版本为后续集成SAM2高级功能奠定了基础！
"""
        
        report_path = "sam2_demos/simple_video_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📋 分析报告已保存: {report_path}")


def demo_simple_video_applications():
    """运行简单视频应用演示"""
    try:
        processor = SimpleVideoProcessor()
        processor.demo_simple_video_processing()
    except Exception as e:
        print(f"❌ 演示失败: {e}")


if __name__ == "__main__":
    demo_simple_video_applications()
