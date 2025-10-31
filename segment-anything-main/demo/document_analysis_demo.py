#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
SAM文档图像分析演示 - 文档内容分割与版面分析
应用场景：文档数字化、版面分析、文字识别前处理、档案管理
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from common_tools import show_mask, show_box, show_points, show_anns
import json
import os
from collections import defaultdict

class DocumentAnalysisSystem:
    def __init__(self, model_path="models/sam_vit_h_4b8939.pth", device="cpu"):
        self.sam_checkpoint = model_path
        self.model_type = "vit_h"
        self.device = device
        self.sam = None
        self.predictor = None
        self.mask_generator = None
        self.load_model()
        
        # 文档元素类型定义
        self.element_types = {
            'text_block': {'min_area': 500, 'aspect_ratio_range': (0.1, 10)},
            'image': {'min_area': 2000, 'aspect_ratio_range': (0.5, 2.0)},
            'table': {'min_area': 1000, 'aspect_ratio_range': (1.2, 4.0)},
            'header': {'min_area': 200, 'aspect_ratio_range': (2.0, 20.0)},
            'figure': {'min_area': 1500, 'aspect_ratio_range': (0.7, 1.5)}
        }
    
    def load_model(self):
        """加载SAM模型"""
        print("正在加载文档分析模型...")
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        # 文档专用掩码生成器配置
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=24,  # 文档需要细致分割
            pred_iou_thresh=0.92,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # 捕获小文本块
        )
        print("文档分析模型加载完成！")
    
    def analyze_document_layout(self, image_path):
        """分析文档版面布局"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print("正在分析文档版面布局...")
        masks = self.mask_generator.generate(image_rgb)
        
        # 分类文档元素
        document_elements = []
        for mask_data in masks:
            element = self.classify_document_element(image_rgb, mask_data)
            if element:
                document_elements.append(element)
        
        # 按位置排序（从上到下，从左到右）
        document_elements.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        print(f"检测到 {len(document_elements)} 个文档元素")
        return image_rgb, document_elements
    
    def classify_document_element(self, image, mask_data):
        """分类文档元素"""
        bbox = mask_data['bbox']
        area = mask_data['area']
        
        # 计算长宽比
        width = bbox[2]
        height = bbox[3]
        aspect_ratio = width / height if height > 0 else 0
        
        # 提取区域图像用于进一步分析
        x, y, w, h = bbox
        region = image[y:y+h, x:x+w]
        
        # 颜色分析
        avg_color = np.mean(region, axis=(0, 1))
        color_variance = np.var(region)
        
        # 边缘密度分析（用于区分文本和图像）
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_region, 50, 150)
        edge_density = np.sum(edges) / (w * h) if w * h > 0 else 0
        
        # 基于规则的分类
        element_type = "unknown"
        confidence = 0.5
        
        # 文本块：高边缘密度，适中的长宽比
        if edge_density > 10 and 0.1 <= aspect_ratio <= 10 and area >= 500:
            element_type = "text_block"
            confidence = 0.8
        
        # 标题：长条形，较高边缘密度
        elif aspect_ratio > 2.0 and edge_density > 8 and area >= 200:
            element_type = "header"
            confidence = 0.7
        
        # 图像：低边缘密度，方形或矩形
        elif edge_density < 5 and 0.5 <= aspect_ratio <= 2.0 and area >= 2000:
            element_type = "image"
            confidence = 0.6
        
        # 表格：规则形状，中等边缘密度
        elif 1.2 <= aspect_ratio <= 4.0 and 5 <= edge_density <= 15 and area >= 1000:
            element_type = "table"
            confidence = 0.7
        
        # 图表：接近正方形，中等面积
        elif 0.7 <= aspect_ratio <= 1.5 and area >= 1500:
            element_type = "figure"
            confidence = 0.6
        
        return {
            'type': element_type,
            'bbox': bbox,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'avg_color': avg_color.tolist(),
            'edge_density': edge_density,
            'confidence': confidence,
            'mask': mask_data['segmentation'],
            'stability_score': mask_data['stability_score']
        }
    
    def extract_reading_order(self, elements):
        """提取文档阅读顺序"""
        # 简单的阅读顺序：从上到下，从左到右
        reading_order = []
        
        # 按Y坐标分组（同一行的元素）
        tolerance = 20  # 像素容差
        rows = []
        current_row = []
        current_y = None
        
        for element in sorted(elements, key=lambda x: x['bbox'][1]):
            y = element['bbox'][1]
            
            if current_y is None or abs(y - current_y) <= tolerance:
                current_row.append(element)
                current_y = y if current_y is None else (current_y + y) / 2
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda x: x['bbox'][0]))
                current_row = [element]
                current_y = y
        
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x['bbox'][0]))
        
        # 展平为阅读顺序
        for row in rows:
            reading_order.extend(row)
        
        return reading_order
    
    def segment_specific_elements(self, image_path, element_points, element_types):
        """分割特定的文档元素"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image_rgb)
        
        segmented_elements = []
        
        for i, (point, elem_type) in enumerate(zip(element_points, element_types)):
            print(f"分割 {elem_type} 元素 {i+1}/{len(element_points)}")
            
            input_point = np.array([point])
            input_label = np.array([1])
            
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # 提取元素属性
            element_analysis = self.analyze_element_properties(image_rgb, best_mask, elem_type)
            element_analysis.update({
                'element_id': i + 1,
                'specified_type': elem_type,
                'point': point,
                'mask': best_mask,
                'score': float(scores[best_mask_idx])
            })
            
            segmented_elements.append(element_analysis)
        
        return image_rgb, segmented_elements
    
    def analyze_element_properties(self, image, mask, element_type):
        """分析文档元素属性"""
        # 计算边界框
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return {'error': 'empty_mask'}
        
        bbox = [int(np.min(x_indices)), int(np.min(y_indices)),
               int(np.max(x_indices) - np.min(x_indices)),
               int(np.max(y_indices) - np.min(y_indices))]
        
        # 提取区域
        x, y, w, h = bbox
        region = image[y:y+h, x:x+w] if y+h <= image.shape[0] and x+w <= image.shape[1] else image
        
        # 基本属性
        area = int(np.sum(mask))
        aspect_ratio = w / h if h > 0 else 0
        
        # 颜色分析
        avg_color = np.mean(region, axis=(0, 1)) if region.size > 0 else [0, 0, 0]
        
        # 文本密度估计（基于边缘）
        if region.size > 0:
            gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_region, 50, 150)
            text_density = np.sum(edges) / (w * h) if w * h > 0 else 0
        else:
            text_density = 0
        
        # 元素类型特定分析
        specific_analysis = {}
        if element_type == "text_block":
            specific_analysis = self.analyze_text_properties(region)
        elif element_type == "table":
            specific_analysis = self.analyze_table_properties(region)
        elif element_type == "image":
            specific_analysis = self.analyze_image_properties(region)
        
        return {
            'bbox': bbox,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'avg_color': avg_color.tolist(),
            'text_density': text_density,
            'specific_analysis': specific_analysis
        }
    
    def analyze_text_properties(self, region):
        """分析文本属性"""
        if region.size == 0:
            return {}
        
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        # 文本行数估计（基于水平投影）
        horizontal_projection = np.sum(gray < 128, axis=1)
        text_lines = len([i for i, proj in enumerate(horizontal_projection) if proj > len(gray[0]) * 0.1])
        
        # 字符密度估计
        char_pixels = np.sum(gray < 128)
        char_density = char_pixels / gray.size if gray.size > 0 else 0
        
        return {
            'estimated_lines': text_lines,
            'character_density': float(char_density),
            'text_coverage': float(char_pixels / gray.size) if gray.size > 0 else 0
        }
    
    def analyze_table_properties(self, region):
        """分析表格属性"""
        if region.size == 0:
            return {}
        
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        # 检测水平和垂直线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        h_line_pixels = np.sum(horizontal_lines > 0)
        v_line_pixels = np.sum(vertical_lines > 0)
        
        return {
            'horizontal_lines_density': float(h_line_pixels / gray.size) if gray.size > 0 else 0,
            'vertical_lines_density': float(v_line_pixels / gray.size) if gray.size > 0 else 0,
            'estimated_table': bool(h_line_pixels > 0 and v_line_pixels > 0)
        }
    
    def analyze_image_properties(self, region):
        """分析图像属性"""
        if region.size == 0:
            return {}
        
        # 颜色多样性
        colors = region.reshape(-1, 3)
        unique_colors = len(np.unique(colors, axis=0))
        color_diversity = unique_colors / len(colors) if len(colors) > 0 else 0
        
        # 纹理复杂度
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        texture_complexity = np.std(gray)
        
        return {
            'color_diversity': float(color_diversity),
            'texture_complexity': float(texture_complexity),
            'is_likely_photo': bool(color_diversity > 0.1 and texture_complexity > 30)
        }
    
    def create_document_report(self, elements, reading_order, output_dir):
        """创建文档分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        element_stats = defaultdict(int)
        for element in elements:
            element_stats[element['type']] += 1
        
        # 报告数据
        report = {
            'total_elements': len(elements),
            'element_statistics': dict(element_stats),
            'reading_order': [],
            'layout_analysis': {
                'document_structure': self.analyze_document_structure(reading_order),
                'element_distribution': self.analyze_element_distribution(elements)
            }
        }
        
        # 阅读顺序
        for i, element in enumerate(reading_order):
            report['reading_order'].append({
                'order': i + 1,
                'type': element['type'],
                'bbox': element['bbox'],
                'confidence': element['confidence']
            })
        
        # 保存JSON报告
        report_file = os.path.join(output_dir, 'document_analysis_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 创建可视化报告
        self.create_layout_visualization(elements, reading_order, output_dir)
        
        print(f"文档分析报告已保存到: {output_dir}")
        return report
    
    def analyze_document_structure(self, reading_order):
        """分析文档结构"""
        if not reading_order:
            return {}
        
        # 简单的结构分析
        headers = [e for e in reading_order if e['type'] == 'header']
        text_blocks = [e for e in reading_order if e['type'] == 'text_block']
        tables = [e for e in reading_order if e['type'] == 'table']
        images = [e for e in reading_order if e['type'] == 'image']
        
        return {
            'has_header': len(headers) > 0,
            'main_content_blocks': len(text_blocks),
            'supplementary_tables': len(tables),
            'illustrations': len(images),
            'estimated_document_type': self.estimate_document_type(reading_order)
        }
    
    def estimate_document_type(self, reading_order):
        """估计文档类型"""
        if not reading_order:
            return "unknown"
        
        headers = len([e for e in reading_order if e['type'] == 'header'])
        text_blocks = len([e for e in reading_order if e['type'] == 'text_block'])
        tables = len([e for e in reading_order if e['type'] == 'table'])
        images = len([e for e in reading_order if e['type'] == 'image'])
        
        if tables > text_blocks:
            return "report_with_data"
        elif headers > 0 and text_blocks > 3:
            return "structured_document"
        elif images > 0 and text_blocks > 0:
            return "illustrated_document"
        elif text_blocks > 0:
            return "text_document"
        else:
            return "unknown"
    
    def analyze_element_distribution(self, elements):
        """分析元素分布"""
        if not elements:
            return {}
        
        # 计算元素密度
        total_area = sum(e['area'] for e in elements)
        avg_area = total_area / len(elements)
        
        # 位置分布
        y_positions = [e['bbox'][1] for e in elements]
        top_third = sum(1 for y in y_positions if y < max(y_positions) / 3)
        middle_third = sum(1 for y in y_positions if max(y_positions) / 3 <= y < 2 * max(y_positions) / 3)
        bottom_third = len(elements) - top_third - middle_third
        
        return {
            'average_element_area': avg_area,
            'position_distribution': {
                'top_third': top_third,
                'middle_third': middle_third,
                'bottom_third': bottom_third
            }
        }
    
    def create_layout_visualization(self, elements, reading_order, output_dir):
        """创建版面可视化"""
        if not elements:
            return
        
        # 创建布局图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 左图：元素类型分布
        element_counts = defaultdict(int)
        for element in elements:
            element_counts[element['type']] += 1
        
        if element_counts:
            ax1.pie(element_counts.values(), labels=element_counts.keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax1.set_title('文档元素类型分布')
        
        # 右图：阅读顺序可视化（模拟）
        type_colors = {
            'header': 'red', 'text_block': 'blue', 'table': 'green',
            'image': 'orange', 'figure': 'purple', 'unknown': 'gray'
        }
        
        order_numbers = list(range(1, len(reading_order) + 1))
        type_counts_ordered = defaultdict(list)
        
        for i, element in enumerate(reading_order):
            type_counts_ordered[element['type']].append(i + 1)
        
        bottom = 0
        for elem_type, positions in type_counts_ordered.items():
            ax2.bar(positions, [1] * len(positions), bottom=bottom, 
                   label=elem_type, color=type_colors.get(elem_type, 'gray'), alpha=0.7)
        
        ax2.set_xlabel('阅读顺序')
        ax2.set_ylabel('元素')
        ax2.set_title('文档阅读顺序')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layout_visualization.png'), dpi=150, bbox_inches='tight')
        plt.show()

def demo_document_layout_analysis():
    """演示文档版面分析"""
    system = DocumentAnalysisSystem()
    
    print("\n=== 文档版面分析演示 ===")
    
    # 使用groceries.jpg作为演示（模拟文档图像）
    image_path = 'images/groceries.jpg'
    if not os.path.exists(image_path):
        print(f"演示图像 {image_path} 不存在")
        return
    
    image, elements = system.analyze_document_layout(image_path)
    reading_order = system.extract_reading_order(elements)
    
    # 可视化分析结果
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    
    # 显示检测到的元素
    type_colors = {
        'header': 'red', 'text_block': 'blue', 'table': 'green',
        'image': 'orange', 'figure': 'purple', 'unknown': 'gray'
    }
    
    for i, element in enumerate(elements[:15]):  # 只显示前15个
        mask = element['mask']
        elem_type = element['type']
        confidence = element['confidence']
        
        # 显示掩码
        show_mask(mask, plt.gca(), random_color=False)
        
        # 添加标签
        bbox = element['bbox']
        center_x = bbox[0] + bbox[2] // 2
        center_y = bbox[1] + bbox[3] // 2
        
        plt.text(center_x, center_y, f"{elem_type}\n{confidence:.2f}", 
                fontsize=8, color='white', weight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", 
                         facecolor=type_colors.get(elem_type, 'gray'), alpha=0.8))
    
    plt.title(f"文档版面分析结果 - 共检测到 {len(elements)} 个文档元素")
    plt.axis('off')
    plt.savefig('document_layout_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 输出分析统计
    element_stats = defaultdict(int)
    for element in elements:
        element_stats[element['type']] += 1
    
    print(f"\n📄 文档元素统计:")
    for elem_type, count in element_stats.items():
        print(f"  {elem_type}: {count} 个")
    
    print(f"\n📖 阅读顺序（前10个）:")
    for i, element in enumerate(reading_order[:10], 1):
        print(f"  {i}. {element['type']} (置信度: {element['confidence']:.2f})")

def demo_specific_element_extraction():
    """演示特定元素提取"""
    system = DocumentAnalysisSystem()
    
    print("\n=== 特定文档元素提取演示 ===")
    
    # 定义要提取的元素点和类型
    element_points = [
        [300, 200],  # 标题位置
        [200, 350],  # 文本块位置
        [400, 500],  # 可能的表格位置
    ]
    
    element_types = ["header", "text_block", "table"]
    
    image_path = 'images/truck.jpg'  # 使用truck.jpg作为演示
    if not os.path.exists(image_path):
        print(f"演示图像 {image_path} 不存在")
        return
    
    image, segmented_elements = system.segment_specific_elements(
        image_path, element_points, element_types
    )
    
    # 可视化提取结果
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    
    for element in segmented_elements:
        mask = element['mask']
        point = element['point']
        elem_type = element['specified_type']
        score = element['score']
        
        # 显示掩码
        show_mask(mask, plt.gca(), random_color=False)
        
        # 显示点击点
        show_points(np.array([point]), np.array([1]), plt.gca())
        
        # 添加标签
        plt.text(point[0], point[1] - 30, f"{elem_type}\nScore: {score:.3f}", 
                fontsize=10, color='white', weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.8))
    
    plt.title(f"特定文档元素提取结果")
    plt.axis('off')
    plt.savefig('element_extraction_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 输出提取分析
    print(f"\n📑 元素提取分析:")
    for i, element in enumerate(segmented_elements, 1):
        print(f"  元素 {i}: {element['specified_type']}")
        print(f"    分割得分: {element['score']:.3f}")
        print(f"    区域面积: {element['area']} 像素")
        if 'specific_analysis' in element:
            spec_analysis = element['specific_analysis']
            if spec_analysis:
                for key, value in spec_analysis.items():
                    print(f"    {key}: {value}")

if __name__ == '__main__':
    print("=== SAM文档图像分析演示 ===")
    print("注意：此演示使用普通图像模拟文档，实际应用需要扫描文档或PDF图像")
    
    try:
        # 演示文档版面分析
        demo_document_layout_analysis()
        
        # 演示特定元素提取
        demo_specific_element_extraction()
        
        print("\n✅ 文档分析演示完成！")
        print("💡 在实际文档分析应用中，建议:")
        print("   1. 使用高质量的扫描文档图像")
        print("   2. 集成OCR技术进行文字识别")
        print("   3. 建立文档类型分类模型")
        print("   4. 添加表格结构识别功能")
        print("   5. 支持多页文档的批量处理")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确保模型文件和图像文件存在")
