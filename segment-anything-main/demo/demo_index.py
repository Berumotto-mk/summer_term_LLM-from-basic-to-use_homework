#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
SAM Demo快速索引和推荐系统
根据用户需求推荐合适的demo文件
"""

def print_demo_index():
    """打印demo文件索引"""
    print("=" * 80)
    print("                        SAM Demo 快速索引")
    print("=" * 80)
    print()
    
    # 按应用场景分类
    scenarios = {
        "🔰 基础学习": [
            ("fast_sam_demo.py", "快速入门，了解SAM基本用法", "51行"),
            ("predictor_one_point.py", "学习点提示分割的完整流程", "49行"),
            ("test_sam_simple.py", "性能测试和系统诊断", "55行")
        ],
        
        "🎯 精确分割": [
            ("predictor_bbox.py", "使用边界框进行精确分割", "42行"),
            ("predictor_bbox_point.py", "结合框和点的混合分割", "33行"),
            ("predictor_multimask_param.py", "参数化控制分割质量", "35行")
        ],
        
        "🚀 批量处理": [
            ("predictor_multimask.py", "自动多目标检测", "24行"),
            ("preditor_batch_prompts.py", "批量处理多图像", "68行")
        ],
        
        "🏥 医疗应用": [
            ("medical_segmentation_demo.py", "医学图像分割与分析", "194行")
        ],
        
        "📹 视频分析": [
            ("video_tracking_demo.py", "视频对象跟踪与运动分析", "253行")
        ],
        
        "🌾 农业应用": [
            ("agricultural_demo.py", "作物检测与病害识别", "314行")
        ],
        
        "📄 文档处理": [
            ("document_analysis_demo.py", "文档版面分析与内容提取", "417行")
        ],
        
        "🏷️ 数据标注": [
            ("interactive_annotation_demo.py", "交互式标注与质量控制", "380行")
        ]
    }
    
    for category, demos in scenarios.items():
        print(f"{category}")
        for filename, description, lines in demos:
            print(f"  📁 {filename:<35} {description:<25} ({lines})")
        print()

def recommend_demo(user_need):
    """根据用户需求推荐demo"""
    recommendations = {
        "学习": ["fast_sam_demo.py", "predictor_one_point.py", "test_sam_simple.py"],
        "入门": ["fast_sam_demo.py", "predictor_bbox.py"],
        "精确": ["predictor_bbox.py", "predictor_bbox_point.py"],
        "快速": ["fast_sam_demo.py", "predictor_multimask.py"],
        "批量": ["preditor_batch_prompts.py", "predictor_multimask.py"],
        "医疗": ["medical_segmentation_demo.py"],
        "医学": ["medical_segmentation_demo.py"],
        "视频": ["video_tracking_demo.py"],
        "跟踪": ["video_tracking_demo.py"],
        "农业": ["agricultural_demo.py"],
        "作物": ["agricultural_demo.py"],
        "文档": ["document_analysis_demo.py"],
        "标注": ["interactive_annotation_demo.py"],
        "数据": ["interactive_annotation_demo.py"]
    }
    
    user_need_lower = user_need.lower()
    matches = []
    
    for keyword, demos in recommendations.items():
        if keyword in user_need_lower:
            matches.extend(demos)
    
    # 去重并保持顺序
    unique_matches = []
    for demo in matches:
        if demo not in unique_matches:
            unique_matches.append(demo)
    
    return unique_matches

def print_usage_guide():
    """打印使用指南"""
    print("🚀 使用指南")
    print("-" * 40)
    print()
    
    guides = [
        ("初学者", "建议从 fast_sam_demo.py 开始，然后尝试 predictor_one_point.py"),
        ("开发者", "先运行 test_sam_simple.py 测试环境，再选择相应应用demo"),
        ("研究者", "重点关注新增的5个专业应用demo，每个都有完整的分析流程"),
        ("工程师", "参考 preditor_batch_prompts.py 了解批量处理，然后选择具体应用场景")
    ]
    
    for user_type, guide in guides:
        print(f"👤 {user_type}: {guide}")
    print()

def print_complexity_analysis():
    """打印复杂度分析"""
    print("📊 复杂度分析")
    print("-" * 40)
    print()
    
    complexity_levels = {
        "⭐ 简单 (< 50行)": [
            "predictor_multimask.py (24行)",
            "predictor_bbox_point.py (33行)", 
            "predictor_multimask_param.py (35行)",
            "predictor_bbox.py (42行)",
            "predictor_one_point.py (49行)"
        ],
        "⭐⭐ 中等 (50-100行)": [
            "fast_sam_demo.py (51行)",
            "test_sam_simple.py (55行)",
            "preditor_batch_prompts.py (68行)"
        ],
        "⭐⭐⭐ 复杂 (100-300行)": [
            "medical_segmentation_demo.py (194行)",
            "video_tracking_demo.py (253行)"
        ],
        "⭐⭐⭐⭐ 高级 (300+行)": [
            "agricultural_demo.py (314行)",
            "interactive_annotation_demo.py (380行)",
            "document_analysis_demo.py (417行)"
        ]
    }
    
    for level, demos in complexity_levels.items():
        print(f"{level}")
        for demo in demos:
            print(f"  📄 {demo}")
        print()

def interactive_recommendation():
    """交互式推荐系统"""
    print("🤖 智能推荐系统")
    print("-" * 40)
    print("请描述您的需求（例如：学习分割、医学应用、批量处理等）")
    
    try:
        user_input = input("您的需求: ").strip()
        if user_input:
            recommendations = recommend_demo(user_input)
            if recommendations:
                print(f"\n💡 根据您的需求 '{user_input}'，推荐以下demo:")
                for i, demo in enumerate(recommendations, 1):
                    print(f"  {i}. {demo}")
            else:
                print("\n❓ 未找到匹配的demo，建议从基础demo开始:")
                print("  1. fast_sam_demo.py (快速入门)")
                print("  2. predictor_one_point.py (基础学习)")
        else:
            print("未输入需求，显示所有demo索引。")
    except:
        print("输入错误，显示基础推荐。")

if __name__ == "__main__":
    print_demo_index()
    print_usage_guide()
    print_complexity_analysis()
    interactive_recommendation()
    
    print("\n" + "=" * 80)
    print("💡 提示: 每个demo都包含详细的中文注释，建议先阅读代码再运行")
    print("📚 详细功能介绍请查看: EXTENSION_README.md")
    print("🔢 代码统计信息请运行: python count_lines.py")
    print("📊 项目总结报告请运行: python project_summary.py")
    print("=" * 80)
