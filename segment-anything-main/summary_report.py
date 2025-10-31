#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Demo文件夹代码行数统计结果汇总
"""

print("=" * 80)
print("                    DEMO文件夹代码行数统计报告")
print("                     （不包括注释和空行）")
print("=" * 80)
print()

# 统计结果数据
files_data = [
    ("fast_sam_demo.py", 70, 51, 9, "快速版SAM演示"),
    ("predictor_bbox.py", 74, 42, 17, "使用边界框进行分割"),
    ("predictor_bbox_point.py", 62, 33, 16, "边界框+点提示分割"),
    ("predictor_multimask.py", 28, 24, 2, "多掩码自动生成"),
    ("predictor_multimask_param.py", 41, 35, 4, "参数化多掩码生成"),
    ("predictor_one_point.py", 91, 49, 23, "单点/多点提示分割"),
    ("preditor_batch_prompts.py", 106, 68, 19, "批量处理多图像"),
    ("test_sam_simple.py", 75, 55, 10, "SAM性能测试脚本")
]

print(f"{'序号':<4} {'文件名':<28} {'总行数':<8} {'代码行数':<8} {'注释行数':<8} {'功能描述'}")
print("-" * 80)

total_lines = 0
total_code_lines = 0
total_comment_lines = 0

for i, (filename, total, code, comment, description) in enumerate(files_data, 1):
    print(f"{i:<4} {filename:<28} {total:<8} {code:<8} {comment:<8} {description}")
    total_lines += total
    total_code_lines += code
    total_comment_lines += comment

print("-" * 80)
print(f"{'总计':<4} {'':<28} {total_lines:<8} {total_code_lines:<8} {total_comment_lines:<8}")
print("=" * 80)

print()
print("📊 统计摘要:")
print(f"   • 总文件数: {len(files_data)} 个")
print(f"   • 总代码行数: {total_code_lines} 行")
print(f"   • 平均每个文件代码行数: {total_code_lines/len(files_data):.1f} 行")
print(f"   • 代码行占总行数比例: {total_code_lines/total_lines*100:.1f}%")
print()

# 按代码行数排序
sorted_files = sorted(files_data, key=lambda x: x[2], reverse=True)
print("📈 按代码行数排序（前5名）:")
for i, (filename, total, code, comment, description) in enumerate(sorted_files[:5], 1):
    print(f"   {i}. {filename:<28} {code} 行")

print()
print("✅ 统计完成！")
