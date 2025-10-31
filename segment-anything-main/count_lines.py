#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
统计demo文件夹中各个文件的代码行数（不包括注释）
"""
import os
import re

def count_code_lines(file_path):
    """统计文件中的代码行数，排除注释和空行"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                lines = f.readlines()
        except:
            return 0, 0, 0
    
    total_lines = len(lines)
    code_lines = 0
    comment_lines = 0
    empty_lines = 0
    
    in_multiline_comment = False
    
    for line_num, line in enumerate(lines, 1):
        original_line = line
        stripped = line.strip()
        
        # 空行
        if not stripped:
            empty_lines += 1
            continue
            
        # 检查多行注释开始/结束
        triple_quote_count = stripped.count('"""') + stripped.count("'''")
        
        # 处理多行注释
        if '"""' in stripped or "'''" in stripped:
            if not in_multiline_comment:
                # 检查是否是单行多行注释
                if (stripped.count('"""') >= 2) or (stripped.count("'''") >= 2):
                    comment_lines += 1
                    continue
                else:
                    # 多行注释开始
                    in_multiline_comment = True
                    comment_lines += 1
                    continue
            else:
                # 多行注释结束
                in_multiline_comment = False
                comment_lines += 1
                continue
        
        if in_multiline_comment:
            comment_lines += 1
            continue
            
        # 单行注释（以#开头）
        if stripped.startswith('#'):
            comment_lines += 1
            continue
            
        # 行末注释的处理
        if '#' in stripped:
            # 简单处理：检查#前是否有实际代码
            before_hash = stripped.split('#')[0].strip()
            if before_hash and not before_hash.startswith(('"""', "'''")):
                code_lines += 1
            else:
                comment_lines += 1
            continue
            
        # 普通代码行
        code_lines += 1
    
    return total_lines, code_lines, comment_lines

def main():
    demo_folder = "demo"
    files = [
        "fast_sam_demo.py",
        "predictor_bbox.py", 
        "predictor_bbox_point.py",
        "predictor_multimask.py",
        "predictor_multimask_param.py",
        "predictor_one_point.py",
        "preditor_batch_prompts.py",
        "test_sam_simple.py",
        "medical_segmentation_demo.py",
        "video_tracking_demo.py",
        "agricultural_demo.py",
        "document_analysis_demo.py",
        "interactive_annotation_demo.py"
    ]
    
    total_code_lines = 0
    print("=" * 60)
    print("Demo文件夹代码行数统计（不包括注释）")
    print("=" * 60)
    print(f"{'文件名':<30} {'总行数':<8} {'代码行数':<8} {'注释行数':<8}")
    print("-" * 60)
    
    for file_name in files:
        file_path = os.path.join(demo_folder, file_name)
        if os.path.exists(file_path):
            total_lines, code_lines, comment_lines = count_code_lines(file_path)
            total_code_lines += code_lines
            print(f"{file_name:<30} {total_lines:<8} {code_lines:<8} {comment_lines:<8}")
        else:
            print(f"{file_name:<30} {'文件不存在'}")
    
    print("-" * 60)
    print(f"{'总计':<30} {'':<8} {total_code_lines:<8}")
    print("=" * 60)

if __name__ == '__main__':
    main()
