#!/usr/bin/env python3

import os

def count_lines_in_file(filepath):
    """统计文件中的代码行数（不包括注释和空行）"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        code_lines = 0
        in_multiline_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过空行
            if not stripped:
                continue
            
            # 处理多行字符串/注释 (""" 或 ''')
            if '"""' in stripped or "'''" in stripped:
                # 简单处理：如果行以这些开始，认为是注释
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                        in_multiline_comment = not in_multiline_comment
                    continue
            
            # 在多行注释中
            if in_multiline_comment:
                continue
            
            # 跳过单行注释
            if stripped.startswith('#'):
                continue
            
            # 这是一行有效代码
            code_lines += 1
        
        return len(lines), code_lines
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, 0

def main():
    files = [
        'sam2_demos/sam2_demo.py',
        'sam2_demos/simple_demo.py', 
        'sam2_demos/interactive_segmentation.py',
        'sam2_demos/batch_segmentation.py',
        'sam2_demos/video_segmentation.py',
        'sam2_demos/auto_mask_generation.py',
        'sam2_demos/sam2_suite.py'
    ]
    
    total_lines = 0
    total_code_lines = 0
    
    print("📊 SAM2 Demo 代码统计")
    print("=" * 50)
    
    for filepath in files:
        if os.path.exists(filepath):
            lines, code_lines = count_lines_in_file(filepath)
            total_lines += lines
            total_code_lines += code_lines
            filename = os.path.basename(filepath)
            print(f"{filename:<30} | 总行数: {lines:>3} | 代码行数: {code_lines:>3}")
        else:
            print(f"❌ 文件不存在: {filepath}")
    
    print("=" * 50)
    print(f"📈 总计:")
    print(f"   文件数: {len(files)}")
    print(f"   总行数: {total_lines}")
    print(f"   代码行数: {total_code_lines}")
    print(f"   代码占比: {total_code_lines/total_lines*100:.1f}%")

if __name__ == "__main__":
    main()
