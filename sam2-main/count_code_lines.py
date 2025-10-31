#!/usr/bin/env python3
"""
代码行数统计工具
统计Python文件中的有效代码行数（不包括注释和空行）
"""

import os
import re

def count_code_lines(file_path):
    """统计单个文件的有效代码行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        in_multiline_string = False
        string_delimiter = None
        
        for line in lines:
            stripped = line.strip()
            
            # 空行
            if not stripped:
                blank_lines += 1
                continue
            
            # 检查多行字符串
            if '"""' in line or "'''" in line:
                if '"""' in line:
                    delimiter = '"""'
                else:
                    delimiter = "'''"
                
                count = line.count(delimiter)
                if count % 2 == 1:  # 奇数个分隔符，切换状态
                    in_multiline_string = not in_multiline_string
                    if in_multiline_string:
                        string_delimiter = delimiter
                
                # 如果这行只包含分隔符或注释，算作注释行
                if stripped.startswith(delimiter) or stripped == delimiter:
                    comment_lines += 1
                    continue
            
            # 在多行字符串内
            if in_multiline_string:
                comment_lines += 1
                continue
            
            # 单行注释
            if stripped.startswith('#'):
                comment_lines += 1
                continue
            
            # 行内注释检查 - 但要注意字符串内的#
            line_has_code = False
            in_string = False
            string_char = None
            i = 0
            
            while i < len(stripped):
                char = stripped[i]
                
                if not in_string:
                    if char in ['"', "'"]:
                        in_string = True
                        string_char = char
                        line_has_code = True
                    elif char == '#':
                        break  # 找到注释，停止检查
                    elif char not in ' \t':
                        line_has_code = True
                else:
                    if char == string_char and (i == 0 or stripped[i-1] != '\\'):
                        in_string = False
                        string_char = None
                
                i += 1
            
            if line_has_code:
                code_lines += 1
            else:
                comment_lines += 1
        
        return {
            'total_lines': len(lines),
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'blank_lines': blank_lines
        }
    
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

def main():
    """主函数"""
    print("📊 SAM2 Demo 代码行数统计")
    print("=" * 60)
    
    demo_files = [
        'sam2_demos/sam2_demo.py',           # 原始基础demo
        'sam2_demos/simple_demo.py',         # 增强基础demo
        'sam2_demos/interactive_segmentation.py',  # 交互式分割
        'sam2_demos/batch_segmentation.py',  # 批量处理
        'sam2_demos/video_segmentation.py',  # 视频分割
        'sam2_demos/auto_mask_generation.py', # 自动掩码生成
        'sam2_demos/sam2_suite.py'          # 应用集成管理器
    ]
    
    total_stats = {
        'total_lines': 0,
        'code_lines': 0,
        'comment_lines': 0,
        'blank_lines': 0
    }
    
    file_stats = []
    
    for file_path in demo_files:
        if os.path.exists(file_path):
            stats = count_code_lines(file_path)
            if stats:
                file_stats.append((file_path, stats))
                
                # 累加统计
                for key in total_stats:
                    total_stats[key] += stats[key]
                
                print(f"📁 {os.path.basename(file_path):<30}")
                print(f"   总行数: {stats['total_lines']:>4} | "
                      f"代码: {stats['code_lines']:>4} | "
                      f"注释: {stats['comment_lines']:>4} | "
                      f"空行: {stats['blank_lines']:>4}")
        else:
            print(f"❌ 文件不存在: {file_path}")
    
    print("\n" + "=" * 60)
    print("📈 总计统计")
    print("-" * 60)
    print(f"总文件数: {len(file_stats)}")
    print(f"总行数:   {total_stats['total_lines']:>6}")
    print(f"代码行数: {total_stats['code_lines']:>6} ({total_stats['code_lines']/total_stats['total_lines']*100:.1f}%)")
    print(f"注释行数: {total_stats['comment_lines']:>6} ({total_stats['comment_lines']/total_stats['total_lines']*100:.1f}%)")
    print(f"空行数:   {total_stats['blank_lines']:>6} ({total_stats['blank_lines']/total_stats['total_lines']*100:.1f}%)")
    print()
    print(f"🎯 有效代码行数（不含注释和空行）: {total_stats['code_lines']} 行")
    
    # 详细分析
    print("\n📋 文件详细分析")
    print("-" * 60)
    file_stats.sort(key=lambda x: x[1]['code_lines'], reverse=True)
    
    for file_path, stats in file_stats:
        filename = os.path.basename(file_path)
        code_ratio = stats['code_lines'] / stats['total_lines'] * 100
        print(f"{filename:<30} | 代码行: {stats['code_lines']:>4} | 代码占比: {code_ratio:>5.1f}%")

if __name__ == "__main__":
    main()
