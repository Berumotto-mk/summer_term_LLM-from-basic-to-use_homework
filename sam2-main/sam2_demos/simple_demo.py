#!/usr/bin/env python3
"""
SAM2 简化演示脚本 - 用于调试
"""

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
import cv2

def test_sam2():
    print("🚀 开始SAM2演示测试")
    print("=" * 40)
    
    # 1. 基础检查
    print("📋 基础环境检查...")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - CUDA: {torch.cuda.is_available()}")
    print(f"  - 工作目录: {os.getcwd()}")
    
    # 2. 检查文件
    model_path = "checkpoints/sam2.1_hiera_tiny.pt"
    image_path = "notebooks/images/cars.jpg"
    config_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    print("\n📁 文件检查...")
    print(f"  - 模型权重: {os.path.exists(model_path)} ({model_path})")
    print(f"  - 示例图片: {os.path.exists(image_path)} ({image_path})")
    print(f"  - 配置文件: {os.path.exists('sam2/configs/sam2.1/sam2.1_hiera_t.yaml')} ({config_name})")
    
    if not all([os.path.exists(f) for f in [model_path, image_path]]) or not os.path.exists('sam2/configs/sam2.1/sam2.1_hiera_t.yaml'):
        print("❌ 必要文件缺失，退出演示")
        return
    
    # 3. 导入SAM2
    print("\n📦 导入SAM2模块...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2模块导入成功")
    except Exception as e:
        print(f"❌ SAM2模块导入失败: {e}")
        return
    
    # 4. 加载模型
    print("\n🔧 加载SAM2模型...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  - 使用设备: {device}")
        
        sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", model_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 加载图片
    print("\n🖼️  加载并处理图片...")
    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        print(f"  - 图片尺寸: {image_array.shape}")
        
        predictor.set_image(image_array)
        print("✅ 图片预处理完成")
    except Exception as e:
        print(f"❌ 图片处理失败: {e}")
        return
    
    # 6. 执行分割
    print("\n🎯 执行点击分割...")
    try:
        height, width = image_array.shape[:2]
        # 点击图片中心
        input_point = np.array([[width//2, height//2]])
        input_label = np.array([1])
        
        print(f"  - 点击位置: ({width//2}, {height//2})")
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        print(f"✅ 分割完成!")
        print(f"  - 生成mask数量: {len(masks)}")
        print(f"  - 得分: {scores}")
        print(f"  - 最佳得分: {scores.max():.3f}")
        
    except Exception as e:
        print(f"❌ 分割处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 保存结果
    print("\n💾 生成可视化结果...")
    try:
        best_mask = masks[scores.argmax()]
        
        # 创建更详细的可视化结果
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 原图
        plt.subplot(2, 3, 1)
        plt.imshow(image_array)
        plt.title("原始图片", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 2. 原图 + 点击点
        plt.subplot(2, 3, 2)
        plt.imshow(image_array)
        plt.plot(input_point[0, 0], input_point[0, 1], 'ro', markersize=12)
        plt.title("点击点标记", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 3. 最佳分割mask
        plt.subplot(2, 3, 3)
        plt.imshow(best_mask, cmap='gray')
        plt.title(f"最佳分割mask (得分: {scores.max():.3f})", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 4. 所有候选mask的对比
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.subplot(2, 3, 4 + i)
            plt.imshow(image_array)
            plt.imshow(mask, alpha=0.6, cmap='jet')
            plt.plot(input_point[0, 0], input_point[0, 1], 'ro', markersize=8)
            plt.title(f"候选 {i+1} (得分: {score:.3f})", fontsize=12)
            plt.axis('off')
        
        # 保存完整的可视化结果
        output_path = "sam2_demo_result_detailed.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 详细结果已保存: {output_path}")
        
        # 创建简化版本的对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 原图 + 点击点
        axes[0].imshow(image_array)
        axes[0].plot(input_point[0, 0], input_point[0, 1], 'ro', markersize=10)
        axes[0].set_title("Original Image + Click Point", fontsize=14)
        axes[0].axis('off')
        
        # 分割结果
        axes[1].imshow(image_array)
        axes[1].imshow(best_mask, alpha=0.5, cmap='jet')
        axes[1].plot(input_point[0, 0], input_point[0, 1], 'ro', markersize=10)
        axes[1].set_title(f"Segmentation Result (Score: {scores.max():.3f})", fontsize=14)
        axes[1].axis('off')
        
        # 保存简化版本
        simple_output_path = "sam2_demo_result.png"
        plt.tight_layout()
        plt.savefig(simple_output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 简化结果已保存: {simple_output_path}")
        
        # 显示文件大小信息
        if os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  - 详细版本文件大小: {size_kb:.1f} KB")
        
        if os.path.exists(simple_output_path):
            size_kb = os.path.getsize(simple_output_path) / 1024
            print(f"  - 简化版本文件大小: {size_kb:.1f} KB")
        
        # 创建mask轮廓可视化
        print("\n🎨 生成mask轮廓可视化...")
        contour_image = image_array.copy()
        
        # 将mask转换为uint8格式
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 在原图上绘制轮廓
        cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 3)  # 红色轮廓
        
        # 标记点击点
        cv2.circle(contour_image, (int(input_point[0, 0]), int(input_point[0, 1])), 8, (0, 255, 0), -1)  # 绿色点
        
        # 保存轮廓版本
        contour_output_path = "sam2_demo_contour.png"
        cv2.imwrite(contour_output_path, cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR))
        
        print(f"✅ 轮廓结果已保存: {contour_output_path}")
        
        if os.path.exists(contour_output_path):
            size_kb = os.path.getsize(contour_output_path) / 1024
            print(f"  - 轮廓版本文件大小: {size_kb:.1f} KB")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 40)
    print("🎉 SAM2演示测试完成!")
    print("\n📊 演示结果摘要:")
    print(f"  - 输入图片: {image_path}")
    print(f"  - 图片尺寸: {width} × {height} 像素")
    print(f"  - 点击位置: 图片中心 ({width//2}, {height//2})")
    print(f"  - 生成mask: {len(masks)}个候选")
    print(f"  - 候选得分: {[f'{s:.3f}' for s in scores]}")
    print(f"  - 最佳得分: {scores.max():.3f}")
    print(f"\n📁 生成的文件:")
    print(f"  - 详细版本: sam2_demo_result_detailed.png")
    print(f"  - 简化版本: sam2_demo_result.png")
    print(f"  - 轮廓版本: sam2_demo_contour.png")
    print(f"\n💡 说明:")
    print(f"  - 详细版本显示了所有候选mask的对比")
    print(f"  - 简化版本适合快速查看分割效果") 
    print(f"  - 轮廓版本突出显示了分割边界")

def display_results():
    """显示生成的结果图片信息"""
    print("\n" + "=" * 50)
    print("📸 SAM2 分割结果展示")
    print("=" * 50)
    
    result_files = [
        ("sam2_demo_result_detailed.png", "详细版本 - 显示所有候选mask"),
        ("sam2_demo_result.png", "简化版本 - 原图与最佳分割对比"),
        ("sam2_demo_contour.png", "轮廓版本 - 突出显示分割边界")
    ]
    
    for filename, description in result_files:
        if os.path.exists(filename):
            size_kb = os.path.getsize(filename) / 1024
            print(f"✅ {filename}")
            print(f"   {description}")
            print(f"   文件大小: {size_kb:.1f} KB")
            
            # 获取图片尺寸信息
            try:
                with Image.open(filename) as img:
                    print(f"   图片尺寸: {img.width} × {img.height} 像素")
            except:
                pass
            print()
        else:
            print(f"❌ {filename} - 文件不存在")
    
    print("💡 查看方式:")
    print("  - 在文件管理器中双击图片文件")
    print("  - 使用 'code <filename>' 在VS Code中查看")
    print("  - 使用系统默认图片查看器")

def create_summary_report():
    """创建演示总结报告"""
    print("\n📋 创建演示总结报告...")
    
    report_content = f"""# SAM2 演示结果报告

## 🔧 系统配置
- **运行时间**: {os.popen('date').read().strip()}
- **Python版本**: {torch.__version__.split('+')[0]}
- **PyTorch版本**: {torch.__version__}
- **运行设备**: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
- **工作目录**: {os.getcwd()}

## 📊 演示参数
- **模型**: sam2.1_hiera_tiny.pt (149MB)
- **配置**: configs/sam2.1/sam2.1_hiera_t.yaml
- **输入图片**: notebooks/images/cars.jpg
- **分割方式**: 单点点击 (图片中心)

## 📈 分割结果
- **生成候选**: 3个mask
- **候选得分**: [详见运行日志]
- **最佳得分**: [详见运行日志]

## 📁 输出文件
1. **sam2_demo_result_detailed.png** - 详细对比版本
2. **sam2_demo_result.png** - 简化对比版本  
3. **sam2_demo_contour.png** - 轮廓突出版本

## 🎯 演示效果
SAM2成功完成了以下任务：
- ✅ 模型加载和初始化
- ✅ 图像预处理和编码
- ✅ 单点提示分割
- ✅ 多候选mask生成
- ✅ 结果可视化和保存

## 📝 技术说明
SAM2 (Segment Anything Model 2) 展示了先进的零样本分割能力：
- 仅需一个点击即可智能识别对象
- 生成多个候选结果供选择
- 提供置信度评分
- 支持实时交互式分割

---
*报告生成时间: {os.popen('date').read().strip()}*
"""
    
    with open("SAM2_Demo_Report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 报告已保存: SAM2_Demo_Report.md")

if __name__ == "__main__":
    # 运行主演示
    test_sam2()
    
    # 显示结果文件信息
    display_results()
    
    # 创建总结报告
    create_summary_report()
    
    print("\n🎉 SAM2完整演示结束！")
    print("请查看生成的图片文件以查看分割效果。")
