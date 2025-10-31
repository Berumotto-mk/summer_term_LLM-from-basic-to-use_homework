#!/usr/bin/env python3
"""
SAM2 应用集成中心
统一运行和管理所有SAM2应用
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

class SAM2ApplicationSuite:
    def __init__(self):
        self.apps = {
            "basic": {
                "name": "基础分割演示",
                "script": "simple_demo.py",
                "description": "单点点击分割，生成多种可视化"
            },
            "interactive": {
                "name": "交互式分割",
                "script": "interactive_segmentation.py", 
                "description": "多点交互式分割，支持前景/背景点"
            },
            "batch": {
                "name": "批量处理",
                "script": "batch_segmentation.py",
                "description": "批量处理多个图像，多种策略"
            },
            "video": {
                "name": "视频分割",
                "script": "video_segmentation.py",
                "description": "视频对象追踪和分割"
            },
            "auto": {
                "name": "自动掩码生成",
                "script": "auto_mask_generation.py",
                "description": "全图自动分割，无需用户输入"
            },
            "advanced": {
                "name": "高级视频应用",
                "script": "advanced_video_applications.py",
                "description": "目标提取、背景替换、目标跟踪"
            },
            "real-video": {
                "name": "真实视频处理",
                "script": "real_video_processor.py",
                "description": "处理实际视频文件的高级分割和跟踪"
            },
            "simple-video": {
                "name": "简单视频处理",
                "script": "simple_video_processor.py", 
                "description": "轻量级视频分析和运动检测"
            }
        }
    
    def list_applications(self):
        """列出所有可用应用"""
        print("🚀 SAM2 应用套件")
        print("=" * 60)
        print("可用应用:")
        print()
        
        for key, app in self.apps.items():
            status = "✅" if self.check_script_exists(app["script"]) else "❌"
            print(f"{status} {key:12} - {app['name']}")
            print(f"   {'':12}   {app['description']}")
            print(f"   {'':12}   脚本: {app['script']}")
            print()
        
        print("使用方法:")
        print("  python sam2_suite.py <应用名称>")
        print("  python sam2_suite.py --list    # 显示此列表")
        print("  python sam2_suite.py --all     # 运行所有应用")
    
    def check_script_exists(self, script_name):
        """检查脚本文件是否存在"""
        script_path = os.path.join("sam2_demos", script_name)
        return os.path.exists(script_path)
    
    def run_application(self, app_key):
        """运行指定应用"""
        if app_key not in self.apps:
            print(f"❌ 未知应用: {app_key}")
            print("使用 --list 查看可用应用")
            return False
        
        app = self.apps[app_key]
        script_path = os.path.join("sam2_demos", app["script"])
        
        if not os.path.exists(script_path):
            print(f"❌ 脚本文件不存在: {script_path}")
            return False
        
        print(f"🚀 启动应用: {app['name']}")
        print(f"📝 描述: {app['description']}")
        print(f"📁 脚本: {script_path}")
        print("-" * 50)
        
        try:
            # 运行应用脚本
            result = subprocess.run([sys.executable, script_path], 
                                  cwd=os.getcwd(),
                                  capture_output=False)
            
            if result.returncode == 0:
                print(f"✅ 应用 '{app['name']}' 运行完成")
                return True
            else:
                print(f"❌ 应用 '{app['name']}' 运行失败 (退出码: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"❌ 运行应用时出错: {e}")
            return False
    
    def run_all_applications(self):
        """运行所有可用应用"""
        print("🚀 运行所有SAM2应用")
        print("=" * 60)
        
        results = {}
        available_apps = [key for key, app in self.apps.items() 
                         if self.check_script_exists(app["script"])]
        
        print(f"找到 {len(available_apps)} 个可用应用")
        print()
        
        for i, app_key in enumerate(available_apps, 1):
            print(f"📋 [{i}/{len(available_apps)}] 运行应用: {app_key}")
            results[app_key] = self.run_application(app_key)
            print()
        
        # 输出总结
        print("📊 运行总结")
        print("-" * 40)
        success_count = sum(results.values())
        total_count = len(results)
        
        for app_key, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {app_key:12} - {self.apps[app_key]['name']}")
        
        print()
        print(f"总计: {success_count}/{total_count} 个应用运行成功")
        
        return success_count == total_count
    
    def create_summary_report(self):
        """创建应用套件总结报告"""
        report_content = f"""# SAM2 应用套件运行报告

## 📋 基本信息
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **应用总数**: {len(self.apps)}

## 🚀 应用列表

"""
        
        for key, app in self.apps.items():
            exists = self.check_script_exists(app["script"])
            status = "✅ 可用" if exists else "❌ 缺失"
            
            report_content += f"""### {app['name']} ({key})
- **状态**: {status}
- **脚本**: `{app['script']}`
- **描述**: {app['description']}

"""
        
        report_content += f"""## 💡 使用指南

### 单独运行应用
```bash
python sam2_suite.py <应用名称>
```

### 运行所有应用
```bash
python sam2_suite.py --all
```

### 查看应用列表
```bash
python sam2_suite.py --list
```

## 📂 文件结构
```
sam2_demos/
├── simple_demo.py           # 基础分割演示
├── interactive_segmentation.py  # 交互式分割
├── batch_segmentation.py    # 批量处理
├── video_segmentation.py    # 视频分割
├── auto_mask_generation.py  # 自动掩码生成
└── sam2_suite.py           # 应用集成中心
```

## 🎯 应用场景
- **医疗影像**: 器官和病变分割
- **自动驾驶**: 道路场景理解
- **工业检测**: 缺陷检测和质量控制
- **内容创作**: 视频编辑和对象移除
- **科研分析**: 显微镜图像分析
"""
        
        report_path = "sam2_demos/SAM2_Suite_Report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📋 应用套件报告已保存: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="SAM2 应用套件管理器")
    parser.add_argument("app", nargs="?", help="要运行的应用名称")
    parser.add_argument("--list", action="store_true", help="列出所有可用应用")
    parser.add_argument("--all", action="store_true", help="运行所有可用应用")
    parser.add_argument("--report", action="store_true", help="生成应用套件报告")
    
    args = parser.parse_args()
    
    suite = SAM2ApplicationSuite()
    
    if args.list or (not args.app and not args.all and not args.report):
        suite.list_applications()
    elif args.all:
        suite.run_all_applications()
        suite.create_summary_report()
    elif args.report:
        suite.create_summary_report()
    elif args.app:
        suite.run_application(args.app)
    else:
        suite.list_applications()

if __name__ == "__main__":
    main()
