import os
import json
import shutil
from pathlib import Path
import argparse

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='处理JSON文件并移动符合条件的文件')
    parser.add_argument('--input_dir', type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511//vlm_prepare/正样本/分类and描述_1215", help='输入的根目录路径')
    parser.add_argument('--output_dir', type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511//vlm_prepare/正样本/分类and描述_1215_删除", help='输出移动的目>标大目录')
    return parser.parse_args()

def should_move_json(json_data):
    """
    判断JSON文件是否需要移动
    条件：comprehensive_assessment中的"异常类型": "无异常", "风险等级": "0级 - 无风险"
    """
    try:
        # 获取comprehensive_assessment
        analyses = json_data.get('analyses', [])
        if not analyses:
            print("  未找到analyses字段")
            return False
        
        if not isinstance(analyses, list) or len(analyses) == 0:
            print("  analyses不是列表或为空")
            return False
        
        event_analyses = analyses[0].get('event_analyses', [])
        if not event_analyses or not isinstance(event_analyses, list):
            print("  event_analyses不是列表或为空")
            return False
        
        comprehensive_assessment = event_analyses[0].get('comprehensive_assessment', {})
        
        abnormal_type = comprehensive_assessment.get('异常类型', '')
        risk_level = comprehensive_assessment.get('风险等级', '')
        
        print(f"  异常类型: '{abnormal_type}', 风险等级: '{risk_level}'")
        
        # 检查是否符合移动条件
        if (abnormal_type == "无异常" and 
            risk_level == "0级 - 无风险"):
            print("  ✓ 符合条件：无异常 + 0级风险")
            return True
        else:
            print(f"  ✗ 不符合移动条件")
            return False
            
    except Exception as e:
        print(f"  解析JSON时出错: {e}")
        return False

def move_json_file(source_path, target_dir, filename, category_name):
    """
    移动JSON文件到目标目录
    """
    try:
        # 创建目标目录（如果不存在）
        target_category_dir = os.path.join(target_dir, category_name)
        os.makedirs(target_category_dir, exist_ok=True)
        
        # 构建目标文件路径
        target_path = os.path.join(target_category_dir, filename)
        
        # 如果目标文件已存在，添加后缀避免覆盖
        counter = 1
        original_target_path = target_path
        while os.path.exists(target_path):
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{counter}{ext}"
            target_path = os.path.join(target_category_dir, new_filename)
            counter += 1
        
        # 移动文件
        shutil.move(source_path, target_path)
        print(f"  ✓ 已移动: {filename} -> {category_name}/")
        return True
        
    except Exception as e:
        print(f"  ✗ 移动文件失败 {filename}: {e}")
        return False

def process_json_files(input_dir, output_dir):
    """
    主处理函数：遍历所有文件夹和JSON文件
    """
    # 统计信息
    total_files = 0
    moved_files = 0
    no_move_needed = 0
    error_files = 0
    
    print(f"开始扫描目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    # 遍历输入目录下的所有子文件夹
    for root, dirs, files in os.walk(input_dir):
        # 跳过输出目录本身（如果它在输入目录内）
        if output_dir.startswith(root):
            continue
            
        # 获取当前文件夹相对于输入目录的路径，用于确定事件类别
        relative_path = os.path.relpath(root, input_dir)
        
        # 如果relative_path是当前目录（即输入目录的直接子文件夹），则直接使用文件夹名作为类别
        if relative_path == ".":
            # 这是输入目录本身，跳过
            continue
        
        # 获取事件类别（子文件夹名称）
        # 如果是在输入目录的直接子文件夹中，relative_path就是文件夹名
        # 如果在更深层的文件夹中，我们需要获取最底层的文件夹名
        category_name = os.path.basename(root)
        
        print(f"\n处理类别文件夹: [{category_name}]")
        print(f"完整路径: {root}")
        
        for file in files:
            if file.lower().endswith('.json'):
                total_files += 1
                source_path = os.path.join(root, file)
                
                try:
                    # 读取JSON文件
                    with open(source_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    print(f"\n  [{total_files}] 处理文件: {file}")
                    
                    # 检查是否需要移动
                    if should_move_json(json_data):
                        print(f"  → 符合条件，准备移动到类别: {category_name}")
                        
                        # 移动文件（类别名称直接从文件夹名获取）
                        if move_json_file(source_path, output_dir, file, category_name):
                            moved_files += 1
                        else:
                            error_files += 1
                    else:
                        print(f"  → 不符合移动条件，保留在原位置")
                        no_move_needed += 1
                        
                except json.JSONDecodeError as e:
                    print(f"  ✗ JSON解析错误 {file}: {e}")
                    error_files += 1
                except Exception as e:
                    print(f"  ✗ 处理文件 {file} 时发生错误: {e}")
                    error_files += 1
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("处理完成！统计信息:")
    print(f"总JSON文件数: {total_files}")
    print(f"成功移动文件数: {moved_files}")
    print(f"不符合移动条件: {no_move_needed}")
    print(f"错误/跳过文件数: {error_files}")
    print("=" * 80)

def main():
    """主函数"""
    args = parse_arguments()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # 验证输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在!")
        return
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    print("JSON文件处理和移动工具")
    print("源事件类别将从输入目录的子文件夹名称获取")
    print("=" * 80)
    
    # 处理JSON文件
    process_json_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
