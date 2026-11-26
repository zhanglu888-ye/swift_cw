import os
import json
import shutil
import argparse
from pathlib import Path

# 定义支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ================= 配置区域 =================
# 定义标签转换映射表 { "旧标签": "新标签" }
LABEL_MAPPING = {
    "出店经营": "占道经营",
    "非机动车乱停乱放": "非机动车乱停放",
    "雨水井违规倾倒污水": "污水井倾倒",
    "人群聚集事件": "人员异常聚集",
    "倒地事件": "人员异常倒地"
}
# ===========================================

def get_unique_filename(destination_dir, filename):
    """
    如果目标文件夹已存在同名文件，生成一个唯一的新文件名。
    例如: 1.jpg -> 1_1.jpg -> 1_2.jpg
    """
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(destination_dir, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1
    return new_filename

def transform_labels(raw_labels):
    """
    根据映射表转换标签，并去重
    """
    transformed = set()
    for label in raw_labels:
        # 如果标签在映射表中，则使用新标签；否则保留原标签
        new_label = LABEL_MAPPING.get(label, label)
        transformed.add(new_label)
    return list(transformed)

def extract_labels_from_source_json(json_path):
    """
    从源JSON中提取 catetips 标签
    """
    labels = set()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 遍历 annotations 列表
        annotations = data.get('annotations', [])
        for ann in annotations:
            catetips = ann.get('catetips', [])
            for tip in catetips:
                labels.add(tip)
                
    except Exception as e:
        print(f"读取JSON失败 {json_path}: {e}")
        
    return list(labels)

def process_dataset(input_root, output_root):
    # 1. 创建输出目录
    large_images_dir = os.path.join(output_root, "large_images")
    label_json_dir = os.path.join(output_root, "label_json")
    
    os.makedirs(large_images_dir, exist_ok=True)
    os.makedirs(label_json_dir, exist_ok=True)
    
    print(f"开始处理数据...")
    print(f"输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    print(f"标签映射规则: {json.dumps(LABEL_MAPPING, ensure_ascii=False)}")
    
    total_processed = 0
    
    # 获取输入目录下的所有一级子文件夹 (例如: "出店经营", "垃圾堆积")
    sub_dirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    for sub_dir in sub_dirs:
        current_dir = os.path.join(input_root, sub_dir)
        
        # 检查是否存在 pic 和 json 子文件夹 (情况1)
        pic_subdir = os.path.join(current_dir, "pic")
        json_subdir = os.path.join(current_dir, "json")
        
        is_scenario_a = os.path.isdir(pic_subdir) and os.path.isdir(json_subdir)
        
        if is_scenario_a:
            # === 情况 1: 有 pic 和 json 文件夹 ===
            print(f"正在处理 [情况1 - 原有JSON]: {sub_dir}")
            
            # 遍历 pic 文件夹下的图片
            for img_name in os.listdir(pic_subdir):
                img_ext = os.path.splitext(img_name)[1].lower()
                if img_ext not in IMAGE_EXTENSIONS:
                    continue
                
                src_img_path = os.path.join(pic_subdir, img_name)
                
                # 寻找对应的 JSON 文件
                json_name = os.path.splitext(img_name)[0] + ".json"
                src_json_path = os.path.join(json_subdir, json_name)
                
                raw_labels = []
                
                # 读取原始标签
                if os.path.exists(src_json_path):
                    raw_labels = extract_labels_from_source_json(src_json_path)
                else:
                    print(f"警告: 找不到对应的JSON {src_json_path}，标签设为空")
                    raw_labels = []
                
                # [修改点] 进行标签转换
                final_labels = transform_labels(raw_labels)
                
                # 执行复制和生成
                total_processed += 1
                save_data(src_img_path, img_name, final_labels, large_images_dir, label_json_dir)
                
        else:
            # === 情况 2: 只有图片，文件夹名即为标签 ===
            print(f"正在处理 [情况2 - 文件夹名即标签]: {sub_dir}")
            
            # 文件夹名称就是原始标签
            raw_folder_label = sub_dir
            
            # [修改点] 进行标签转换 (把文件夹名作为标签传入)
            final_labels = transform_labels([raw_folder_label])
            
            for img_name in os.listdir(current_dir):
                img_ext = os.path.splitext(img_name)[1].lower()
                if img_ext not in IMAGE_EXTENSIONS:
                    continue
                
                src_img_path = os.path.join(current_dir, img_name)
                
                # 执行复制和生成
                total_processed += 1
                save_data(src_img_path, img_name, final_labels, large_images_dir, label_json_dir)

    print("-" * 30)
    print(f"处理完成！共处理图片: {total_processed} 张")
    print(f"图片保存至: {large_images_dir}")
    print(f"标签保存至: {label_json_dir}")

def save_data(src_img_path, original_filename, labels, images_out_dir, jsons_out_dir):
    """
    复制图片并保存对应的 label json
    """
    # 1. 确定唯一的目标文件名 (防止不同文件夹下有重名文件)
    dst_filename = get_unique_filename(images_out_dir, original_filename)
    dst_img_path = os.path.join(images_out_dir, dst_filename)
    
    # 2. 复制图片
    try:
        shutil.copy2(src_img_path, dst_img_path)
    except Exception as e:
        print(f"复制图片失败 {src_img_path}: {e}")
        return

    # 3. 生成目标 JSON 内容
    output_data = {
        "file_name": dst_filename,
        "labels": labels
    }
    
    # 4. 保存 JSON
    dst_json_name = os.path.splitext(dst_filename)[0] + ".json"
    dst_json_path = os.path.join(jsons_out_dir, dst_json_name)
    
    with open(dst_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并多类别图片并生成统一格式的标签JSON")
    
    # 默认路径配置，可根据需要修改
    parser.add_argument("--input_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao/原始数据", help="输入的数据集根目录")
    parser.add_argument("--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao/合并数据", help="输出结果的根目录")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 {args.input_dir}")
    else:
        process_dataset(args.input_dir, args.output_dir)