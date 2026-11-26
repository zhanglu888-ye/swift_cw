import os
import json
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import Counter

# --- 1. 配置区域 ---

# 输入路径
# 图片根目录
image_root = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类试标注数据集"
# 3个标注人员的JSON文件根目录
json_root_person1 = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类试标注数据集_json/4类试标注数据集_zl"
json_root_person2 = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类试标注数据集_json/4类试标注数据集_ls"
json_root_person3 = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类试标注数据集_json/4类试标注数据集_dfl"

# 输出路径
output_root = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类试标注数据集_show"
output_consistent_dir = os.path.join(output_root, "consistent")
output_inconsistent_dir = os.path.join(output_root, "inconsistent")

# 字体文件路径 (请确保此文件存在于脚本同级目录或提供绝对路径)
font_path = "/host/ssddata/disk2/zhanglu/SimHei.ttf"
try:
    font = ImageFont.truetype(font_path, 40)
    font_large = ImageFont.truetype(font_path, 50)
except IOError:
    print(f"错误：字体文件 '{font_path}' 未找到。请确保文件路径正确。")
    print("将使用默认字体，中文可能显示为方框。")
    font = ImageFont.load_default()
    font_large = ImageFont.load_default()


# 为不同类别定义不同颜色，可以自行添加
LABEL_COLORS = {
    "垃圾堆积": "purple",
    "人员倒地": "blue",
    "垃圾桶满溢": "yellow",
    "其他类别": "green",
    "default": "red" # 未在字典中定义的类别使用此颜色
}

# --- 2. 核心功能函数 ---

def get_color(label):
    """根据标签获取颜色"""
    return LABEL_COLORS.get(label, LABEL_COLORS["default"])

def parse_annotation_json(json_path):
    """
    解析单个JSON文件，提取标注信息。
    返回: [(label, bbox, is_valid), ...], bbox为[xmin, ymin, xmax, ymax], is_valid为布尔值
    如果文件不存在或解析失败，返回 None。
    """
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = []
        # 检查 'result' 和 'data' 键是否存在
        if 'result' in data and 'data' in data['result']:
            for item in data['result']['data']:
                label = item['label'][0]
                # 'coordinate' 键包含绝对像素坐标
                coords = item['coordinate']
                x_coords = [p['x'] for p in coords]
                y_coords = [p['y'] for p in coords]
                
                xmin = min(x_coords)
                ymin = min(y_coords)
                xmax = max(x_coords)
                ymax = max(y_coords)
                
                # --- 新增修改：获取 _valid 字段 ---
                # 使用 .get() 方法以防该字段不存在，默认为 True
                is_valid = item.get('_valid', True)
                
                annotations.append((label, [xmin, ymin, xmax, ymax], is_valid))
        return annotations
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"警告：解析JSON文件 '{json_path}' 失败: {e}")
        return [] # 返回空列表表示文件有问题但存在

def draw_annotations(image_path, annotations):
    """
    在图片上绘制标注框和标签。
    如果 annotations 为 None，则写上 "无标注结果"。
    如果 _valid 为 False，额外标注。
    """
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        if annotations is None:
            # 标注文件不存在
            text = "无标注结果"
            try: 
                bbox = draw.textbbox((0, 0), text, font=font_large)
            except AttributeError:
                w, h = draw.textsize(text, font=font_large)
                bbox = (0, 0, w, h)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            pos = ((image.width - text_width) / 2, (image.height - text_height) / 2)
            draw.text(pos, text, fill="red", font=font_large)
        elif not annotations:
            # 标注文件存在但为空
            text = "无标注框"
            try:
                bbox = draw.textbbox((0, 0), text, font=font_large)
            except AttributeError:
                w, h = draw.textsize(text, font=font_large)
                bbox = (0, 0, w, h)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            pos = ((image.width - text_width) / 2, (image.height - text_height) / 2)
            draw.text(pos, text, fill="orange", font=font_large)
        else:
            # --- 修改部分：循环现在包含 is_valid ---
            for label, box, is_valid in annotations:
                color = get_color(label)
                draw.rectangle(box, outline=color, width=3)
                
                # 在框的左上角绘制带背景的标签
                try: 
                    text_bbox = draw.textbbox((box[0], box[1] - 25), label, font=font)
                except AttributeError:
                    tw, th = draw.textsize(label, font=font)
                    text_bbox = (box[0], box[1] - 25, box[0] + tw, box[1] - 25 + th)

                text_x = box[0] + 2
                text_y = box[1] - (text_bbox[3] - text_bbox[1]) - 2
                if text_y < 0:
                    text_y = box[1] + 2
                
                text_bg_box = [text_x, text_y, text_x + (text_bbox[2] - text_bbox[0]), text_y + (text_bbox[3] - text_bbox[1])]
                draw.rectangle(text_bg_box, fill=color)
                draw.text((text_x, text_y), label, fill="black", font=font)

                # --- 新增修改：如果 _valid 为 False，添加额外文字 ---
                if not is_valid:
                    valid_text = "_valid: false"
                    # 将文字放在框的左下角内部
                    valid_text_pos_y = box[3] - 25  # 距底部25像素
                    # 防止文字超出框的顶部（对于很矮的框）
                    if valid_text_pos_y < box[1]:
                        valid_text_pos_y = box[1] + 2
                    
                    draw.text((box[0] + 5, valid_text_pos_y), valid_text, fill="red", font=font)

        return image
    except FileNotFoundError:
        print(f"错误：图片文件未找到 '{image_path}'")
        return None
    except Exception as e:
        print(f"错误：绘制时发生未知问题 '{image_path}': {e}")
        return None

def check_consistency(annotations_list):
    """
    检查3个标注结果是否一致。
    一致性标准：标注框数量相同，且每种标签的数量也相同。
    """
    valid_annotations = [ann for ann in annotations_list if ann is not None]
    
    if len(valid_annotations) != 3:
        return False
        
    counts = [len(ann) for ann in valid_annotations]
    if len(set(counts)) != 1:
        return False

    if counts[0] == 0:
        return True

    # --- 注意: 这里比较的是标签，不考虑_valid状态 ---
    # ann是一个元组列表 [(label, box, is_valid), ...]
    label_counts = [Counter(label for label, box, is_valid in ann) for ann in valid_annotations]
    
    if label_counts[0] != label_counts[1] or label_counts[1] != label_counts[2]:
        return False
        
    return True

# --- 3. 主逻辑 ---

def main():
    """主执行函数"""
    print("开始处理...")
    
    # 清理并创建输出目录
    for path in [output_consistent_dir, output_inconsistent_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # 遍历所有图片
    total_images = 0
    processed_images = 0
    image_paths = []
    for subdir, _, files in os.walk(image_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(subdir, f))
    
    total_images = len(image_paths)

    for image_path in image_paths:
        processed_images += 1
        print(f"处理中 ({processed_images}/{total_images}): {image_path}")

        # 构造相关文件路径
        relative_path = os.path.relpath(image_path, image_root)
        base_name, _ = os.path.splitext(relative_path)
        json_relative_path = base_name + ".json"

        json_path1 = os.path.join(json_root_person1, json_relative_path)
        json_path2 = os.path.join(json_root_person2, json_relative_path)
        json_path3 = os.path.join(json_root_person3, json_relative_path)

        # 解析3个JSON文件
        annotations1 = parse_annotation_json(json_path1)
        annotations2 = parse_annotation_json(json_path2)
        annotations3 = parse_annotation_json(json_path3)
        
        # 检查一致性
        is_consistent = check_consistency([annotations1, annotations2, annotations3])
        
        # 绘制可视化结果
        vis_image1 = draw_annotations(image_path, annotations1)
        vis_image2 = draw_annotations(image_path, annotations2)
        vis_image3 = draw_annotations(image_path, annotations3)

        if not all([vis_image1, vis_image2, vis_image3]):
            print(f"  -> 跳过，因为无法生成所有可视化图片。")
            continue

        # 拼接图片
        max_height = max(vis_image1.height, vis_image2.height, vis_image3.height)
        total_width = vis_image1.width + vis_image2.width + vis_image3.width
        
        concatenated_image = Image.new('RGB', (total_width, max_height))
        concatenated_image.paste(vis_image1, (0, 0))
        concatenated_image.paste(vis_image2, (vis_image1.width, 0))
        concatenated_image.paste(vis_image3, (vis_image1.width + vis_image2.width, 0))
        
        # 确定输出路径并保存
        output_dir = output_consistent_dir if is_consistent else output_inconsistent_dir
        # 在输出文件名中保留目录结构
        output_sub_dir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(output_sub_dir, exist_ok=True)
        
        output_filename = os.path.basename(relative_path)
        output_path = os.path.join(output_sub_dir, output_filename)
        concatenated_image.save(output_path)

    print(f"\n处理完成！总共处理了 {processed_images} 张图片。")
    print(f"一致的结果保存在: {os.path.abspath(output_consistent_dir)}")
    print(f"不一致的结果保存在: {os.path.abspath(output_inconsistent_dir)}")


if __name__ == "__main__":
    main()