import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import argparse

def parse_xml(xml_path):
    """
    解析PASCAL VOC格式的XML文件，提取所有对象的类别和边界框。
    
    Args:
        xml_path (str): XML文件路径。
        
    Returns:
        list: 包含字典的列表，每个字典代表一个对象，
              格式为 {'class_name': str, 'bbox': [xmin, ymin, xmax, ymax]}。
    """
    annotations = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            annotations.append({'class_name': class_name, 'bbox': [xmin, ymin, xmax, ymax]})
    except Exception as e:
        print(f"错误: 解析XML文件失败 {xml_path}. 原因: {e}")
    return annotations

def parse_json(json_path):
    """
    解析特定格式的JSON文件，提取所有对象的类别和边界框。
    
    Args:
        json_path (str): JSON文件路径。
        
    Returns:
        list: 包含字典的列表，每个字典代表一个对象，
              格式为 {'class_name': str, 'bbox': [xmin, ymin, xmax, ymax]}。
    """
    annotations = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        objects = data.get('result', {}).get('data', [])
        for obj in objects:
            if obj.get('type') == 'rect':
                # 类别名称在 'label' 列表的第一个元素
                class_name = obj.get('label', [None])[0]
                if not class_name:
                    continue

                # 坐标在 'coordinate' 列表中
                coords = obj.get('coordinate', [])
                if not coords or len(coords) < 2:
                    continue
                
                # 从点坐标计算bbox
                x_coords = [p['x'] for p in coords]
                y_coords = [p['y'] for p in coords]
                xmin = int(min(x_coords))
                ymin = int(min(y_coords))
                xmax = int(max(x_coords))
                ymax = int(max(y_coords))
                
                annotations.append({'class_name': class_name, 'bbox': [xmin, ymin, xmax, ymax]})
    except Exception as e:
        print(f"错误: 解析JSON文件失败 {json_path}. 原因: {e}")
    return annotations


def crop_objects_from_image(image_path, label_path, output_root):
    """
    根据标签文件中的bbox信息，从图片中抠图并保存。
    """
    # 1. 确定标签文件类型并解析
    if label_path.lower().endswith('.xml'):
        annotations = parse_xml(label_path)
    elif label_path.lower().endswith('.json'):
        annotations = parse_json(label_path)
    else:
        # 如果有其他格式，可以在这里扩展
        return

    if not annotations:
        print(f"信息: 在 {label_path} 中未找到任何可用的标注对象。")
        return

    # 2. 打开原始图片
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误: 找不到图片文件 {image_path}")
        return
    except Exception as e:
        print(f"错误: 打开图片失败 {image_path}. 原因: {e}")
        return
        
    # 获取原始图片的基本信息
    img_filename = os.path.basename(image_path)
    img_basename, img_ext = os.path.splitext(img_filename)

    # 3. 遍历所有标注，进行抠图和保存
    crop_counter = 1
    for ann in annotations:
        class_name = ann['class_name']
        bbox = ann['bbox']

        # 创建输出目录结构: output_root/大图名称/类别名称/
        output_dir = os.path.join(output_root, img_basename, class_name)
        os.makedirs(output_dir, exist_ok=True)

        # 抠图
        cropped_image = image.crop(bbox)

        # 构建保存路径和文件名: 大图名称_序号.jpg
        # 注意：这里我们统一保存为.jpg格式，如果需要保留原格式，可以使用 img_ext
        output_filename = f"{img_basename}_{crop_counter}.jpg"
        save_path = os.path.join(output_dir, output_filename)
        
        # 保存抠图
        cropped_image.save(save_path)
        
        crop_counter += 1
    
    print(f"处理完成: 从 {img_filename} 中成功抠取了 {len(annotations)} 个对象。")


def process_directory(input_dir, output_dir):
    """
    处理指定目录下的所有图片和标签文件。
    """
    supported_img_formats = ('.jpg', '.jpeg', '.png', '.bmp')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_img_formats):
            image_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            # 查找对应的标签文件
            xml_path = os.path.join(input_dir, base_name + '.xml')
            json_path = os.path.join(input_dir, base_name + '.json')
            
            label_path = None
            if os.path.exists(xml_path):
                label_path = xml_path
            elif os.path.exists(json_path):
                label_path = json_path

            if label_path:
                crop_objects_from_image(image_path, label_path, output_dir)
            else:
                print(f"跳过: 找不到图片 '{filename}' 对应的XML或JSON标签文件。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="根据XML或JSON标签对图片进行抠图。")
    parser.add_argument(
        '--input_dir', 
        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/tmp_0827/imgandlabel", 
        help="包含图片和标签文件的输入文件夹路径。"
    )
    parser.add_argument(
        '-o', '--output_dir', 
        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/tmp_0827/imgandlabel_show", 
        help="保存抠图的根目录 (默认: 'output')。"
    )
    
    args = parser.parse_args()

    print(f"输入文件夹: {args.input_dir}")
    print(f"输出文件夹: {args.output_dir}")
    
    process_directory(args.input_dir, args.output_dir)
    
    print("\n所有文件处理完毕！")