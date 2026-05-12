# -*- coding: utf-8 -*-
import argparse
import json
import random
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def get_color_for_category(category_name: str):
    """
    根据类别名称生成一个固定的颜色 (RGB)
    """
    # 使用 MD5 哈希生成确定性的颜色
    hash_object = hashlib.md5(category_name.encode())
    hex_dig = hash_object.hexdigest()
    
    # 取前6位转RGB
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    
    # 确保颜色不要太暗或太亮，稍微鲜艳一点
    return (r, g, b)

def load_font(font_path=None, size=24):
    """
    加载字体。为了支持中文，尝试加载系统常见中文字体。
    """
    # 如果用户指定了路径
    if font_path and Path(font_path).exists():
        try:
            return ImageFont.truetype(str(font_path), size)
        except:
            pass
            
    # 常用系统字体列表 (Windows, Linux, MacOS)
    system_fonts = [
        "msyh.ttc",       # Windows 微软雅黑
        "simhei.ttf",     # Windows 黑体
        "arial.ttf",      # 通用
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", # Linux 常见
        "/System/Library/Fonts/PingFang.ttc" # MacOS
    ]
    
    for font in system_fonts:
        try:
            return ImageFont.truetype(font, size)
        except:
            continue
            
    # 如果都失败，使用默认字体 (可能无法显示中文)
    logging.warning("未找到支持中文的字体，中文类别可能会显示为方块。建议通过 --font_path 指定字体文件。")
    return ImageFont.load_default()

def parse_bbox_from_json(json_path: Path):
    """
    从JSON文件中提取 bbox 和 category。
    兼容多种字段名。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        category = data.get("category", "Unknown")
        
        # 尝试获取坐标，兼容之前的脚本版本
        bbox = data.get("bbox") or data.get("bbox_xyxy") or data.get("original_bbox")
        
        if bbox and len(bbox) == 4:
            return category, bbox
            
    except Exception as e:
        logging.warning(f"解析JSON失败 {json_path}: {e}")
        
    return None, None

def visualize_dataset(input_root: Path, output_root: Path, font_path: str = None):
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    src_large_dir = input_root / "large_images"
    src_crop_dir = input_root / "cropped_data"
    
    if not src_large_dir.exists():
        logging.error(f"找不到 large_images 目录: {src_large_dir}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    
    # 加载字体 (字号稍微大一点)
    font = load_font(font_path, size=30)
    
    images = list(src_large_dir.glob("*"))
    processed_count = 0
    
    # 遍历每一张大图
    for img_path in tqdm(images, desc="Visualizing"):
        if not img_path.is_file(): continue
        
        # 寻找对应的标注文件夹
        # 规则: cropped_data/{image_stem}/
        case_name = img_path.stem
        case_dir = src_crop_dir / case_name
        
        # 如果没有对应的标注文件夹，说明这张图没有框，直接跳过或复制原图
        # 这里选择跳过，只画有框的
        if not case_dir.exists():
            continue

        # 收集这张图上的所有框
        annotations = [] # List of (category, bbox)
        
        # 遍历该图下的所有类别文件夹
        for cat_dir in case_dir.iterdir():
            if not cat_dir.is_dir(): continue
            
            # 遍历该类别下的 json 文件
            for json_file in cat_dir.glob("*.json"):
                cat, bbox = parse_bbox_from_json(json_file)
                if cat and bbox:
                    annotations.append((cat, bbox))
        
        if not annotations:
            continue

        try:
            # 打开图片
            with Image.open(img_path) as img:
                # 转换为RGB，方便画图
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                draw = ImageDraw.Draw(img)
                
                # 开始绘制
                for category, bbox in annotations:
                    x1, y1, x2, y2 = bbox
                    color = get_color_for_category(category)
                    
                    # 1. 画矩形框 (线宽5)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                    
                    # 2. 绘制文字背景 (防止文字看不清)
                    text = f"{category}"
                    
                    # 计算文字大小
                    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                    text_w = right - left
                    text_h = bottom - top
                    
                    # 文字背景位置 (在框的左上角上方，如果上方没位置则画在框内)
                    text_bg_x1 = x1
                    text_bg_y1 = y1 - text_h - 10
                    text_bg_x2 = x1 + text_w + 10
                    text_bg_y2 = y1
                    
                    if text_bg_y1 < 0: # 边界处理
                        text_bg_y1 = y1
                        text_bg_y2 = y1 + text_h + 10

                    draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color)
                    
                    # 3. 写文字 (白色)
                    draw.text((text_bg_x1 + 5, text_bg_y1 + 2), text, fill=(255, 255, 255), font=font)
                
                # 保存结果
                save_path = output_root / img_path.name
                img.save(save_path)
                processed_count += 1
                
        except Exception as e:
            logging.error(f"处理图片出错 {img_path.name}: {e}")

    logging.info(f"可视化完成。共生成 {processed_count} 张图片。")
    logging.info(f"保存目录: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将标注框可视化到原始大图上。")
    
    parser.add_argument("--input_root", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并_deduplicated", 
                        help="数据集根目录 (包含 large_images 和 cropped_data)。")
    
    parser.add_argument("--output_root", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并_deduplicated_show", 
                        help="可视化结果保存目录。")
    
    parser.add_argument("--font_path", type=str, default="/host/ssddata/disk2/zhanglu/SimHei.ttf",
                        help="[可选] .ttf/.ttc 字体文件路径。如果在Linux/Docker下且有中文标签，建议指定字体路径 (如 /usr/share/fonts/...)，否则中文可能乱码。")

    args = parser.parse_args()
    
    visualize_dataset(args.input_root, args.output_root, args.font_path)