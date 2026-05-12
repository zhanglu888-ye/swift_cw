# -*- coding: utf-8 -*-
import argparse
import json
import os
import hashlib
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def get_color_from_string(string: str):
    """
    根据字符串生成确定的颜色
    """
    hash_object = hashlib.md5(string.encode())
    hex_dig = hash_object.hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    return (r, g, b)

def load_font(font_path=None, size=24):
    """
    加载字体，优先尝试中文字体
    """
    if font_path and Path(font_path).exists():
        try:
            return ImageFont.truetype(str(font_path), size)
        except:
            pass
    
    # 尝试系统常用中文字体
    system_fonts = [
        "msyh.ttc", "simhei.ttf", "arial.ttf", 
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/System/Library/Fonts/PingFang.ttc"
    ]
    
    for f in system_fonts:
        try:
            return ImageFont.truetype(f, size)
        except:
            continue
            
    logging.warning("未找到中文字体，标签可能会显示为方块。建议使用 --font_path 指定字体。")
    return ImageFont.load_default()

def resolve_image_path(root_dir: Path, raw_url: str) -> Path:
    """
    解析图片路径：自动去除 .rar/.zip 之前的部分
    例如: 80/175.../data.rar/folder/1.jpg -> root_dir/folder/1.jpg
    """
    normalized_path = raw_url.replace('\\', '/')
    relative_path = normalized_path
    
    # 分割压缩包路径
    for marker in ['.rar', '.zip']:
        if marker in normalized_path:
            parts = normalized_path.split(marker)
            if len(parts) > 1:
                # 取压缩包后缀之后的部分，并去除开头的 /
                relative_path = parts[-1].lstrip('/')
                break
                
    return root_dir / relative_path

def is_valid_data(info_list):
    """
    检查数据是否标记为有效 (Yes)
    """
    if not info_list: return False
    for item in info_list:
        if item.get("value", "").lower() == "yes":
            return True
    return False

def visualize_raw_json(json_path: Path, image_root: Path, output_dir: Path, font_path: str = None):
    """
    主处理函数
    """
    if not json_path.exists():
        logging.error(f"JSON文件不存在: {json_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(font_path, size=26)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except Exception as e:
        logging.error(f"JSON解析失败: {e}")
        return

    success_count = 0
    missing_count = 0

    for item in tqdm(data_list, desc="Processing Images"):
        # 1. 获取基本信息
        data_info = item.get("data", {})
        result_info = item.get("result", {})
        
        # 2. 检查有效性
        if not is_valid_data(result_info.get("info", [])):
            continue

        raw_url = data_info.get("image_url")
        if not raw_url: continue

        # 3. 定位本地图片
        img_path = resolve_image_path(image_root, raw_url)
        
        if not img_path.exists():
            # 尝试直接拼接（防止没有rar结构的情况）
            img_path_alt = image_root / raw_url
            if img_path_alt.exists():
                img_path = img_path_alt
            else:
                # logging.warning(f"图片未找到: {img_path}")
                missing_count += 1
                continue

        # 4. 绘图
        try:
            with Image.open(img_path) as img:
                # 修正旋转并转RGB
                img = ImageOps.exif_transpose(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                draw = ImageDraw.Draw(img)
                annotations = result_info.get("data", [])
                
                has_bbox = False

                for ann in annotations:
                    if ann.get("type") != "rect": continue
                    
                    # 获取标签 (优先取中文 catetips)
                    labels = ann.get("catetips") or ann.get("label") or ann.get("category")
                    if not labels: continue
                    label_text = labels[0]
                    
                    # 获取坐标 (Raw JSON 是 coordinate 列表)
                    coords = ann.get("coordinate")
                    if not coords: continue
                    
                    xs = [p['x'] for p in coords]
                    ys = [p['y'] for p in coords]
                    
                    x1, y1 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                    
                    # 绘制矩形
                    color = get_color_from_string(label_text)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    
                    # 绘制标签背景和文字
                    text_bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    
                    # 标签位置：框的上方，如果上方不够则画在框内
                    tx, ty = x1, y1 - text_h - 10
                    if ty < 0: ty = y1 + 5
                    
                    draw.rectangle([tx, ty, tx + text_w + 10, ty + text_h + 10], fill=color)
                    draw.text((tx + 5, ty + 2), label_text, fill=(255, 255, 255), font=font)
                    
                    has_bbox = True
                
                if has_bbox:
                    # 保存图片，使用原始文件名
                    save_name = img_path.name
                    # 如果重名，添加前缀区分
                    save_path = output_dir / save_name
                    if save_path.exists():
                        # 使用父文件夹名作为前缀避免冲突
                        save_path = output_dir / f"{img_path.parent.name}_{save_name}"
                        
                    img.save(save_path)
                    success_count += 1

        except Exception as e:
            logging.error(f"处理图片出错 {img_path}: {e}")

    logging.info("=" * 40)
    logging.info(f"可视化完成")
    logging.info(f"成功生成图片: {success_count}")
    logging.info(f"图片缺失数量: {missing_count}")
    logging.info(f"结果保存目录: {output_dir}")
    logging.info("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据原始JSON标注文件可视化图片。")
    
    parser.add_argument("--json_file", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/原始标注结果/垃圾堆积/下载结果_json_317_more_20251126104125.json", type=str,
                        help="原始标注 JSON 文件路径。")
    
    parser.add_argument("--image_root", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/原始标注结果/垃圾堆积", type=str,
                        help="本地图片所在的根目录（脚本会自动去匹配JSON中的相对路径）。")
    
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/原始标注结果/垃圾堆积/317_show", type=str,
                        help="可视化结果保存目录。")
    
    parser.add_argument("--font_path", type=str,default="/host/ssddata/disk2/zhanglu/SimHei.ttf",
                        help="字体文件路径 (.ttf/.ttc)，用于正常显示中文标签。")

    args = parser.parse_args()
    
    visualize_raw_json(
        Path(args.json_file),
        Path(args.image_root),
        Path(args.output_dir),
        args.font_path
    )