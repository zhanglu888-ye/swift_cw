# -*- coding: utf-8 -*-
import argparse
import json
import shutil
import os
import requests
import logging
import urllib.parse
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

# 防止超大图片导致 DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    if not name:
        return "unknown"
    return name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_').replace('\n', '')

def get_validity(info_list: list) -> bool:
    """
    根据 info 字段判断数据是否有效。
    """
    if not info_list:
        return False
    for item in info_list:
        val = item.get("value", "")
        # 根据你的需求，如果是 "Yes" 或 "正样本" 则保留
        if val and (val.lower() == "yes" or val == "正样本"):
            return True
    return False

def download_image(url: str, save_path: Path) -> bool:
    """从网络下载图片"""
    try:
        # 如果文件已存在且大小正常，直接跳过下载
        if save_path.exists() and save_path.stat().st_size > 0:
            return True

        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            logging.warning(f"下载失败 (Status {response.status_code}): {url}")
            return False
    except Exception as e:
        logging.error(f"下载异常 {url}: {e}")
        return False

def process_dataset(json_file: Path, output_dir: Path, source_images_dir: Path):
    """
    核心处理逻辑
    :param json_file: 标注的JSON文件路径
    :param output_dir: 结果输出根目录 (包含 cropped_data)
    :param source_images_dir: 存放已下载大图的目录
    """
    # 1. 目录准备
    # 如果没指定大图源目录，默认在 output_dir 下的 large_images
    if not source_images_dir:
        large_images_dir = output_dir / "large_images"
    else:
        large_images_dir = source_images_dir
    
    # 确保大图目录存在，如果不存在则创建（虽然主要期望是读取现有的）
    large_images_dir.mkdir(parents=True, exist_ok=True)
    
    cropped_data_dir = output_dir / "cropped_data"
    cropped_data_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"读取标注文件: {json_file}")
    logging.info(f"图片源目录: {large_images_dir}")
    logging.info(f"结果输出目录: {cropped_data_dir}")

    # 2. 读取JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except Exception as e:
        logging.error(f"JSON文件读取失败: {e}")
        return

    success_cnt = 0
    fail_cnt = 0
    skip_cnt = 0

    # 3. 遍历处理
    for item in tqdm(data_list, desc="Processing"):
        data_info = item.get("data", {})
        result_info = item.get("result", {})

        # --- 校验有效性 ---
        # 如果JSON里标记为"负样本"或者没有"Yes/正样本"，会被跳过
        if not get_validity(result_info.get("info", [])):
            skip_cnt += 1
            continue

        raw_image_url = data_info.get("image_url")
        if not raw_image_url:
            skip_cnt += 1
            continue

        # --- 确定文件名 ---
        # 逻辑：优先使用 data.name，如果没有则从 URL 解析
        save_name = data_info.get("name")
        if not save_name:
            path_part = urllib.parse.urlparse(raw_image_url).path
            save_name = os.path.basename(path_part)
        
        save_name = sanitize_filename(save_name)
        
        # 目标大图路径
        target_image_path = large_images_dir / save_name

        # --- 获取图片 (优先本地查找) ---
        if not target_image_path.exists() or target_image_path.stat().st_size == 0:
            # 如果本地没有，尝试补救下载
            # logging.warning(f"本地未找到图片，尝试下载: {save_name}")
            if not download_image(raw_image_url, target_image_path):
                fail_cnt += 1
                continue

        # --- 处理标注与抠图 ---
        annotations = result_info.get("data", [])
        if not annotations:
            # 只有图片没有标注框
            success_cnt += 1
            continue

        try:
            with Image.open(target_image_path) as img:
                # 修正EXIF旋转
                img = ImageOps.exif_transpose(img)
                
                # 目录层级 1: 图片名 (Stem 去除后缀)
                # 例如: cropped_data/image_01
                case_dir = cropped_data_dir / target_image_path.stem
                
                crop_idx_map = {} # 用于同类别的计数

                for ann in annotations:
                    if ann.get("type") != "rect":
                        continue
                    
                    # 获取类别
                    labels = ann.get("category") or ann.get("label")
                    if not labels:
                        continue
                    
                    # 目录层级 2: 标签名
                    # 例如: cropped_data/image_01/person
                    label_name = sanitize_filename(labels[0])
                    
                    # 获取坐标
                    coords = ann.get("coordinate")
                    if not coords:
                        continue
                    
                    xs = [p['x'] for p in coords]
                    ys = [p['y'] for p in coords]
                    
                    x1, y1 = max(0, min(xs)), max(0, min(ys))
                    x2, y2 = min(img.width, max(xs)), min(img.height, max(ys))
                    
                    # 过滤无效框
                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue

                    box = (x1, y1, x2, y2)
                    crop_img = img.crop(box)

                    # 转RGB防止保存报错 (如PNG转JPG时有透明通道)
                    if crop_img.mode != 'RGB':
                        crop_img = crop_img.convert('RGB')

                    # --- 关键修复：创建带标签层级的目录 ---
                    label_dir = case_dir / label_name
                    label_dir.mkdir(parents=True, exist_ok=True)

                    idx = crop_idx_map.get(label_name, 0) + 1
                    crop_idx_map[label_name] = idx

                    # 保存 Crop 图片
                    crop_filename = f"crop_{idx}.jpg"
                    save_path = label_dir / crop_filename
                    crop_img.save(save_path, quality=95)

                    # 保存 Crop JSON元数据
                    meta = {
                        "source_image": save_name,
                        "category": label_name,
                        "bbox_xyxy": [float(f"{v:.2f}") for v in box],
                        "original_width": img.width,
                        "original_height": img.height
                    }
                    with open(label_dir / f"crop_{idx}.json", 'w', encoding='utf-8') as f_meta:
                        json.dump(meta, f_meta, ensure_ascii=False, indent=2)

            success_cnt += 1

        except Exception as e:
            logging.error(f"处理图片数据出错 {save_name}: {e}")
            fail_cnt += 1

    logging.info("-" * 40)
    logging.info(f"处理完成。")
    logging.info(f"成功处理(图片): {success_cnt}")
    logging.info(f"失败(图片不存在/IO错误): {fail_cnt}")
    logging.info(f"跳过(非正样本/无URL): {skip_cnt}")
    logging.info(f"输出路径: {output_dir}")
    logging.info("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重新处理数据集：使用已下载的大图生成带标签层级的截图数据。")
    
    parser.add_argument("--json_file", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/原始标注结果/非机动车/7953990393325457408/下载结果_json_377_1334_20251205114406.json",
                        help="输入的标注结果JSON文件路径。")
    
    parser.add_argument("--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/非机动车/非机动车_377",
                        help="结果输出目录 (将包含 cropped_data)。")
    
    parser.add_argument("--large_images_dir", type=str, default=None,
                        help="[可选] 已下载的大图存放目录。如果不填，默认去 output_dir/large_images 找。")

    args = parser.parse_args()
    
    source_imgs = Path(args.large_images_dir) if args.large_images_dir else None
    
    process_dataset(Path(args.json_file), Path(args.output_dir), source_imgs)