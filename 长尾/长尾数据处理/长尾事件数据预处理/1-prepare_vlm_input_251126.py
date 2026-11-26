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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    return name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_').replace('\n', '')

def get_validity(info_list: list) -> bool:
    """
    根据 info 字段判断数据是否有效。
    只要有一项 value 为 "Yes" (不区分大小写)，即视为有效。
    """
    if not info_list:
        return False
    for item in info_list:
        val = item.get("value", "")
        if val and val.lower() == "yes":
            return True
        if val and val == "正样本":
            return True
    return False

def download_image(url: str, save_path: Path) -> bool:
    """从网络下载图片"""
    try:
        response = requests.get(url, stream=True, timeout=20)
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

def resolve_local_path(root_dir: Path, raw_path: str) -> Path:
    """
    解析本地路径。
    核心逻辑：
    1. 如果路径中包含 .rar 或 .zip，截取其后的部分。
    2. 将截取后的部分拼接到 root_dir。
    
    示例: 
    Input: "80/175.../data.rar/folder/img.jpg"
    Output: root_dir / "folder/img.jpg"
    """
    # 统一路径分隔符，方便处理
    normalized_path = raw_path.replace('\\', '/')
    
    relative_path = normalized_path
    
    # 查找分隔符位置
    split_markers = ['.rar', '.zip']
    for marker in split_markers:
        # 使用 rsplit 确保处理的是路径中出现的后缀（虽然通常在中间）
        # 这里逻辑是：只要路径里有 .rar/ 或 .zip/，取它后面的部分
        # 注意：这里判断 .rar/ 或 .zip/ 是为了防止文件名本身以rar结尾但不是压缩包目录结构
        # 如果格式是 .rar/... 或 .zip/...
        if marker in normalized_path:
            # 分割，取最后一部分（假设路径中只出现一次压缩包层级，或者我们只关心最内层的）
            parts = normalized_path.split(marker)
            if len(parts) > 1:
                # parts[-1] 是 .rar 之后的部分，通常以 / 开头，如 /folder/1.jpg
                # lstrip('/') 去掉开头的斜杠
                potential_path = parts[-1].lstrip('/')
                if potential_path:
                    relative_path = potential_path
                    break # 找到一种匹配就跳出

    # 拼接最终路径
    full_path = root_dir / relative_path
    return full_path

def process_dataset(json_file: Path, output_dir: Path, local_image_root: Path = None):
    # 1. 目录准备
    large_images_dir = output_dir / "large_images"
    cropped_data_dir = output_dir / "cropped_data"
    large_images_dir.mkdir(parents=True, exist_ok=True)
    cropped_data_dir.mkdir(parents=True, exist_ok=True)

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
        if not get_validity(result_info.get("info", [])):
            skip_cnt += 1
            continue

        raw_image_url = data_info.get("image_url")
        if not raw_image_url:
            skip_cnt += 1
            continue

        # --- 确定图片源类型 (远程 vs 本地) ---
        is_remote = raw_image_url.lower().startswith("http")

        # --- 确定保存的文件名 ---
        # 优先用 name 字段，没有则从 URL/Path 解析
        save_name = data_info.get("name")
        if not save_name:
            if is_remote:
                path_part = urllib.parse.urlparse(raw_image_url).path
                save_name = os.path.basename(path_part)
            else:
                save_name = os.path.basename(raw_image_url)
        
        save_name = sanitize_filename(save_name)
        target_image_path = large_images_dir / save_name

        # --- 获取图片 (下载或拷贝) ---
        image_acquired = False

        if target_image_path.exists():
            # 如果目标路径已有图片，且文件大小>0，直接使用
            if target_image_path.stat().st_size > 0:
                image_acquired = True
        
        if not image_acquired:
            if is_remote:
                # 远程下载
                if download_image(raw_image_url, target_image_path):
                    image_acquired = True
            else:
                # 本地拷贝
                if local_image_root:
                    # 调用路径解析逻辑：去掉 .rar/.zip 之前的部分
                    src_path = resolve_local_path(local_image_root, raw_image_url)
                    
                    if src_path.exists():
                        try:
                            shutil.copy2(src_path, target_image_path)
                            image_acquired = True
                        except Exception as e:
                            logging.error(f"拷贝失败: {src_path} -> {e}")
                    else:
                        logging.warning(f"本地源文件未找到: {src_path} (原始路径: {raw_image_url})")
                else:
                    logging.warning(f"遇到本地路径但未提供 --local_image_root: {raw_image_url}")

        if not image_acquired:
            fail_cnt += 1
            continue

        # --- 处理标注与抠图 ---
        annotations = result_info.get("data", [])
        if not annotations:
            # 图片有了但没标注，也算成功处理了图片，只是不产出crop
            success_cnt += 1
            continue

        try:
            with Image.open(target_image_path) as img:
                # 修正EXIF旋转
                img = ImageOps.exif_transpose(img)
                
                # 创建该图片的输出文件夹
                case_dir = cropped_data_dir / target_image_path.stem
                
                crop_idx_map = {} # 用于同类别的计数

                for ann in annotations:
                    if ann.get("type") != "rect":
                        continue
                    
                    # 获取类别
                    # 优先取 catetips (中文/展示名)，其次 category (英文/ID)
                    labels = ann.get("catetips") or ann.get("category") or ann.get("label")
                    if not labels:
                        continue
                    
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

                    # 转RGB防止保存报错
                    if crop_img.mode != 'RGB':
                        crop_img = crop_img.convert('RGB')

                    # 准备保存路径
                    label_dir = case_dir / label_name
                    label_dir.mkdir(parents=True, exist_ok=True)

                    idx = crop_idx_map.get(label_name, 0) + 1
                    crop_idx_map[label_name] = idx

                    # 保存 Crop 图片
                    crop_filename = f"crop_{idx}.jpg"
                    crop_img.save(label_dir / crop_filename, quality=95)

                    # 保存 Crop JSON元数据
                    meta = {
                        "source_image": save_name,
                        "category": label_name,
                        "bbox_xyxy": [float(f"{v:.2f}") for v in box],
                        "source_type": "remote" if is_remote else "local"
                    }
                    with open(label_dir / f"crop_{idx}.json", 'w', encoding='utf-8') as f_meta:
                        json.dump(meta, f_meta, ensure_ascii=False, indent=2)

            success_cnt += 1

        except Exception as e:
            logging.error(f"处理图片数据出错 {save_name}: {e}")
            fail_cnt += 1

    logging.info("-" * 40)
    logging.info(f"处理完成。")
    logging.info(f"成功: {success_cnt}")
    logging.info(f"失败(下载/查找/IO错误): {fail_cnt}")
    logging.info(f"跳过(无效数据): {skip_cnt}")
    logging.info(f"输出路径: {output_dir}")
    logging.info("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="标注数据处理脚本：支持URL下载与本地路径解析（自动去除压缩包前缀）。")
    
    parser.add_argument("--json_file", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/原始标注结果/占道经营/下载结果_json_314_more_20251205113738.json", type=str, 
                        help="输入的标注结果JSON文件路径。")
    
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/占道经营/占道经营_314", type=str, 
                        help="结果输出目录。")
    
    parser.add_argument("--local_image_root", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/原始标注结果/占道经营", 
                        help="[可选] 本地图片根目录。当JSON中是相对路径时，将在此目录下查找图片。")

    args = parser.parse_args()
    
    local_root = Path(args.local_image_root) if args.local_image_root else None
    
    process_dataset(Path(args.json_file), Path(args.output_dir), local_root)