# -*- coding: utf-8 -*-
import argparse
import json
import shutil
import os
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import logging

# ================= 配置区域 =================

# 1. 【删除列表】：需要彻底丢弃的标签
LABELS_TO_REMOVE = {
    "垃圾桶",
}

# 2. 【修改映射】：将 Key 修改为 Value
LABEL_MAPPING = {
    "河道垃圾": "垃圾堆积",
    "建筑垃圾": "垃圾堆积",
    "街道垃圾": "垃圾堆积",
    "厨余垃圾": "垃圾堆积",
    "垃圾满溢": "垃圾桶满溢",
    "占道经营_002": "占道经营",
    "占道经营_003": "占道经营",
    "占道经营_004": "占道经营",
    "出店经营": "占道经营",
    "非机动车乱停放_有停车标线_正样本": "非机动车乱停放",
    "非机动车乱停放_人行道_正样本": "非机动车乱停放",    
    "非机动车乱停放_楼梯口_正样本": "非机动车乱停放",
    "非机动车乱停放_商圈或店铺门口_正样本": "非机动车乱停放",
    "非机动车乱停放_其他_正样本": "非机动车乱停放",
    "非机动车乱停放_盲道_正样本": "非机动车乱停放",
    "开放空间异常聚集": "人员异常聚集",
    "异质文化异常聚集": "人员异常聚集",    
}

# 3. 【文件名清洗前缀】
FILENAME_PREFIX_TO_STRIP = "UploadImageType."

# ===========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def clean_filename_rule(filename: str) -> str:
    """去除文件名前缀"""
    if filename.startswith(FILENAME_PREFIX_TO_STRIP):
        return filename[len(FILENAME_PREFIX_TO_STRIP):]
    return filename

def get_unique_path(directory: Path, filename: str) -> Path:
    """获取不冲突的文件路径"""
    file_path = directory / filename
    if not file_path.exists():
        return file_path
    
    stem = file_path.stem
    suffix = file_path.suffix
    counter = 1
    while file_path.exists():
        file_path = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return file_path

def get_label_from_folder_name(folder_name: str) -> str:
    """从文件夹名称提取标签"""
    clean_name = re.sub(r'_\d+$', '', folder_name)
    return clean_name

def merge_datasets(input_dirs: list, output_dir: Path):
    out_large_img_dir = output_dir / "large_images"
    out_cropped_dir = output_dir / "cropped_data"
    out_label_json_dir = output_dir / "label_json"

    # --- 自动清理旧数据 ---
    if output_dir.exists():
        logging.warning(f"检测到输出目录已存在: {output_dir}，正在清理...")
        if out_large_img_dir.exists(): shutil.rmtree(out_large_img_dir)
        if out_cropped_dir.exists(): shutil.rmtree(out_cropped_dir)
        if out_label_json_dir.exists(): shutil.rmtree(out_label_json_dir)
    
    out_large_img_dir.mkdir(parents=True, exist_ok=True)
    out_cropped_dir.mkdir(parents=True, exist_ok=True)
    out_label_json_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"开始合并 {len(input_dirs)} 个数据集...")
    
    total_images_copied = 0
    total_crops_copied = 0
    total_images_skipped = 0

    for idx, input_root in enumerate(input_dirs):
        input_root = Path(input_root)
        src_large_dir = input_root / "large_images"
        src_cropped_dir = input_root / "cropped_data"

        # 判断是否为“检测数据集”（是否存在 cropped_data 目录）
        has_cropped_folder = src_cropped_dir.exists()

        # 准备默认标签（仅供无 cropped_data 数据集使用）
        folder_raw_label = get_label_from_folder_name(input_root.name)
        folder_final_label = LABEL_MAPPING.get(folder_raw_label, folder_raw_label)

        if not src_large_dir.exists():
            logging.warning(f"跳过无效目录(无large_images): {input_root}")
            continue

        dataset_type_str = "检测数据集(严格过滤)" if has_cropped_folder else f"分类数据集(使用目录名: {folder_final_label})"
        logging.info(f"处理第 {idx+1} 个数据集: {input_root} [{dataset_type_str}]")
        
        images = list(src_large_dir.glob("*"))
        
        for img_path in tqdm(images, desc=f"Merging Set {idx+1}"):
            if not img_path.is_file(): continue
            
            src_case_folder_name = img_path.stem
            
            # --- 核心逻辑修改：定位 Crop 文件夹 ---
            target_crop_src_path = None # 最终确定的源crop路径
            
            if has_cropped_folder:
                # 尝试查找对应的 crop 文件夹
                possible_path = src_cropped_dir / src_case_folder_name
                if possible_path.exists():
                    target_crop_src_path = possible_path
                elif FILENAME_PREFIX_TO_STRIP:
                    # 尝试前缀兼容查找
                    if not src_case_folder_name.startswith(FILENAME_PREFIX_TO_STRIP):
                        alt_path = src_cropped_dir / (FILENAME_PREFIX_TO_STRIP + src_case_folder_name)
                        if alt_path.exists():
                            target_crop_src_path = alt_path

            # --- 过滤逻辑 ---
            # 如果是检测数据集（has_cropped_folder=True），但找不到对应的 crop 目录
            # 则跳过该大图，不进行复制
            if has_cropped_folder and target_crop_src_path is None:
                total_images_skipped += 1
                continue

            # --- 1. 复制大图 (只有通过了上面的过滤才会执行到这里) ---
            original_name = img_path.name
            cleaned_name = clean_filename_rule(original_name)
            
            target_img_path = get_unique_path(out_large_img_dir, cleaned_name)
            final_img_name = target_img_path.name
            
            shutil.copy2(img_path, target_img_path)
            total_images_copied += 1
            
            current_img_labels = set()
            
            # --- 2. 处理 Cropped Data (如果存在) ---
            has_valid_crops = False
            
            if target_crop_src_path and target_crop_src_path.exists():
                target_case_dir = out_cropped_dir / target_img_path.stem
                
                for cat_dir in target_crop_src_path.iterdir():
                    if not cat_dir.is_dir(): continue
                    raw_cat_name = cat_dir.name
                    
                    if raw_cat_name in LABELS_TO_REMOVE: continue 
                    
                    final_cat_name = LABEL_MAPPING.get(raw_cat_name, raw_cat_name)
                    current_img_labels.add(final_cat_name)

                    if not has_valid_crops:
                        target_case_dir.mkdir(exist_ok=True)
                        has_valid_crops = True
                        
                    target_cat_dir = target_case_dir / final_cat_name
                    target_cat_dir.mkdir(exist_ok=True)

                    for file_item in cat_dir.iterdir():
                        if not file_item.is_file(): continue
                        target_file_path = get_unique_path(target_cat_dir, file_item.name)
                        
                        if file_item.suffix.lower() == ".json":
                            try:
                                with open(file_item, 'r', encoding='utf-8') as f:
                                    meta = json.load(f)
                                meta['original_image'] = final_img_name
                                meta['category'] = final_cat_name
                                with open(target_file_path, 'w', encoding='utf-8') as f:
                                    json.dump(meta, f, ensure_ascii=False, indent=2)
                            except Exception:
                                pass
                        else:
                            shutil.copy2(file_item, target_file_path)
                            if file_item.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                                total_crops_copied += 1
            
            # --- 3. 处理无 Crop 的情况 (仅针对没有 cropped_data 目录的数据集) ---
            if not has_cropped_folder:
                # 这里进入的条件是：整个数据集都没有 cropped_data 文件夹
                if folder_final_label not in LABELS_TO_REMOVE:
                    current_img_labels.add(folder_final_label)

            # --- 4. 生成 Label JSON ---
            if current_img_labels:
                label_info = {
                    "file_name": final_img_name,
                    "labels": list(current_img_labels)
                }
                label_json_path = out_label_json_dir / f"{target_img_path.stem}.json"
                with open(label_json_path, 'w', encoding='utf-8') as f:
                    json.dump(label_info, f, ensure_ascii=False, indent=2)

    logging.info("-" * 40)
    logging.info(f"处理完成。")
    logging.info(f"保留大图: {total_images_copied}")
    logging.info(f"剔除大图(无Crop数据): {total_images_skipped}")

def print_statistics(output_dir: Path):
    """
    统计：框级别 + 图片级别
    """
    logging.info("正在生成统计报表...")
    
    cropped_data_root = output_dir / "cropped_data"
    label_json_root = output_dir / "label_json"
    
    # 统计 Crops
    stats_crops = defaultdict(int)
    if cropped_data_root.exists():
        for case_dir in cropped_data_root.iterdir():
            if not case_dir.is_dir(): continue
            for cat_dir in case_dir.iterdir():
                if not cat_dir.is_dir(): continue
                valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
                count = sum(1 for f in cat_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts)
                if count > 0:
                    stats_crops[cat_dir.name] += count

    # 统计 Images
    stats_images = defaultdict(int)
    total_labeled_images = 0
    if label_json_root.exists():
        json_files = list(label_json_root.glob("*.json"))
        total_labeled_images = len(json_files)
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    labels = data.get("labels", [])
                    for lab in labels:
                        stats_images[lab] += 1
            except Exception as e:
                logging.error(f"读取JSON失败 {jf}: {e}")

    # 打印报表
    all_categories = set(stats_crops.keys()) | set(stats_images.keys())
    sorted_cats = sorted(all_categories, key=lambda k: stats_images[k], reverse=True)
    
    header = f"{'类别名称 (Category)':<30} | {'图片数 (Images)':<15} | {'框数 (Crops)':<15}"
    lines = []
    lines.append("=" * 65)
    lines.append(header)
    lines.append("-" * 65)
    
    total_crop_count = 0
    total_img_count_sum = 0 

    for cat in sorted_cats:
        img_c = stats_images[cat]
        crp_c = stats_crops[cat]
        total_crop_count += crp_c
        total_img_count_sum += img_c
        
        row = f"{cat:<30} | {img_c:<15} | {crp_c:<15}"
        lines.append(row)
        logging.info(row)

    footer_sep = "-" * 65
    footer_total = f"{'Total (Sum)':<30} | {total_img_count_sum:<15} | {total_crop_count:<15}"
    footer_unique = f"Unique Labeled Images: {total_labeled_images}"
    
    lines.append(footer_sep)
    lines.append(footer_total)
    lines.append(footer_unique)
    lines.append("=" * 65)

    logging.info(footer_sep)
    logging.info(footer_total)
    logging.info(footer_unique)
    
    stats_file = output_dir / "final_merge_stats.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"统计文件已保存: {stats_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并数据集：严格过滤无标注大图，并生成统计。")
    parser.add_argument("--inputs", nargs='+', required=True, help="输入文件夹路径列表。")
    parser.add_argument("--output", required=True, help="输出文件夹路径。")

    args = parser.parse_args()
    
    out_path = Path(args.output)
    
    merge_datasets(args.inputs, out_path)
    print_statistics(out_path)