# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from imagededup.methods import PHash
from collections import defaultdict
import logging

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def print_statistics(cropped_data_root: Path):
    """
    统计并打印 cropped_data 目录下的类别分布情况
    """
    logging.info("\n" + "="*65)
    logging.info("正在生成去重后的统计报表...")
    
    if not cropped_data_root.exists():
        logging.warning(f"目录不存在，无法统计: {cropped_data_root}")
        return

    stats_crops = defaultdict(int)
    stats_images = defaultdict(int)
    
    # 遍历 Case 文件夹
    for case_dir in cropped_data_root.iterdir():
        if not case_dir.is_dir(): continue
        
        # 遍历 Category 文件夹
        for cat_dir in case_dir.iterdir():
            if not cat_dir.is_dir(): continue
            
            cat_name = cat_dir.name
            
            # 统计图片文件
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            count = sum(1 for f in cat_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts)
            
            if count > 0:
                stats_crops[cat_name] += count
                stats_images[cat_name] += 1

    # 排序 (按框数量从多到少)
    sorted_cats = sorted(stats_crops.keys(), key=lambda k: stats_crops[k], reverse=True)
    
    # 打印表格
    header = f"{'类别名称 (Category)':<30} | {'原图数 (Images)':<15} | {'框数 (Crops)':<15}"
    separator = "-" * 65
    
    logging.info(separator)
    logging.info(header)
    logging.info(separator)
    
    total_crops = 0
    total_images_accumulated = 0 # 仅作参考，非去重原图总数
    
    output_lines = [
        f"去重后统计结果 (目录: {cropped_data_root})",
        separator,
        header,
        separator
    ]

    for cat in sorted_cats:
        c_crops = stats_crops[cat]
        c_imgs = stats_images[cat]
        total_crops += c_crops
        
        line = f"{cat:<30} | {c_imgs:<15} | {c_crops:<15}"
        logging.info(line)
        output_lines.append(line)
        
    logging.info(separator)
    logging.info(f"{'Total (所有类别总计)':<48} | {total_crops:<15}")
    logging.info("=" * 65)

    # 保存统计文件到输出目录的父级
    output_lines.append(separator)
    output_lines.append(f"Total Crops: {total_crops}")
    
    stats_file = cropped_data_root.parent / "deduplicated_stats.txt"
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        logging.info(f"统计文件已保存: {stats_file}")
    except Exception as e:
        logging.error(f"保存统计文件失败: {e}")


def deduplicate_dataset(input_root, output_root, threshold=15):
    """
    对包含 large_images 和 cropped_data 的数据集进行去重。
    1. 计算 large_images 的 PHash。
    2. 保留唯一图片，删除重复图片。
    3. 复制保留的大图及其对应的 cropped_data 文件夹到新目录。
    4. 统计新目录的数据分布。
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    src_large_dir = input_root / "large_images"
    src_crop_dir = input_root / "cropped_data"
    
    dst_large_dir = output_root / "large_images"
    dst_crop_dir = output_root / "cropped_data"

    if not src_large_dir.exists():
        logging.error(f"错误: 输入目录下找不到 large_images 文件夹: {input_root}")
        return

    # 创建输出目录
    dst_large_dir.mkdir(parents=True, exist_ok=True)
    dst_crop_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: PHash 计算 ---
    logging.info(f"正在计算 PHash (阈值: {threshold})...")
    phasher = PHash()

    try:
        # imagededup 需要字符串路径
        encodings = phasher.encode_images(image_dir=str(src_large_dir))
        duplicates = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=threshold)
    except Exception as e:
        logging.error(f"PHash 处理出错: {e}")
        return

    # --- Step 2: 分析保留列表 ---
    keep_files = []      
    processed_files = set() 
    
    # 排序以保证结果确定性
    all_files = sorted(duplicates.keys())

    for filename in all_files:
        if filename in processed_files:
            continue
        
        # 遇到未处理的文件，视为保留
        keep_files.append(filename)
        processed_files.add(filename)
        
        # 将其重复项标记为已处理 (即丢弃)
        matches = duplicates.get(filename, [])
        for match in matches:
            processed_files.add(match)

    total_files = len(all_files)
    kept_count = len(keep_files)
    removed_count = total_files - kept_count
    
    logging.info(f"去重分析完成: 原有 {total_files} 张, 保留 {kept_count} 张, 计划删除 {removed_count} 张")

    # --- Step 3: 执行复制 ---
    logging.info("开始迁移文件 (大图 + 标注数据)...")
    
    success_crop_folders = 0

    for filename in tqdm(keep_files, desc="Copying"):
        # A. 复制大图
        src_img_path = src_large_dir / filename
        dst_img_path = dst_large_dir / filename
        
        if src_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        
        # B. 复制对应的小图文件夹
        # 规则: cropped_data 下的文件夹名通常是大图的文件名(无后缀)
        case_name = src_img_path.stem
        src_case_folder = src_crop_dir / case_name
        dst_case_folder = dst_crop_dir / case_name
        
        if src_case_folder.exists() and src_case_folder.is_dir():
            # 完整复制该图片的标注文件夹 (包含 crop 图片和 json)
            shutil.copytree(src_case_folder, dst_case_folder, dirs_exist_ok=True)
            success_crop_folders += 1

    logging.info("=" * 40)
    logging.info(f"迁移完成 Summary:")
    logging.info(f"保留大图: {kept_count}")
    logging.info(f"保留Crop文件夹: {success_crop_folders}")
    logging.info("=" * 40)

    # --- Step 4: 统计新数据集 ---
    print_statistics(dst_crop_dir)


if __name__ == '__main__':
    # ================= 参数配置 =================
    
    # 输入：应该是包含 large_images 和 cropped_data 的父目录
    input_directory = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并" 
    
    # 输出：去重后保存的新目录
    output_directory = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并_deduplicated"
   
    # 哈希距离阈值 (建议 10-15)
    hash_threshold = 15
    
    # ===========================================
    
    deduplicate_dataset(
        input_root=input_directory,
        output_root=output_directory,
        threshold=hash_threshold
    )