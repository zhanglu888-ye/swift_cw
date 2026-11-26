# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from collections import defaultdict
import logging

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(message)s')

def count_categories(cropped_data_root: Path):
    """
    统计 cropped_data 结构下的:
    1. 每个类别的标注框总数 (Crop Count)
    2. 包含该类别的原图数量 (Image Count)
    """
    if not cropped_data_root.exists():
        logging.error(f"目录不存在: {cropped_data_root}")
        return

    # 两个字典用于统计
    # stats_crops[category] = 框的总数
    # stats_images[category] = 包含该类别的图片数
    stats_crops = defaultdict(int)
    stats_images = defaultdict(int)
    
    total_cases_processed = 0
    
    # 遍历第一层：原图/Case 目录
    for case_dir in cropped_data_root.iterdir():
        if not case_dir.is_dir():
            continue
        
        total_cases_processed += 1
        
        # 遍历第二层：类别目录
        for category_dir in case_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            
            # 统计有效图片文件
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            current_dir_crop_count = 0
            
            for file_path in category_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    current_dir_crop_count += 1
            
            # 只有当该目录下确实有切图时，才计入统计
            if current_dir_crop_count > 0:
                stats_crops[category_name] += current_dir_crop_count
                stats_images[category_name] += 1 # 该类别在当前这张原图中出现了，原图计数+1

    # --- 准备输出 ---
    # 按框的数量从大到小排序
    sorted_categories = sorted(stats_crops.keys(), key=lambda k: stats_crops[k], reverse=True)

    header = f"{'类别名称 (Category)':<30} | {'原图数量 (Images)':<18} | {'框数量 (Crops)':<15}"
    separator = "-" * 70

    # 控制台打印
    logging.info("=" * 70)
    logging.info(f"统计根目录: {cropped_data_root}")
    logging.info(f"总处理原图文件夹数: {total_cases_processed}")
    logging.info(separator)
    logging.info(header)
    logging.info(separator)

    total_crops_all = 0
    total_images_accumulated = 0 # 这个数字意义不大，仅作参考

    output_lines = []
    output_lines.append(f"统计目录: {cropped_data_root}")
    output_lines.append(f"总原图文件夹: {total_cases_processed}")
    output_lines.append(separator)
    output_lines.append(header)
    output_lines.append(separator)

    for cat in sorted_categories:
        img_cnt = stats_images[cat]
        crop_cnt = stats_crops[cat]
        
        line = f"{cat:<30} | {img_cnt:<18} | {crop_cnt:<15}"
        logging.info(line)
        output_lines.append(line)
        
        total_crops_all += crop_cnt

    logging.info(separator)
    logging.info(f"{'总计 (Total)':<51} | {total_crops_all:<15}")
    logging.info("=" * 70)

    output_lines.append(separator)
    output_lines.append(f"总框数: {total_crops_all}")

    # 保存报告
    save_report_path = cropped_data_root.parent / "category_stats_report.txt"
    try:
        with open(save_report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        logging.info(f"\n[提示] 统计报告已保存至: {save_report_path}")
    except Exception as e:
        logging.warning(f"无法保存报告: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 cropped_data 目录下各子类别的标注框数量。")
    
    parser.add_argument("--input_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/非机动车/非机动车_377/cropped_data", type=str,
                        help="cropped_data 文件夹的路径。")

    args = parser.parse_args()
    
    count_categories(Path(args.input_dir))