#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import math
import random
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main():
    parser = argparse.ArgumentParser(
        description="将包含01-04目录的数据集按case_id划分为训练集和测试集，并保持目录结构。"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_appdata_0916",
        help="包含01_original到04_vqa的输入根目录。"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset",
        help="存放划分后train和test集的输出根目录。"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.9,
        help="训练集所占的比例。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，确保每次划分结果一致。"
    )
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 定义源目录和目标目录
    train_dir = args.output_dir / "train"
    test_dir = args.output_dir / "test"

    if args.input_dir == args.output_dir:
        logging.error("输入目录和输出目录不能相同，请指定不同的输出目录。")
        return
        
    if args.output_dir.exists():
        logging.warning(f"输出目录 {args.output_dir} 已存在，可能会覆盖文件。")
    
    # 1. 识别所有唯一的 case_id
    # 我们以 01_original 目录为基准来收集所有 case_id
    original_dir = args.input_dir / "01_original"
    if not original_dir.is_dir():
        logging.error(f"基准目录 {original_dir} 不存在，无法识别case。")
        return

    all_case_ids = set()
    # 遍历所有事件子目录下的图片文件
    for image_path in original_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            all_case_ids.add(image_path.stem)
    
    if not all_case_ids:
        logging.error("在基准目录中没有找到任何图片文件，无法进行划分。")
        return

    case_id_list = sorted(list(all_case_ids))
    logging.info(f"共发现 {len(case_id_list)} 个唯一的 case_id。")

    # 2. 随机打乱并划分 case_id 列表
    random.shuffle(case_id_list)
    split_index = math.ceil(len(case_id_list) * args.split_ratio)
    
    # 使用集合以获得 O(1) 的查找效率
    train_ids = set(case_id_list[:split_index])
    test_ids = set(case_id_list[split_index:])

    logging.info(f"划分数据集：训练集 {len(train_ids)} 个，测试集 {len(test_ids)} 个。")

    # 3. 遍历所有文件并移动
    top_level_dirs = ["01_original", "02_visualized", "03_vlm", "04_vqa"]
    files_moved_count = 0

    logging.info("开始移动文件...")
    for top_dir_name in top_level_dirs:
        source_top_dir = args.input_dir / top_dir_name
        if not source_top_dir.is_dir():
            logging.warning(f"源目录 {source_top_dir} 不存在，跳过。")
            continue

        for src_path in source_top_dir.rglob("*"):
            if not src_path.is_file():
                continue

            case_id = src_path.stem
            dest_base_dir = None

            if case_id in train_ids:
                dest_base_dir = train_dir
            elif case_id in test_ids:
                dest_base_dir = test_dir
            else:
                # 如果有文件的 case_id 不在我们的列表中，则跳过
                continue

            # 计算并创建目标路径
            relative_path = src_path.relative_to(args.input_dir)
            dest_path = dest_base_dir / relative_path
            
            # 确保目标文件夹存在
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 移动文件
            try:
                shutil.move(str(src_path), str(dest_path))
                files_moved_count += 1
            except Exception as e:
                logging.error(f"移动文件失败: 从 {src_path} 到 {dest_path} - {e}")
    
    logging.info(f"全部完成！共移动 {files_moved_count} 个文件。")
    logging.info(f"训练集数据已移动至: {train_dir}")
    logging.info(f"测试集数据已移动至: {test_dir}")


if __name__ == "__main__":
    main()