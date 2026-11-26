#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TARGET_SIZE = 1000.0

def normalize_bbox(bbox: list, original_width: int, original_height: int) -> list:
    """
    将bbox坐标从原始尺寸归一化到1000x1000。
    
    Args:
        bbox (list): [x1, y1, x2, y2] 原始坐标。
        original_width (int): 原始图片宽度。
        original_height (int): 原始图片高度。
        
    Returns:
        list: 归一化后的整数坐标 [nx1, ny1, nx2, ny2]。
    """
    if not (original_width > 0 and original_height > 0):
        return bbox # 避免除以零错误

    scale_x = TARGET_SIZE / original_width
    scale_y = TARGET_SIZE / original_height
    
    x1, y1, x2, y2 = bbox
    
    nx1 = int(round(x1 * scale_x))
    ny1 = int(round(y1 * scale_y))
    nx2 = int(round(x2 * scale_x))
    ny2 = int(round(y2 * scale_y))
    
    return [nx1, ny1, nx2, ny2]

def main():
    """
    主函数，读取、处理并保存归一化后的JSON数据。
    """
    parser = argparse.ArgumentParser(description="将VLM训练数据中的bbox归一化到1000x1000。")
    parser.add_argument(
        "--input_file",
        type=Path,
        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified.json",
        help="输入的训练数据JSON文件路径。"
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified_normalized.json",
        help="输出的归一化后的JSON文件路径。"
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        logging.error(f"输入文件不存在: {args.input_file}")
        return
        
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"无法读取或解析输入文件 {args.input_file}: {e}")
        return

    normalized_data = []
    
    logging.info(f"开始处理 {len(original_data)} 条数据...")

    for sample in tqdm(original_data, desc="归一化Bbox"):
        # 假设每个样本只有一个图片，并且在 'images' 列表的第一个位置
        if not sample.get("images") or not sample["images"][0]:
            logging.warning("样本中找不到图片路径，已跳过。")
            normalized_data.append(sample)
            continue
            
        image_path = Path(sample["images"][0])
        
        try:
            with Image.open(image_path) as img:
                original_width, original_height = img.size
        except FileNotFoundError:
            logging.warning(f"图片文件未找到，已跳过样本: {image_path}")
            continue
        except Exception as e:
            logging.error(f"打开图片失败 {image_path}: {e}")
            continue

        # 假设assistant的回答在第二个message中
        if len(sample.get("messages", [])) < 2:
            normalized_data.append(sample)
            continue
            
        assistant_message = sample["messages"][1]
        
        if assistant_message.get("role") != "assistant":
            normalized_data.append(sample)
            continue
            
        try:
            # assistant的内容是一个JSON字符串，需要先解析
            assistant_content_json = json.loads(assistant_message["content"])
            
            # 遍历并修改bbox
            if "event_analyses" in assistant_content_json:
                for analysis in assistant_content_json["event_analyses"]:
                    if "bbox" in analysis and isinstance(analysis["bbox"], list) and len(analysis["bbox"]) == 4:
                        analysis["bbox"] = normalize_bbox(
                            analysis["bbox"], original_width, original_height
                        )
            
            # 将修改后的JSON对象转换回字符串，并保持格式
            assistant_message["content"] = json.dumps(
                assistant_content_json, ensure_ascii=False, indent=2
            )
            
        except json.JSONDecodeError:
            # 如果内容不是有效的JSON，则不作处理
            pass
        except Exception as e:
            logging.error(f"处理样本时发生未知错误 (图片: {image_path}): {e}")

        normalized_data.append(sample)

    # 保存新的JSON文件
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(normalized_data, f, ensure_ascii=False, indent=2)
        logging.info(f"处理完成！归一化后的数据已保存到: {args.output_file}")
    except IOError as e:
        logging.error(f"写入输出文件失败 {args.output_file}: {e}")

if __name__ == "__main__":
    main()