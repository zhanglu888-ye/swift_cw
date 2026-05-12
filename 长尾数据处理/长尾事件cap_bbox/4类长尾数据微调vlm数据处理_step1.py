#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =======================
#     训练数据配置
# =======================
TRAINING_CONFIG: Dict[str, Dict[str, Any]] = {
    "垃圾事件": {
        "sub_categories": {"垃圾堆积", "垃圾桶满溢"},
        "prompt_template": (
            "你是一个专业的AI视觉分析助手。你的任务是分析图像<image>中与“垃圾事件”相关的情况。\n"
            "请严格按照以下定义，识别图中的子事件，并评估其风险等级。\n\n"
            "子事件定义: \"垃圾堆积\", \"垃圾桶满溢\", \"无异常事件\"。\n"
            "风险等级定义: \"3级-高风险\", \"2级-中风险\", \"1级-低风险\", \"0级-无风险\"。\n\n"
            "请将你的分析结果以JSON对象格式输出。"
        ),
    },
    "占道经营事件": {
        "sub_categories": {"流动摊贩", "出店经营"},
        "prompt_template": (
            "你是一个专业的AI视觉分析助手。你的任务是分析图像<image>中与“占道经营事件”相关的情况。\n"
            "请严格按照以下定义，识别图中的子事件，并评估其风险等级。\n\n"
            "子事件定义: \"流动摊贩\", \"出店经营\", \"无异常事件\"。\n"
            "风险等级定义: \"3级-高风险\", \"2级-中风险\", \"1级-低风险\", \"0级-无风险\"。\n\n"
            "请将你的分析结果以JSON对象格式输出。"
        ),
    },
}

# =======================
#      辅助函数
# =======================
def find_large_image(large_image_dir: Path, case_id: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        img_path = large_image_dir / f"{case_id}{ext}"
        if img_path.exists():
            return img_path
    return None



##简单描述
def clean_analysis_entry(analysis: dict) -> dict:
    """
    移除不需要的字段，只保留训练所需的字段。
    """
    return {
        "simplified_cot": analysis.get("simplified_cot", {}),
        "comprehensive_assessment": analysis.get("comprehensive_assessment", {}),
        "bbox": analysis.get("bbox", [])
    }
    
##复杂描述
# def clean_analysis_entry(analysis: dict) -> dict:
#     """
#     移除不需要的字段，只保留训练所需的字段。
#     """
#     return {
#         "detailed_cot": analysis.get("detailed_cot", {}),
#         "comprehensive_assessment": analysis.get("comprehensive_assessment", {}),
#         "bbox": analysis.get("bbox", [])
#     }    
    
    

# =======================
#        主流程
# =======================
def main():
    parser = argparse.ArgumentParser(description="根据清洗后的VLM结果生成精简的训练数据。")
    parser.add_argument(
        "--large_image_dir", 
        type=Path,
        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/large_images", 
        help="大图所在的目录。"
    )
    parser.add_argument(
        "--train_vlm_json_dir", 
        type=Path,
        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/03_vlm",  
        help="VLM分析结果JSON所在的根目录 (例如 .../train/03_vlm)。"
    )
    parser.add_argument(
        "--output_file", 
        type=Path,
        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified.json", 
        help="最终输出的训练数据JSON文件路径。"
    )
    args = parser.parse_args()

    if not args.large_image_dir.is_dir():
        logging.error(f"大图目录不存在: {args.large_image_dir}")
        return
    if not args.train_vlm_json_dir.is_dir():
        logging.error(f"训练集VLM JSON目录不存在: {args.train_vlm_json_dir}")
        return

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    all_training_samples = []
    
    # --- 核心修正点：使用 rglob 进行递归搜索 ---
    train_json_files = list(args.train_vlm_json_dir.rglob("*.json"))
    logging.info(f"在训练集目录 {args.train_vlm_json_dir} 及其子目录中发现 {len(train_json_files)} 个JSON文件，开始处理...")

    for json_path in train_json_files:
        case_id = json_path.stem

        large_image_path = find_large_image(args.large_image_dir, case_id)
        if not large_image_path:
            logging.warning(f"[{case_id}] 找不到对应的大图，跳过。")
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                vlm_data = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"[{case_id}] VLM JSON解析失败，跳过: {json_path}")
            continue

        all_analyses = vlm_data.get("event_analyses", [])
        if not all_analyses:
            continue

        for event_type, config in TRAINING_CONFIG.items():
            target_sub_cats = config["sub_categories"]
            
            relevant_analyses = [
                analysis for analysis in all_analyses
                if analysis.get("comprehensive_assessment", {}).get("异常类型") in target_sub_cats
            ]

            if not relevant_analyses:
                continue
            
            cleaned_analyses = [clean_analysis_entry(analysis) for analysis in relevant_analyses]

            assistant_content = {"event_analyses": cleaned_analyses}
            assistant_content_str = json.dumps(assistant_content, ensure_ascii=False, indent=2)

            sample = {
                "messages": [
                    {"role": "user", "content": config["prompt_template"]},
                    {"role": "assistant", "content": assistant_content_str}
                ],
                "images": [str(large_image_path.resolve())]
            }
            all_training_samples.append(sample)

    logging.info(f"处理完成，共生成 {len(all_training_samples)} 条训练数据。")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_training_samples, f, ensure_ascii=False, indent=2)
    logging.info(f"最终训练数据已保存到: {args.output_file}")


if __name__ == "__main__":
    main()