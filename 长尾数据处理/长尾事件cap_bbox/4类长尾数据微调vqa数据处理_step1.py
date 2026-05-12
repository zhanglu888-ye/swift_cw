#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =======================
#     VQA训练数据配置
# =======================

KEYWORD_MAPPING = [
    # 精确匹配
    ("垃圾桶满溢", "垃圾桶满溢"),
    ("垃圾堆积", "垃圾堆积"),
    ("流动摊贩", "流动摊贩"),
    ("出店经营", "出店经营"),
    # 关键词/别名匹配
    ("小商贩", "流动摊贩"),
    ("摊贩", "流动摊贩"),
    ("商户", "出店经营"),
    ("出店", "出店经营"),
    ("满溢", "垃圾桶满溢"),
    ("堆积", "垃圾堆积"),
    # 兜底通用词，注意顺序，放在最后
    ("占道经营", "流动摊贩"), 
]

VQA_TRAINING_CONFIG: Dict[str, Dict[str, Any]] = {
    "垃圾事件": {
        "sub_categories": {"垃圾堆积", "垃圾桶满溢"},
        "location_question": "图片中发生垃圾事件的具体位置是哪里？"
    },
    "占道经营事件": {
        "sub_categories": {"流动摊贩", "出店经营"},
        "location_question": "图片中发生占道经营的具体位置是哪里？"
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

def extract_event_type_from_answer(answer: str) -> Optional[str]:
    for keyword, category in KEYWORD_MAPPING:
        if keyword in answer:
            return category
    return None

def parse_bbox_string(bbox_str: str) -> str:
    try:
        coords = [int(float(c)) for c in bbox_str.split()]
        if len(coords) == 4:
            return f"<{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}>"
    except (ValueError, IndexError):
        pass
    return "<bbox解析错误>"

def crop_and_save_image(
    large_image_path: Path, 
    bbox_str: str, 
    output_path: Path
) -> Optional[Path]:
    try:
        coords = [int(float(c)) for c in bbox_str.split()]
        if len(coords) != 4: return None
        x1, y1, x2, y2 = coords
        
        with Image.open(large_image_path) as img:
            width, height = img.size
            box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if box[0] >= box[2] or box[1] >= box[3]: return None
                
            cropped_img = img.crop(box)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if cropped_img.mode == 'RGBA':
                cropped_img = cropped_img.convert('RGB')
            
            cropped_img.save(output_path)
            return output_path
            
    except Exception as e:
        logging.error(f"裁剪图片失败: {large_image_path} with bbox {bbox_str} - {e}")
        return None

# =======================
#        主流程
# =======================

def main():
    parser = argparse.ArgumentParser(description="生成VQA训练数据，构建多轮对话并精确绑定图片上下文。")
    # ... 参数定义 ...
    parser.add_argument("--train_vqa_json_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/train/04_vqa")
    parser.add_argument("--large_image_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/large_images")
    parser.add_argument("--output_crop_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/train_pro/generated_crops")
    parser.add_argument("--output_file", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/train_pro/training_data_vqa.json")
    args = parser.parse_args()

    # ... 路径验证 ...
    for p in [args.train_vqa_json_dir, args.large_image_dir]:
        if not p.is_dir(): logging.error(f"目录不存在: {p}"); return
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_crop_dir.mkdir(parents=True, exist_ok=True)

    all_training_samples = []
    
    # 1. 统一收集所有case的数据
    vqa_json_files = list(args.train_vqa_json_dir.rglob("*.json"))
    cases_data = defaultdict(lambda: {"items": [], "primary_event_type": None})
    for path in vqa_json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            case_id = data.get("case_id")
            if not case_id: continue
            if not cases_data[case_id]["primary_event_type"]:
                cases_data[case_id]["primary_event_type"] = data.get("event_type")
            cases_data[case_id]["items"].extend(data.get("items", []))
        except Exception as e:
            logging.warning(f"处理文件 {path.name} 失败: {e}")

    logging.info(f"发现 {len(cases_data)} 个唯一的 case 需要处理。")

    # 2. 以 case_id 为单位构建对话
    for case_id, data in cases_data.items():
        large_image_path = find_large_image(args.large_image_dir, case_id)
        if not large_image_path: continue

        primary_event_type = data["primary_event_type"]
        if primary_event_type not in VQA_TRAINING_CONFIG: continue

        unique_items = {json.dumps(item, sort_keys=True): item for item in data["items"]}.values()
        
        # --- 核心修正：预处理所有item，生成唯一ID和路径 ---
        processed_items = []
        crop_counters = defaultdict(int) # 为每个子类别维护一个计数器
        
        for item in unique_items:
            qa_pairs = item.get("data", {}).get("qa_pairs", [])
            if not qa_pairs: continue
            
            first_answer = qa_pairs[0].get("answer", "")
            sub_category = extract_event_type_from_answer(first_answer)
            if not sub_category:
                logging.warning(f"[{case_id}] 无法从答案'{first_answer[:20]}...'中提取子类别。")
                continue

            crop_counters[sub_category] += 1
            new_crop_id = f"crop_{crop_counters[sub_category]}"
            unique_crop_filename = f"{sub_category}_{new_crop_id}.jpg"
            
            processed_items.append({
                "sub_category": sub_category,
                "unique_filename": unique_crop_filename,
                "qa_pairs": qa_pairs,
                "bbox_str": item.get("data", {}).get("original_bbox", "")
            })

        if not processed_items: continue

        # --- 聚合信息，为前两轮对话做准备 ---
        event_types_found = set(p_item["sub_category"] for p_item in processed_items)
        locations_by_type = defaultdict(set)
        for p_item in processed_items:
            if p_item["bbox_str"]:
                locations_by_type[p_item["sub_category"]].add(parse_bbox_string(p_item["bbox_str"]))

        # --- 构建对话 ---
        conversation_messages = [{"role": "system", "content": "You are a helpful assistant."}]

        # 对话第1轮: 事件检测
        first_question = processed_items[0]["qa_pairs"][0]["question"]
        merged_answer = f"是的，图中可以观察到明显的{'和'.join(sorted(list(event_types_found)))}现象。"
        conversation_messages.extend([
            {"role": "user", "content": f"<images>{first_question}", "images": [str(large_image_path.resolve())]},
            {"role": "assistant", "content": merged_answer}
        ])

        # 对话第2轮: 位置询问
        location_q = VQA_TRAINING_CONFIG[primary_event_type]["location_question"]
        location_a_parts = [f"发生{sub_type}的具体位置是{'、'.join(sorted(list(bboxes)))}" for sub_type, bboxes in sorted(locations_by_type.items()) if bboxes]
        location_a = "，".join(location_a_parts) + "。"
        conversation_messages.extend([
            {"role": "user", "content": location_q},
            {"role": "assistant", "content": location_a}
        ])

        # 对话后续轮次: 逐一切片的详细问答
        for p_item in processed_items:
            bbox_str = p_item["bbox_str"]
            unique_filename = p_item["unique_filename"]
            qa_pairs = p_item["qa_pairs"]

            if not bbox_str or len(qa_pairs) <= 1: continue

            new_crop_output_path = args.output_crop_dir / case_id / unique_filename
            saved_crop_path = crop_and_save_image(large_image_path, bbox_str, new_crop_output_path)
            
            if not saved_crop_path:
                logging.warning(f"[{case_id}/{unique_filename}] 因裁剪失败，跳过。")
                continue
            
            crop_image_path_list = [str(saved_crop_path.resolve())]
            
            for qa in qa_pairs[1:]:
                question, answer = qa.get("question", ""), qa.get("answer", "")
                if not question or not answer: continue
                
                conversation_messages.extend([
                    {"role": "user", "content": f"<images>{question}", "images": crop_image_path_list},
                    {"role": "assistant", "content": answer}
                ])
        
        # 组装最终样本
        all_training_samples.append({"messages": conversation_messages})
        logging.info(f"[{case_id}] 处理完成，生成一个包含 {len(conversation_messages)} 条消息的对话。")

    logging.info(f"全部处理完成，共生成 {len(all_training_samples)} 个多轮对话样本。")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_training_samples, f, ensure_ascii=False, indent=2)
    logging.info(f"VQA训练数据已保存到: {args.output_file}")
    logging.info(f"所有新生成的裁剪图已保存到: {args.output_crop_dir}")

if __name__ == "__main__":
    main()