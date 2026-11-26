import os
import json
import cv2
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse
from collections import defaultdict

# --- 配置区 ---

# 明确指定字体文件，请确保此文件与脚本在同一目录或提供完整路径
FONT_PATH = "/host/ssddata/disk2/zhanglu/SimHei.ttf"
FONT_SIZE = 20

# 颜色设置 (B, G, R)
LABEL_COLOR = (0, 255, 0)
MODEL_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 0, 0)
LINE_COLOR = (128, 128, 128)

# --- 评估与可视化配置 ---
TARGET_VIZ_SIZE = (1100, 800)
IOU_THRESHOLD = 0.4
CLASSIFICATION_CATEGORIES = ["出店经营", "流动摊贩", "垃圾堆积", "垃圾桶满溢", "人员倒地", "人员聚集", "无异常事件"]

# --- 脚本区 ---

def load_json_lines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: data.append(json.loads(line))
                except json.JSONDecodeError as e: print(f"警告: 跳过无法解析的行: {line.strip()}. 错误: {e}")
    return data

def parse_label_file(label_json_path):
    bboxes, descs = [], []
    try:
        with open(label_json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        for ann in data.get("annotations", []):
            category = ann.get("category", ["N/A"])[0]
            descs.append(f"类型: {category}")
            coords = ann.get("coordinate", [])
            if len(coords) == 4:
                x_coords, y_coords = [p['x'] for p in coords], [p['y'] for p in coords]
                bboxes.append([int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))])
    except Exception as e:
        print(f"警告: 解析标签文件 {label_json_path} 失败: {e}")
    return bboxes, descs

def extract_json_from_markdown(text):
    match = re.search(r'```json\s*\n(.*?)\n\s*```', text, re.DOTALL)
    return match.group(1).strip() if match else None

def group_and_parse_all_data(model_outputs, label_dir, stats_collectors):
    """核心函数：智能解析多种格式，聚合信息，并统计真值数量"""
    image_data_map = defaultdict(lambda: {'label_bboxes': [], 'model_bboxes': [], 'label_descs': [], 'model_descs': []})
    
    for data_entry in tqdm(model_outputs, desc="Grouping data by image"):

        image_path = data_entry.get('images', [{}])[0].get('path')
        if "882b_000" in image_path:
            print()        
        
        
        if not image_path: continue

        response_str = data_entry.get("response", "")
        
        # *** CORE FIX 1: 强大的、多格式智能解析器 ***
        try:
            json_str_md = extract_json_from_markdown(response_str)
            if json_str_md:
                parsed_json = json.loads(json_str_md)
                if isinstance(parsed_json, list):
                    for item in parsed_json:
                        if "bbox_2d" in item and len(item["bbox_2d"]) == 4:
                            image_data_map[image_path]['model_bboxes'].append(item["bbox_2d"])
                elif isinstance(parsed_json, dict):
                    if "event_analyses" in parsed_json:
                        for analysis in parsed_json.get("event_analyses", []):
                            assessment = analysis.get("comprehensive_assessment", {})
                            cot = analysis.get("simplified_cot", {})
                            image_data_map[image_path]['model_descs'].append(f"类型: {assessment.get('异常类型', 'N/A')}, 风险: {assessment.get('风险等级', 'N/A')}\n分析: {cot.get('画面分析', 'N/A')}")
                    elif "子事件" in parsed_json:
                        sub_events = parsed_json.get("子事件", ["N/A"])[0]
                        risk_level = parsed_json.get("风险等级", "N/A")
                        image_data_map[image_path]['model_descs'].append(f"类型: {sub_events}, 风险: {risk_level}")
            elif "<|box_start|>" in response_str:
                matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', response_str)
                for match in matches:
                    x_a, y_a, x_b, y_b = map(int, match)
                    image_data_map[image_path]['model_bboxes'].append([min(x_a, x_b), min(y_a, y_b), max(x_a, x_b), max(y_a, y_b)])
            elif response_str.strip().startswith('{'):
                content = json.loads(response_str)
                for analysis in content.get("event_analyses", []):
                    assessment = analysis.get("comprehensive_assessment", {})
                    cot = analysis.get("simplified_cot", {})
                    image_data_map[image_path]['model_descs'].append(f"类型: {assessment.get('异常类型', 'N/A')}, 风险: {assessment.get('风险等级', 'N/A')}\n分析: {cot.get('画面分析', 'N/A')}")
            else:
                if response_str: image_data_map[image_path]['model_descs'].append(response_str)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析模型响应时出错 for {image_path}, response: '{response_str[:100]}...'. 错误: {e}")
            
        # --- 查找、解析并统计标签文件 (只执行一次) ---
        if not image_data_map[image_path]['label_bboxes'] and not image_data_map[image_path]['label_descs']:
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            label_json_path = os.path.join(label_dir, f"{image_basename}.json")
            if os.path.exists(label_json_path):
                label_bboxes, label_descs = parse_label_file(label_json_path)
                image_data_map[image_path]['label_bboxes'] = label_bboxes
                image_data_map[image_path]['label_descs'] = label_descs
                stats_collectors['ground_truth_counts']['detection']['total'] += len(label_bboxes)
                if label_descs:
                    unique_types_in_image = set()
                    for desc in label_descs:
                        type_match = re.search(r"类型:\s*([^,]+)", desc)
                        if type_match: unique_types_in_image.add(type_match.group(1).strip())
                    for label_type in unique_types_in_image:
                        if label_type in stats_collectors['ground_truth_counts']['classification']:
                             stats_collectors['ground_truth_counts']['classification'][label_type] += 1
            else: print(f"警告: 找不到对应的标签文件: {label_json_path}")

    return image_data_map

def draw_text_on_image(title, bboxes, descs, font, title_color):
    w, h = TARGET_VIZ_SIZE; img = Image.new('RGB', (w, h), color='white'); draw = ImageDraw.Draw(img)
    y_text = 15; draw.text((15, y_text), title, font=font, fill=title_color); y_text += FONT_SIZE + 10
    draw.text((15, y_text), "坐标框 (BBoxes):", font=font, fill=TEXT_COLOR); y_text += FONT_SIZE + 5
    if bboxes:
        for i, bbox in enumerate(bboxes): draw.text((25, y_text), f"- BBox {i+1}: {bbox}", font=font, fill=TEXT_COLOR); y_text += FONT_SIZE + 5
    else: draw.text((25, y_text), "N/A", font=font, fill=TEXT_COLOR); y_text += FONT_SIZE + 5
    y_text += 10; draw.text((15, y_text), "描述分析:", font=font, fill=TEXT_COLOR); y_text += FONT_SIZE + 5
    if descs:
        for i, desc in enumerate(descs):
            lines = []
            if '\n' in desc:
                lines.extend(desc.split('\n'))
            else: 
                max_chars = 40
                for j in range(0, len(desc), max_chars):
                    lines.append(desc[j:j+max_chars])
            
            draw.text((25, y_text), f"- 描述 {i+1}:", font=font, fill=TEXT_COLOR); y_text += FONT_SIZE + 5
            for line in lines:
                draw.text((35, y_text), line, font=font, fill=TEXT_COLOR)
                y_text += FONT_SIZE + 5
            y_text += 5
    else: draw.text((25, y_text), "N/A", font=font, fill=TEXT_COLOR); y_text += FONT_SIZE + 5
    return img

def calculate_iou(box_a, box_b):
    xA=max(box_a[0],box_b[0]); yA=max(box_a[1],box_b[1]); xB=min(box_a[2],box_b[2]); yB=min(box_a[3],box_b[3])
    inter_area=max(0,xB-xA)*max(0,yB-yA); box_a_area=(box_a[2]-box_a[0])*(box_a[3]-box_a[1]); box_b_area=(box_b[2]-box_b[0])*(box_b[3]-box_b[1])
    union_area=float(box_a_area+box_b_area-inter_area); return inter_area/union_area if union_area!=0 else 0

def update_stats_for_image(data, stats_collectors):
    if data['label_bboxes'] or data['model_bboxes']:
        gt_boxes, pred_boxes = data['label_bboxes'], data['model_bboxes']
        pred_matches=[False]*len(pred_boxes); image_tp=0
        for gt_box in gt_boxes:
            best_iou,best_pred_idx=0,-1
            for i,pred_box in enumerate(pred_boxes):
                iou=calculate_iou(gt_box,pred_box)
                if iou>best_iou: best_iou,best_pred_idx=iou,i
            if best_iou>=IOU_THRESHOLD and best_pred_idx!=-1 and not pred_matches[best_pred_idx]: image_tp+=1; pred_matches[best_pred_idx]=True
        stats_collectors['detection']['tp']+=image_tp; stats_collectors['detection']['fn']+=len(gt_boxes)-image_tp; stats_collectors['detection']['fp']+=len(pred_boxes)-image_tp
    
    # 仅当存在真值标签时，才进行分类评估
    if data['label_descs']:
        label_desc = data['label_descs'][0]
        model_desc = data['model_descs'][0] if data['model_descs'] else "" # 如果模型无输出，则为空字符串

        label_type_match=re.search(r"类型:\s*([^,\n]+)",label_desc)
        model_type_match=re.search(r"类型:\s*([^,\n]+)",model_desc)
        
        if label_type_match:
            label_type = label_type_match.group(1).strip()
            # 即使模型没有匹配的输出，也要计入分母（在报告中体现）
            # stats_collectors['classification']['overall']['total']+=1 # 这一行不再需要，因为我们从ground_truth_counts获取总数

            if label_type in stats_collectors['classification']['per_category']: 
                # stats_collectors['classification']['per_category'][label_type]['total_predictions']+=1 # 也不需要了
                pass
            
            if model_type_match:
                model_type = model_type_match.group(1).strip()
                if label_type == model_type:
                    stats_collectors['classification']['overall']['correct']+=1
                    if label_type in stats_collectors['classification']['per_category']: 
                        stats_collectors['classification']['per_category'][label_type]['correct']+=1

def print_final_report(stats):
    print("\n\n"+"="*50+"\n"+" "*18+"评估报告\n"+"="*50); 
    print("\n--- 目标检测指标 (整体) ---")
    d=stats['detection']; p=d['tp']/(d['tp']+d['fp']) if (d['tp']+d['fp'])>0 else 0; r=d['tp']/(d['tp']+d['fn']) if (d['tp']+d['fn'])>0 else 0
    total_gt_boxes = stats['ground_truth_counts']['detection']['total']
    print(f"  总计真值框 (Total GT BBoxes): {total_gt_boxes}")
    print(f"  IoU 阈值: {IOU_THRESHOLD}\n  真阳性 (TP): {d['tp']}\n  假阳性 (FP): {d['fp']}\n  假阴性 (FN): {d['fn']}")
    print(f"  精确率 (Precision): {p:.2%}\n  召回率 (Recall):    {r:.2%}"); 
    print("\n--- 分类任务指标 (准确率) ---")

    # *** 修改点：使用正确的总样本数作为分母 ***
    correct_predictions = stats['classification']['overall']['correct']
    # 分母是所有分类真值的总和
    total_classification_samples = sum(stats['ground_truth_counts']['classification'].values())
    
    accuracy = correct_predictions / total_classification_samples if total_classification_samples > 0 else 0
    print(f"  总体准确率: {accuracy:.2%} ({correct_predictions}/{total_classification_samples})")
    
    print("\n  各类别真值样本统计 (用于计算准确率分母):")
    for category, count in stats['ground_truth_counts']['classification'].items():
        if count > 0: print(f"    - {category:<10}: {count} 个样本")
        
    print("\n  分项准确率:")
    for cat, s in stats['classification']['per_category'].items():
        # 使用正确的真值总数作为分母
        total_gt_samples_per_cat = stats['ground_truth_counts']['classification'][cat]
        if total_gt_samples_per_cat > 0: 
            acc = s['correct'] / total_gt_samples_per_cat
            print(f"    - {cat:<10}: {acc:.2%} ({s['correct']}/{total_gt_samples_per_cat})")
    print("\n"+"="*50)


def main(input_file, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_outputs = load_json_lines(input_file)
    if not model_outputs: return
    try: font = ImageFont.truetype(FONT_PATH, FONT_SIZE); print(f"成功加载字体: '{FONT_PATH}'")
    except IOError: font = ImageFont.load_default(); print(f"警告: 找不到字体 '{FONT_PATH}', 使用默认字体。")
    
    stats_collectors = { 'detection': defaultdict(int), 'classification': {'overall': defaultdict(int), 'per_category': {cat: defaultdict(int) for cat in CLASSIFICATION_CATEGORIES}}, 'ground_truth_counts': {'detection': defaultdict(int), 'classification': defaultdict(int, {cat: 0 for cat in CLASSIFICATION_CATEGORIES})}}
    
    image_data_map = group_and_parse_all_data(model_outputs, label_dir, stats_collectors)
    
    print("\n开始生成可视化图并计算指标...")
    for image_path, combined_data in tqdm(image_data_map.items(), desc="Visualizing and Evaluating"):
        if "882b_000" in image_path:
            print()
        if not os.path.exists(image_path): 
            print(f"警告: 图像文件不存在，跳过: {image_path}")
            continue
        update_stats_for_image(combined_data, stats_collectors)

        original_img_pil = Image.open(image_path).convert("RGB")
        label_viz_img = original_img_pil.copy()
        draw_label = ImageDraw.Draw(label_viz_img)
        for bbox in combined_data['label_bboxes']: draw_label.rectangle(bbox, outline=tuple(reversed(LABEL_COLOR)), width=3)
        model_viz_img = original_img_pil.copy()
        draw_model = ImageDraw.Draw(model_viz_img)
        for bbox in combined_data['model_bboxes']: draw_model.rectangle(bbox, outline=tuple(reversed(MODEL_COLOR)), width=3)
        label_desc_img = draw_text_on_image("--- 标签 (Ground Truth) ---", combined_data['label_bboxes'], combined_data['label_descs'], font, tuple(reversed(LABEL_COLOR)))
        model_desc_img = draw_text_on_image("--- 模型输出 (Prediction) ---", combined_data['model_bboxes'], combined_data['model_descs'], font, tuple(reversed(MODEL_COLOR)))
        final_w, final_h = TARGET_VIZ_SIZE[0] * 2, TARGET_VIZ_SIZE[1] * 2
        final_img = Image.new('RGB', (final_w, final_h), 'white')
        final_img.paste(label_viz_img.resize(TARGET_VIZ_SIZE, Image.Resampling.LANCZOS), (0, 0))
        final_img.paste(label_desc_img, (TARGET_VIZ_SIZE[0], 0))
        final_img.paste(model_viz_img.resize(TARGET_VIZ_SIZE, Image.Resampling.LANCZOS), (0, TARGET_VIZ_SIZE[1]))
        final_img.paste(model_desc_img, (TARGET_VIZ_SIZE[0], TARGET_VIZ_SIZE[1]))
        draw = ImageDraw.Draw(final_img)
        draw.line([(TARGET_VIZ_SIZE[0], 0), (TARGET_VIZ_SIZE[0], final_h)], fill=LINE_COLOR, width=2)
        draw.line([(0, TARGET_VIZ_SIZE[1]), (final_w, TARGET_VIZ_SIZE[1])], fill=LINE_COLOR, width=2)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        final_img.save(output_path)
    
    print_final_report(stats_collectors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从模型输出和标签目录可视化并评估模型性能。")
    parser.add_argument("--label_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/05_label_tmp", help="包含真实标签的根目录路径。")
    parser.add_argument("--input_file", default="/host/ssddata/disk2/zhanglu/swift/4cls/test/qwen25_7b_unnorm_original/result_0923_original_model_加定义.json", help="包含模型输出的JSON Lines文件路径。")
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/swift/4cls/test/qwen25_7b_unnorm_original/result_0923_vision_bbox_加定义", help="保存可视化结果的目录路径。")
      
    # parser.add_argument("--label_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/05_label_tmp", help="包含真实标签的根目录路径。")
    # parser.add_argument("--input_file", default="/host/ssddata/disk2/zhanglu/swift/4cls/test/qwen25_7b_lora_unnorm/result_0922.json", help="包含模型输出的JSON Lines文件路径。")
    # parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/swift/4cls/test/qwen25_7b_lora_unnorm/result_0923_vision_bbox", help="保存可视化结果的目录路径。")
        
    
    args = parser.parse_args()
    main(args.input_file, args.label_dir, args.output_dir)