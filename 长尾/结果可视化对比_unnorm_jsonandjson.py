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

def parse_and_group_data(all_data):
    image_data_map = defaultdict(lambda: {'label_bboxes': [], 'model_bboxes': [], 'label_descs': [], 'model_descs': []})
    for data_entry in tqdm(all_data, desc="Grouping data by image"):
        image_path = data_entry.get('images', [{}])[0].get('path')
        if not image_path: continue
        is_object_detection_task = data_entry.get("objects") is not None
        if is_object_detection_task:
            try:
                bboxes = data_entry["objects"].get("bbox", [])
                if bboxes: image_data_map[image_path]['label_bboxes'].extend(bboxes)
                
                response_str = data_entry.get("response", "")
                matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', response_str)
                for match in matches:
                    x_a, y_a, x_b, y_b = map(int, match)
                    
                    # *** 核心修复：对坐标进行归一化，确保是 (min_x, min_y, max_x, max_y) ***
                    x0 = min(x_a, x_b)
                    y0 = min(y_a, y_b)
                    x1 = max(x_a, x_b)
                    y1 = max(y_a, y_b)
                    # *** 修复结束 ***
                    
                    image_data_map[image_path]['model_bboxes'].append([x0, y0, x1, y1])
            except Exception as e: print(f"解析BBox时出错 for {image_path}: {e}")
        else:
            try:
                def extract_descs(json_str):
                    descs = []
                    if json_str and json_str != "<bbox>":
                        content = json.loads(json_str)
                        for analysis in content.get("event_analyses", []):
                            assessment = analysis.get("comprehensive_assessment", {})
                            cot = analysis.get("simplified_cot", {})
                            descs.append(f"类型: {assessment.get('异常类型', 'N/A')}, 风险: {assessment.get('风险等级', 'N/A')}\n分析: {cot.get('画面分析', 'N/A')}")
                    return descs
                image_data_map[image_path]['label_descs'].extend(extract_descs(data_entry.get('labels')))
                image_data_map[image_path]['model_descs'].extend(extract_descs(data_entry.get('response')))
            except (json.JSONDecodeError, KeyError) as e: print(f"解析描述时出错 for {image_path}: {e}")
    return image_data_map

def draw_text_on_image(title, bboxes, descs, font, title_color):
    w, h = TARGET_VIZ_SIZE
    img = Image.new('RGB', (w, h), color='white')
    draw = ImageDraw.Draw(img)
    y_text = 15
    draw.text((15, y_text), title, font=font, fill=title_color)
    y_text += FONT_SIZE + 10
    draw.text((15, y_text), "坐标框 (BBoxes):", font=font, fill=TEXT_COLOR)
    y_text += FONT_SIZE + 5
    if bboxes:
        for i, bbox in enumerate(bboxes):
            draw.text((25, y_text), f"- BBox {i+1}: {bbox}", font=font, fill=TEXT_COLOR)
            y_text += FONT_SIZE + 5
    else:
        draw.text((25, y_text), "N/A", font=font, fill=TEXT_COLOR)
        y_text += FONT_SIZE + 5
    y_text += 10
    draw.text((15, y_text), "描述分析:", font=font, fill=TEXT_COLOR)
    y_text += FONT_SIZE + 5
    if descs:
        for i, desc in enumerate(descs):
            draw.text((25, y_text), f"- 描述 {i+1}:", font=font, fill=TEXT_COLOR)
            y_text += FONT_SIZE + 5
            for line in desc.split('\n'):
                draw.text((35, y_text), line, font=font, fill=TEXT_COLOR)
                y_text += FONT_SIZE + 5
            y_text += 5
    else:
        draw.text((25, y_text), "N/A", font=font, fill=TEXT_COLOR)
        y_text += FONT_SIZE + 5
    return img

def calculate_iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(box_a_area + box_b_area - inter_area)
    return inter_area / union_area if union_area != 0 else 0

def update_stats_for_image(data, stats_collectors):
    if data['label_bboxes'] or data['model_bboxes']:
        gt_boxes = data['label_bboxes_converted']
        pred_boxes = data['model_bboxes']
        
        pred_matches = [False] * len(pred_boxes)
        image_tp = 0
        
        for gt_box in gt_boxes:
            best_iou = 0
            best_pred_idx = -1
            for i, pred_box in enumerate(pred_boxes):
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i
            
            if best_iou >= IOU_THRESHOLD and best_pred_idx != -1 and not pred_matches[best_pred_idx]:
                image_tp += 1
                pred_matches[best_pred_idx] = True
        
        image_fn = len(gt_boxes) - image_tp
        image_fp = len(pred_boxes) - image_tp

        stats_collectors['detection']['tp'] += image_tp
        stats_collectors['detection']['fn'] += image_fn
        stats_collectors['detection']['fp'] += image_fp

    if data['label_descs'] and data['model_descs']:
        label_desc = data['label_descs'][0]
        model_desc = data['model_descs'][0]
        
        label_type_match = re.search(r"类型:\s*([\w\s-]+),", label_desc)
        model_type_match = re.search(r"类型:\s*([\w\s-]+),", model_desc)
        
        if label_type_match and model_type_match:
            label_type = label_type_match.group(1).strip()
            model_type = model_type_match.group(1).strip()
            
            stats_collectors['classification']['overall']['total'] += 1
            if label_type in stats_collectors['classification']['per_category']:
                stats_collectors['classification']['per_category'][label_type]['total'] += 1

            if label_type == model_type:
                stats_collectors['classification']['overall']['correct'] += 1
                if label_type in stats_collectors['classification']['per_category']:
                    stats_collectors['classification']['per_category'][label_type]['correct'] += 1

def print_final_report(stats_collectors):
    print("\n\n" + "="*50)
    print(" " * 18 + "评估报告")
    print("="*50)

    print("\n--- 目标检测指标 (整体) ---")
    det_stats = stats_collectors['detection']
    tp, fp, fn = det_stats['tp'], det_stats['fp'], det_stats['fn']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  IoU 阈值: {IOU_THRESHOLD}")
    print(f"  真阳性 (TP): {tp}")
    print(f"  假阳性 (FP): {fp}")
    print(f"  假阴性 (FN): {fn}")
    print(f"  精确率 (Precision): {precision:.2%}")
    print(f"  召回率 (Recall):    {recall:.2%}")

    print("\n--- 分类任务指标 (准确率) ---")
    cls_overall = stats_collectors['classification']['overall']
    overall_acc = cls_overall['correct'] / cls_overall['total'] if cls_overall['total'] > 0 else 0
    print(f"  总体准确率: {overall_acc:.2%} ({cls_overall['correct']}/{cls_overall['total']})")
    
    print("\n  分项准确率:")
    for category, stats in stats_collectors['classification']['per_category'].items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"    - {category:<10}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    print("\n" + "="*50)

def main(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_data = load_json_lines(input_file)
    if not all_data: return
    try: font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError: font = ImageFont.load_default()
    
    image_data_map = parse_and_group_data(all_data)
    
    stats_collectors = {
        'detection': {'tp': 0, 'fp': 0, 'fn': 0},
        'classification': {
            'overall': {'correct': 0, 'total': 0},
            'per_category': {cat: {'correct': 0, 'total': 0} for cat in CLASSIFICATION_CATEGORIES}
        }
    }

    print("\n开始生成可视化图并计算指标...")
    for image_path, combined_data in tqdm(image_data_map.items()):
        if not os.path.exists(image_path): continue
        original_img_pil = Image.open(image_path).convert("RGB")
        w, h = original_img_pil.size
        
        label_bboxes_converted = []
        for bbox_norm in combined_data['label_bboxes']:
            x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm
            label_bboxes_converted.append([int((x1_norm/1000.0)*w), int((y1_norm/1000.0)*h), int((x2_norm/1000.0)*w), int((y2_norm/1000.0)*h)])
        combined_data['label_bboxes_converted'] = label_bboxes_converted
        
        update_stats_for_image(combined_data, stats_collectors)

        label_viz_img = original_img_pil.copy()
        draw_label = ImageDraw.Draw(label_viz_img)
        for bbox in label_bboxes_converted:
            draw_label.rectangle(bbox, outline=tuple(reversed(LABEL_COLOR)), width=3)
        
        model_viz_img = original_img_pil.copy()
        draw_model = ImageDraw.Draw(model_viz_img)
        for bbox in combined_data['model_bboxes']:
            draw_model.rectangle(bbox, outline=tuple(reversed(MODEL_COLOR)), width=3)

        label_desc_img = draw_text_on_image("--- 标签 (Ground Truth) ---", label_bboxes_converted, combined_data['label_descs'], font, tuple(reversed(LABEL_COLOR)))
        model_desc_img = draw_text_on_image("--- 模型输出 (Prediction) ---", combined_data['model_bboxes'], combined_data['model_descs'], font, tuple(reversed(MODEL_COLOR)))

        final_w, final_h = TARGET_VIZ_SIZE[0] * 2, TARGET_VIZ_SIZE[1] * 2
        final_img = Image.new('RGB', (final_w, final_h), 'white')
        final_img.paste(label_viz_img.resize(TARGET_VIZ_SIZE, Image.LANCZOS), (0, 0))
        final_img.paste(label_desc_img, (TARGET_VIZ_SIZE[0], 0))
        final_img.paste(model_viz_img.resize(TARGET_VIZ_SIZE, Image.LANCZOS), (0, TARGET_VIZ_SIZE[1]))
        final_img.paste(model_desc_img, (TARGET_VIZ_SIZE[0], TARGET_VIZ_SIZE[1]))
        
        draw = ImageDraw.Draw(final_img)
        draw.line([(TARGET_VIZ_SIZE[0], 0), (TARGET_VIZ_SIZE[0], final_h)], fill=LINE_COLOR, width=2)
        draw.line([(0, TARGET_VIZ_SIZE[1]), (final_w, TARGET_VIZ_SIZE[1])], fill=LINE_COLOR, width=2)

        base_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, base_filename)
        final_img.save(output_path)
    
    print_final_report(stats_collectors)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从单个JSONL文件可视化模型输出与标签的对比图（支持多个检测和描述）。")
    parser.add_argument("--input_file", default="/host/ssddata/disk2/zhanglu/swift/4cls/test/qwen25_7b_lora_unnorm/result_0922.json", help="包含模型输出和标签的JSON Lines文件路径。")
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/swift/4cls/test/qwen25_7b_lora_unnorm/result_0922_vision", help="保存可视化结果的目录路径。")
    
    args = parser.parse_args()

    main(args.input_file, args.output_dir)
    print(f"\n处理完成！所有可视化结果已保存在 '{args.output_dir}' 目录中。")