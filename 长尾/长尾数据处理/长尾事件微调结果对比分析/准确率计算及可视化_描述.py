import json
import pandas as pd
from collections import defaultdict
import os
import re  # 导入正则
from PIL import Image, ImageDraw, ImageFont

# ================= 配置区 =================
VISUALIZE_ERRORS = False  # 是否开启错误图片可视化
ERROR_OUTPUT_DIR = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/test_show_tmp"  # 错误图片保存目录
FONT_PATH = "/host/ssddata/disk2/zhanglu/SimHei.ttf"  # 替换为你本地的ttf文件路径，例如 "simhei.ttf"
# ===========================================

def extract_type(json_input):
    """
    强化的异常类型提取函数：
    1. 支持标准解析
    2. 支持 Markdown 清理
    3. 支持正则表达式兜底（应对非法 JSON 结构）
    """
    if not json_input:
        return "空结果"
    
    # 尝试多种解析方式
    data = None
    
    # 如果是字符串，先尝试清理并解析
    if isinstance(json_input, str):
        # 1. 清理 Markdown 代码块
        cleaned = json_input.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```[a-z]*\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned)
        
        try:
            data = json.loads(cleaned)
        except:
            pass # 标准解析失败，进入后续兜底

    # 2. 如果解析成功，尝试标准结构查找
    if data and isinstance(data, dict):
        try:
            # 兼容有些层级在 event_analyses 下
            if "event_analyses" in data and len(data["event_analyses"]) > 0:
                inner = data["event_analyses"][0]
                if "comprehensive_assessment" in inner:
                    return inner["comprehensive_assessment"]["异常类型"]
            # 兼容直接就是 comprehensive_assessment 结构
            if "comprehensive_assessment" in data:
                return data["comprehensive_assessment"]["异常类型"]
        except:
            pass

    # 3. 兜底方案：正则表达式提取 (极其关键)
    # 即使 JSON 结构坏了，只要文本里有 "异常类型": "xxx" 就能抓到
    try:
        raw_str = str(json_input)
        # 匹配 "异常类型": "内容" 或 '异常类型': '内容'
        pattern = r'["\']异常类型["\']\s*:\s*["\']([^"\']+)["\']'
        match = re.search(pattern, raw_str)
        if match:
            return match.group(1).strip()
    except:
        pass

    return "解析失败"

def draw_multiline_text(draw, text, font, start_x, start_y, max_width, max_height):
    """自动换行绘制函数"""
    lines = []
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue
        current_line = ""
        for char in paragraph:
            test_line = current_line + char
            w = draw.textlength(test_line, font=font)
            if w <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = char
        lines.append(current_line)

    y_offset = start_y
    line_spacing = 5
    left, top, right, bottom = font.getbbox("测")
    single_line_height = (bottom - top) + line_spacing

    for line in lines:
        if y_offset + single_line_height > start_y + max_height:
            draw.text((start_x, y_offset), "... (内容过长已截断)", fill=(128, 128, 128), font=font)
            break
        draw.text((start_x, y_offset), line, fill=(0, 0, 0), font=font)
        y_offset += single_line_height

def visualize_error_case(image_path, response_data, labels_data, save_path):
    if not os.path.exists(image_path):
        return
    try:
        img = Image.open(image_path).convert("RGB")
        img_h = 1080  
        img_w = int(img.width * (img_h / img.height))
        img = img.resize((img_w, img_h))

        text_w = 1200  
        canvas_w = img_w + text_w
        canvas = Image.new("RGB", (canvas_w, img_h), (255, 255, 255))
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype(FONT_PATH, 16)
            title_font = ImageFont.truetype(FONT_PATH, 22)
        except:
            font = ImageFont.load_default()
            title_font = font

        def to_pretty_json(data):
            if not data: return ""
            try:
                # 尝试格式化
                obj = json.loads(data) if isinstance(data, str) else data
                return json.dumps(obj, indent=2, ensure_ascii=False)
            except:
                return str(data)

        res_text = to_pretty_json(response_data)
        lab_text = to_pretty_json(labels_data)

        margin = 20
        usable_text_w = text_w - (margin * 2)
        section_h = (img_h // 2) - 60

        # 上半部分：Response
        draw.text((img_w + margin, margin), "--- RESPONSE (PRED) ---", fill=(255, 0, 0), font=title_font)
        draw_multiline_text(draw, res_text, font, img_w + margin, margin + 40, usable_text_w, section_h)

        # 分割线
        draw.line([(img_w, img_h // 2), (canvas_w, img_h // 2)], fill=(150, 150, 150), width=2)

        # 下半部分：Labels
        draw.text((img_w + margin, (img_h // 2) + margin), "--- LABELS (GT) ---", fill=(0, 128, 0), font=title_font)
        draw_multiline_text(draw, lab_text, font, img_w + margin, (img_h // 2) + margin + 40, usable_text_w, section_h)

        canvas.save(save_path)
    except Exception as e:
        print(f"可视化绘图失败: {e}")

def get_data_from_file(file_path):
    pairs = []
    # garbage_group = {"垃圾堆积", "垃圾桶满溢", "垃圾事件"} 
    garbage_group = {}
    crowd_group = {"人群异常聚集", "开放空间异常围聚", "异质文化异常聚集"}

    print(f"正在读取文件: {file_path}")
    if VISUALIZE_ERRORS and not os.path.exists(ERROR_OUTPUT_DIR):
        os.makedirs(ERROR_OUTPUT_DIR)

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                raw_response = item.get("response", "")
                raw_labels = item.get("labels", "")
                
                label_type = extract_type(raw_labels)
                if label_type == "无异常":
                    continue

                pred_type = extract_type(raw_response)
                orig_pred_type = pred_type # 记录用于命名

                # 兼容逻辑
                if label_type in garbage_group and pred_type in garbage_group:
                    pred_type = label_type
                if label_type in crowd_group and pred_type in crowd_group:
                    pred_type = label_type

                pairs.append((label_type, pred_type))

                if VISUALIZE_ERRORS and label_type != pred_type:
                    images_info = item.get("images", [])
                    if images_info and isinstance(images_info, list):
                        img_path = images_info[0].get("path", "")
                        if img_path:
                            original_img_name = os.path.basename(img_path)
                            save_name = f"line{idx}_{original_img_name}_{label_type}_vs_{orig_pred_type}.jpg"
                            save_full_path = os.path.join(ERROR_OUTPUT_DIR, save_name)
                            visualize_error_case(img_path, raw_response, raw_labels, save_full_path)

            except Exception as e:
                print(f"解析行出错: {e}")
    return pairs

def calculate_combined_metrics(file_paths):
    all_results = []
    for path in file_paths:
        all_results.extend(get_data_from_file(path))
    
    total_samples = len(all_results)
    if total_samples == 0:
        print("没有提取到有效数据！")
        return

    categories = set()
    for label, pred in all_results:
        categories.add(label); categories.add(pred)
    
    for err_tag in ["解析失败", "未知", "空结果"]:
        if err_tag in categories: categories.remove(err_tag)
    
    sorted_categories = sorted(list(categories))
    final_stats = []
    for cat in sorted_categories:
        tp = 0; fp = 0; fn = 0
        total_in_label = 0
        
        # 统计 TP, FP, FN
        for label, pred in all_results:
            if label == cat:
                total_in_label += 1
                if pred == cat: tp += 1
                else: fn += 1
            else:
                if pred == cat: fp += 1
                # tn 不再需要用于计算 Precision/Recall/F1
        
        # 计算 Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # 计算 Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # 计算 F1-Score
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        final_stats.append({
            "类别": cat, 
            "Total": total_in_label, 
            "Precision": f"{precision:.2%}", 
            "Recall": f"{recall:.2%}",
            "F1-Score": f"{f1:.2%}", # 新增 F1
            "TP": tp, "FP": fp, "FN": fn
        })

    df = pd.DataFrame(final_stats)
    print("\n" + "="*60)
    print("合并统计结果 (Precision / Recall / F1)")
    print("="*60)
    print(df.to_markdown(index=False))

    correct_count = sum(1 for label, pred in all_results if label == pred)
    print(f"\n有效统计样本数: {total_samples}")
    print(f"总准确率 (Overall Accuracy): {correct_count/total_samples:.2%}")

if __name__ == "__main__":
    files_to_process = [
        "/host/ssddata/disk2/zhanglu/swift/4cls/result/train_260306cls3v1_epochs20/result_0309_jintao_ck1800.json"
    ]   
    calculate_combined_metrics(files_to_process)