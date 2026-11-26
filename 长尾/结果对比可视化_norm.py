import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse

# --- 配置区 ---

# 明确指定字体文件，请确保此文件与脚本在同一目录或提供完整路径
FONT_PATH = "/host/ssddata/disk2/zhanglu/SimHei.ttf"
FONT_SIZE = 20

# 颜色设置 (B, G, R)
LABEL_COLOR = (0, 255, 0)      # 绿色 for Ground Truth
MODEL_COLOR = (0, 0, 255)      # 红色 for Model Prediction
TEXT_COLOR = (0, 0, 0)         # 黑色 for Text
LINE_COLOR = (128, 128, 128)   # 灰色 for Dividers

# --- 脚本区 ---

def load_json_lines(file_path):
    """加载JSON Lines文件，每行一个JSON对象"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"警告: 跳过无法解析的行: {line.strip()}. 错误: {e}")
    return data

def parse_data(data_entry, source_type='label'):
    """从单个数据条目中解析出需要的信息"""
    try:
        image_path = data_entry['images'][0]['path']
        
        if source_type == 'label':
            content_str = data_entry.get('labels')
            if not content_str:
                return image_path, None, "标签信息缺失"
        elif source_type == 'model':
            content_str = data_entry.get('response')
            if not content_str:
                 return image_path, None, "模型响应缺失"
        else:
            return None, None, "未知来源类型"

        content = json.loads(content_str)
        event_analyses = content.get("event_analyses")

        if not event_analyses:
            return image_path, None, "无异常事件"
        
        analysis = event_analyses[0]
        bbox = analysis.get("bbox") # [xmin, ymin, xmax, ymax]
        
        assessment = analysis.get("comprehensive_assessment", {})
        cot = analysis.get("simplified_cot", {})
        
        desc = (
            f"类型: {assessment.get('异常类型', 'N/A')}\n"
            f"风险: {assessment.get('风险等级', 'N/A')}\n\n"
            f"画面分析:\n{cot.get('画面分析', 'N/A')}\n\n"
            f"推理过程:\n{cot.get('推理过程', 'N/A')}"
        )
        
        return image_path, bbox, desc

    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
        print(f"解析 {source_type} 数据时出错: {e} | 数据条目: {str(data_entry)[:200]}...")
        image_path_fallback = data_entry.get('images', [{}])[0].get('path', '未知路径')
        return image_path_fallback, None, f"解析错误: {e}"


def draw_text_on_image(text, width, height, font):
    """创建一个包含换行文本的图像"""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    y_text = 15
    lines = text.split('\n')
    for line in lines:
        draw.text((15, y_text), line, font=font, fill=TEXT_COLOR)
        y_text += FONT_SIZE + 5 
    return img


def main(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"正在从 '{input_file}' 加载数据...")
    all_data = load_json_lines(input_file)
    if not all_data:
        print("错误: 文件中未加载到任何有效数据。")
        return

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        print(f"成功加载字体: '{FONT_PATH}'")
    except IOError:
        print(f"错误: 找不到字体文件 '{FONT_PATH}'。请确保它与脚本在同一目录或提供了正确路径。将使用默认字体。")
        font = ImageFont.load_default()

    print("开始生成可视化对比图...")
    for i, data_entry in enumerate(tqdm(all_data)):
        label_img_path, label_bbox_norm, label_desc = parse_data(data_entry, 'label')
        model_img_path, model_bbox_norm, model_desc = parse_data(data_entry, 'model')

        if label_img_path is None or not os.path.exists(label_img_path):
            print(f"跳过第 {i+1} 个样本: 找不到图片 '{label_img_path}'")
            continue

        original_img = cv2.imread(label_img_path)
        if original_img is None:
            print(f"跳过第 {i+1} 个样本: OpenCV无法读取图片 '{label_img_path}'")
            continue
        h, w, _ = original_img.shape

        # 1. 准备左上角: 标签可视化图
        label_viz_img = original_img.copy()
        if label_bbox_norm:
            # === 反归一化坐标 ===
            x1_norm, y1_norm, x2_norm, y2_norm = label_bbox_norm
            x1 = int((x1_norm / 1000.0) * w)
            y1 = int((y1_norm / 1000.0) * h)
            x2 = int((x2_norm / 1000.0) * w)
            y2 = int((y2_norm / 1000.0) * h)
            cv2.rectangle(label_viz_img, (x1, y1), (x2, y2), LABEL_COLOR, thickness=3)
        label_viz_img_pil = Image.fromarray(cv2.cvtColor(label_viz_img, cv2.COLOR_BGR2RGB))

        # 2. 准备左下角: 模型可视化图
        model_viz_img = original_img.copy()
        if model_bbox_norm:
            # === 反归一化坐标 ===
            x1_norm, y1_norm, x2_norm, y2_norm = model_bbox_norm
            x1 = int((x1_norm / 1000.0) * w)
            y1 = int((y1_norm / 1000.0) * h)
            x2 = int((x2_norm / 1000.0) * w)
            y2 = int((y2_norm / 1000.0) * h)
            cv2.rectangle(model_viz_img, (x1, y1), (x2, y2), MODEL_COLOR, thickness=3)
        model_viz_img_pil = Image.fromarray(cv2.cvtColor(model_viz_img, cv2.COLOR_BGR2RGB))

        # 3. 准备右上角: 标签描述
        label_desc_img = draw_text_on_image(f"--- 标签 (Ground Truth) ---\n\n{label_desc}", w, h, font)
        
        # 4. 准备右下角: 模型描述
        model_desc_img = draw_text_on_image(f"--- 模型输出 (Prediction) ---\n\n{model_desc}", w, h, font)

        # 5. 拼接成2x2大图
        final_img = Image.new('RGB', (w * 2, h * 2), 'white')
        final_img.paste(label_viz_img_pil, (0, 0))
        final_img.paste(label_desc_img, (w, 0))
        final_img.paste(model_viz_img_pil, (0, h))
        final_img.paste(model_desc_img, (w, h))
        
        draw = ImageDraw.Draw(final_img)
        draw.line([(w, 0), (w, h * 2)], fill=LINE_COLOR, width=2)
        draw.line([(0, h), (w * 2, h)], fill=LINE_COLOR, width=2)

        base_filename = os.path.basename(label_img_path)
        output_path = os.path.join(output_dir, base_filename)
        final_img.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从单个JSONL文件可视化模型输出与标签的对比图。")
    # 修改命令行参数，现在只有一个输入文件
    parser.add_argument("--input_file", default="/host/ssddata/disk2/zhanglu/swift/4cls/test/qwen25_7b_lora_norm/result_0922.json", help="包含模型输出和标签的JSON Lines文件路径。")
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/swift/4cls/test/qwen25_7b_lora_norm/result_0922_vision", help="保存可视化结果的目录路径。")
    
    
    args = parser.parse_args()

    # 根据新的函数签名调用main
    main(args.input_file, args.output_dir)
    print(f"\n处理完成！所有可视化结果已保存在 '{args.output_dir}' 目录中。")
    