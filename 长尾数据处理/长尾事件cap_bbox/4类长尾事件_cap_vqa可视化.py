# 文件名: generate_final_report_v2.py
import argparse
import json
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 基本配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- 样式配置 ---
IMG_WIDTH = 640
COL_WIDTH = 580
MARGIN = 30
PADDING = 20
HEADER_HEIGHT = 50
LINE_SPACING = 8
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
HEADER_TEXT_COLOR = (255, 255, 255)
MISSING_TEXT_COLOR = (150, 150, 150)
BBOX_COLORS = [(255, 59, 59), (59, 130, 255), (22, 163, 74), (249, 115, 22), (139, 92, 246), (236, 72, 153)]

# --- 辅助函数 ---
def get_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try: return ImageFont.truetype(font_path, size=size)
    except IOError:
        logging.warning(f"无法加载字体 '{font_path}'，将使用默认字体。")
        return ImageFont.load_default()

def wrap_text(draw: ImageDraw.ImageDraw, text: str, width: int, font: ImageFont.FreeTypeFont) -> List[str]:
    lines = []
    for paragraph in text.split('\n'):
        if not paragraph: lines.append(''); continue
        current_line = ""
        for char in paragraph:
            if draw.textlength(current_line + char, font) <= width: current_line += char
            else: lines.append(current_line); current_line = char
        lines.append(current_line)
    return lines

def draw_text_block(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], title: str, content: str, width: int, fonts: Dict, color: Tuple) -> int:
    x, y = xy
    title_font, main_font = fonts['title'], fonts['main']
    draw.text((x, y), title, font=title_font, fill=color)
    y += title_font.getbbox("A")[3] + LINE_SPACING * 2
    line_height = main_font.getbbox("A")[3] + LINE_SPACING
    lines = wrap_text(draw, content, width, main_font)
    for line in lines:
        draw.text((x, y), line, font=main_font, fill=color)
        y += line_height
    return y + PADDING

def draw_bboxes_on_image(
    image: Image.Image,
    bboxes_info: List[Dict],
    scale_factor: float,
    font: ImageFont.FreeTypeFont
) -> None:
    """在给定的图片上绘制Bbox和标签。"""
    draw = ImageDraw.Draw(image)
    for i, item in enumerate(bboxes_info):
        box = item["box"]
        label = item["label"]
        color = BBOX_COLORS[i % len(BBOX_COLORS)]
        
        # <<< FIX: 应用缩放因子转换坐标 >>>
        scaled_box = [coord * scale_factor for coord in box]
        
        draw.rectangle(scaled_box, outline=color, width=3)
        display_label = f"[{i+1}] {label}"
        
        # 绘制带背景的标签
        label_pos = (scaled_box[0], scaled_box[1] - 24)
        label_bbox = draw.textbbox(label_pos, display_label, font=font)
        bg_box = (label_bbox[0]-2, label_bbox[1]-2, label_bbox[2]+2, label_bbox[3]+2)
        draw.rectangle(bg_box, fill=color)
        draw.text(label_pos, display_label, font=font, fill=(255, 255, 255))

# --- 核心处理函数 ---
def get_ground_truth_bboxes(case_name: str, cropped_data_dir: Path) -> List[Dict]:
    """从cropped_data目录为指定案例提取所有人工标注的Bbox信息。"""
    bboxes_info = []
    case_path = cropped_data_dir / case_name
    if not case_path.is_dir(): return []
    for category_dir in case_path.iterdir():
        if not category_dir.is_dir(): continue
        for meta_path in sorted(category_dir.glob('*.json')):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f: meta_data = json.load(f)
                bboxes_info.append({
                    "box": meta_data["original_bbox"],
                    "label": meta_data["category_name"],
                    "crop_stem": meta_path.stem
                })
            except Exception: pass
    bboxes_info.sort(key=lambda x: x['crop_stem'])
    return bboxes_info

# 文件名: generate_final_report_v2.py

# ... (脚本的其他部分保持不变) ...

def get_caption_bboxes(case_name: str, caption_dir: Path) -> List[Dict]:
    """
    【修正后】从caption结果文件中解析出VLM生成的所有Bbox信息。
    """
    bboxes_info = []
    caption_files = sorted(list(caption_dir.rglob(f"{case_name}_*.json")))
    
    # 使用一个计数器来确保每个Bbox的序号是唯一的
    instance_counter = 0

    for cap_path in caption_files:
        try:
            with open(cap_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # <<< FIX: 遍历列表中的 *所有* 分析结果，而不是只取第一个 >>>
            analyses = data.get("event_analyses", []) # 获取整个列表
            
            for analysis in analyses:
                assessment = analysis.get("comprehensive_assessment") or analysis.get("detailed_cot", {}).get("comprehensive_assessment")
                
                # 确保 assessment 是一个字典
                if not isinstance(assessment, dict):
                    continue

                bbox_str = assessment.get("bbox", "")
                box = [float(c) for c in bbox_str.split()]

                if len(box) == 4:
                    category = assessment.get("异常类型", "未知类别")
                    instance_counter += 1
                    bboxes_info.append({
                        "box": box,
                        "label": category,
                        # 使用计数器作为唯一标识，尽管我们不在VQA中使用它
                        "crop_stem": f"event_{instance_counter}" 
                    })
        except Exception as e:
            logging.warning(f"解析Caption Bbox时出错 (文件: {cap_path.name}): {e}")
            pass
            
    return bboxes_info

# ... (脚本的其余部分，包括 create_visualization 和 main 函数，都保持不变) ...

def create_visualization(
    large_image_path: Path,
    cropped_data_dir: Path, caption_dir: Path, vqa_dir: Path,
    output_path: Path, font_paths: Dict
):
    case_name = large_image_path.stem
    fonts = {
        'header': get_font(font_paths['main'], 24), 'title': get_font(font_paths['main'], 18),
        'main': get_font(font_paths['main'], 16), 'label': get_font(font_paths['bold'], 16)
    }

    # 1. 加载原始大图
    try:
        original_img = Image.open(large_image_path).convert("RGB")
    except Exception as e:
        logging.error(f"无法加载图片 {large_image_path}: {e}"); return

    # 2. 计算缩放比例并准备两张显示用的图片副本
    scale_factor = IMG_WIDTH / original_img.width
    display_height = int(original_img.height * scale_factor)
    img_gt = original_img.resize((IMG_WIDTH, display_height)) # Ground Truth图
    img_caption = original_img.resize((IMG_WIDTH, display_height)) # Caption图
    
    # 3. 获取并绘制Bbox
    gt_bboxes = get_ground_truth_bboxes(case_name, cropped_data_dir)
    caption_bboxes = get_caption_bboxes(case_name, caption_dir)
    draw_bboxes_on_image(img_gt, gt_bboxes, scale_factor, fonts['label'])
    draw_bboxes_on_image(img_caption, caption_bboxes, scale_factor, fonts['label'])

    # 4. 加载文本内容 (Caption 和 VQA)
    caption_contents, vqa_contents_map = [], {}
    caption_files = sorted(list(caption_dir.rglob(f"{case_name}_*.json")))
    for cap_path in caption_files:
        try:
            with open(cap_path, 'r', encoding='utf-8') as f: caption_text = f.read()
            try: caption_text = json.dumps(json.loads(caption_text), ensure_ascii=False, indent=2)
            except: pass
            caption_contents.append({"title": f"Caption: {cap_path.name}", "content": caption_text})
        except Exception: pass

    vqa_files = sorted(list(vqa_dir.rglob(f"{case_name}_*.json")))
    for vqa_path in vqa_files:
        try:
            with open(vqa_path, 'r', encoding='utf-8') as f: data = json.load(f)
            vqa_text = "".join([f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}\n\n" for qa in data.get("qa_pairs", [])])
            vqa_contents_map[vqa_path.stem.replace(f"{case_name}_", "")] = vqa_text.strip()
        except Exception as e:
            vqa_contents_map[vqa_path.stem.replace(f"{case_name}_", "")] = f"JSON解析失败:\n{e}"

    # 5. 创建画布并布局
    # ... (这部分与之前版本类似，但左侧图片高度翻倍)
    temp_canvas = Image.new("RGB", (1, 1)); temp_draw = ImageDraw.Draw(temp_canvas)
    cap_h = sum([draw_text_block(temp_draw, (0,0), c['title'], c['content'], COL_WIDTH, fonts, TEXT_COLOR) for c in caption_contents]) or 100
    vqa_h = sum([draw_text_block(temp_draw, (0,0), f"[{i+1}] VQA for {item['crop_stem']}", vqa_contents_map.get(item['crop_stem'], "N/A"), COL_WIDTH, fonts, TEXT_COLOR) for i, item in enumerate(gt_bboxes)]) or 100
    
    left_col_height = display_height * 2 + PADDING * 2 # 上下图 + 标题 + 间距
    canvas_height = HEADER_HEIGHT + MARGIN * 2 + max(left_col_height, cap_h, vqa_h)
    canvas_width = MARGIN * 4 + IMG_WIDTH + COL_WIDTH * 2 + PADDING * 2
    canvas = Image.new("RGB", (canvas_width, int(canvas_height)), BG_COLOR)
    final_draw = ImageDraw.Draw(canvas)

    final_draw.rectangle([0, 0, canvas_width, HEADER_HEIGHT], fill=(60, 60, 60))
    final_draw.text((MARGIN, 10), f"可视化报告: {case_name}", font=fonts['header'], fill=HEADER_TEXT_COLOR)
    
    # 绘制左侧栏
    y_offset = HEADER_HEIGHT + MARGIN
    final_draw.text((MARGIN, y_offset), "↑ Ground Truth (原始标注)", font=fonts['title'], fill=TEXT_COLOR)
    canvas.paste(img_gt, (MARGIN, y_offset + 30))
    
    y_caption_img = y_offset + 30 + display_height + PADDING
    final_draw.text((MARGIN, y_caption_img), "↓ Caption BBox (模型生成)", font=fonts['title'], fill=TEXT_COLOR)
    canvas.paste(img_caption, (MARGIN, y_caption_img + 30))
    
    # 绘制中间和右侧栏 (与之前相同)
    x_cap, y_cap = MARGIN * 2 + IMG_WIDTH + PADDING, y_offset
    if caption_contents:
        for cap in caption_contents: y_cap = draw_text_block(final_draw, (x_cap, y_cap), cap['title'], cap['content'], COL_WIDTH, fonts, TEXT_COLOR)
    else: draw_text_block(final_draw, (x_cap, y_cap), "Caption 结果", "未找到对应的Caption结果文件。", COL_WIDTH, fonts, MISSING_TEXT_COLOR)

    x_vqa, y_vqa = MARGIN * 3 + IMG_WIDTH + COL_WIDTH + PADDING * 2, y_offset
    if gt_bboxes:
        for i, item in enumerate(gt_bboxes):
            vqa_content = vqa_contents_map.get(item['crop_stem'], "未找到对应的VQA结果。")
            y_vqa = draw_text_block(final_draw, (x_vqa, y_vqa), f"[{i+1}] VQA for {item['crop_stem']}", vqa_content, COL_WIDTH, fonts, TEXT_COLOR if item['crop_stem'] in vqa_contents_map else MISSING_TEXT_COLOR)
    else:
        draw_text_block(final_draw, (x_vqa, y_vqa), "VQA 结果", "此图片无原始Bbox，无对应VQA。", COL_WIDTH, fonts, MISSING_TEXT_COLOR)

    # 6. 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=95)

def main():
    parser = argparse.ArgumentParser(description="生成包含图片对比、Caption和VQA的三栏可视化报告。")
    parser.add_argument("--large_image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/large_images", help="原始大图所在的目录。")
    parser.add_argument("--cropped_data_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/cropped_data", help="包含抠图和元数据JSON的目录（Bbox的来源）。")
    parser.add_argument("--caption_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_caption_bbox", help="注入了Bbox的Caption结果根目录。")
    parser.add_argument("--vqa_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_vqa", help="VQA结果根目录。")
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_show", help="保存可视化图片的输出目录。")
    parser.add_argument("--font_path", default="/host/ssddata/disk2/zhanglu/SimHei.ttf", help="用于渲染文本的常规中文字体路径。")
    parser.add_argument("--font_path_bold", help="（可选）用于标题的粗体中文字体路径。如果未提供，将使用常规字体。")
    parser.add_argument("--workers", type=int, default=8, help="并发处理的线程数。")
    args = parser.parse_args()


    font_paths = {'main': args.font_path, 'bold': args.font_path_bold or args.font_path}
    large_image_dir, cropped_data_dir = Path(args.large_image_dir), Path(args.cropped_data_dir)
    caption_dir, vqa_dir, output_dir = Path(args.caption_dir), Path(args.vqa_dir), Path(args.output_dir)

    # 以caption目录为准，收集所有有结果的案例名
    case_names_to_process = {p.stem.rsplit('_', 1)[0] for p in caption_dir.rglob('*.json')}
    if not case_names_to_process:
        logging.error(f"在 '{caption_dir}' 中未找到任何Caption JSON文件。")
        return
    logging.info(f"从Caption结果中发现 {len(case_names_to_process)} 个需要可视化的案例。")

    tasks = []
    for name in case_names_to_process:
        large_image_path = next(large_image_dir.glob(f"{name}.*"), None)
        if large_image_path and large_image_path.exists(): tasks.append(large_image_path)
        else: logging.warning(f"找到了 '{name}' 的Caption结果，但在 '{large_image_dir}' 中找不到对应的大图，跳过。")

    if not tasks: logging.error("未能匹配到任何有效的大图进行处理。"); return
    logging.info(f"成功匹配到 {len(tasks)} 张大图，将使用 {args.workers} 个并发线程处理...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                create_visualization, img_path, cropped_data_dir, caption_dir,
                vqa_dir, output_dir / f"{img_path.stem}_visualization.jpg", font_paths
            ): img_path.name for img_path in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="生成可视化报告"):
            try: future.result()
            except Exception as e: logging.error(f"处理图片 {futures[future]} 时发生错误: {e}")

    logging.info("所有可视化报告生成完毕！")

if __name__ == "__main__":
    main()
    
    