import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

# --- 配置本地字体文件 ---
# 请将您的字体文件放在与此脚本相同的目录下，并在此处填写正确的文件名。
FONT_REGULAR_NAME = "/host/ssddata/disk2/zhanglu/SimHei.ttf"  # 常规字体文件名 (例如: 微软雅黑常规体)
FONT_BOLD_NAME = "/host/ssddata/disk2/zhanglu/SimHei.ttf"   # 粗体字体文件名 (例如: 微软雅黑粗体)


def wrap_text(text, font, max_width):
    """一个更健壮的文本换行函数，处理长单词和现有换行符"""
    lines = []
    for paragraph in text.split('\n'):
        words = paragraph.split(' ')
        current_line = ''
        for word in words:
            if font.getbbox(word)[2] > max_width:
                temp_word = ''
                for char in word:
                    if font.getbbox(temp_word + char)[2] <= max_width:
                        temp_word += char
                    else:
                        lines.append(temp_word)
                        temp_word = char
                current_line += temp_word + ' '
            elif font.getbbox(current_line + word)[2] <= max_width:
                current_line += word + ' '
            else:
                lines.append(current_line.strip())
                current_line = word + ' '
        lines.append(current_line.strip())
    return '\n'.join(lines)

def create_comparison_image(
    image_path,
    json_path1,
    json_path2,
    output_path,
):
    """
    创建一张对比图，根据特定规则显示两个JSON的内容，并使用本地字体。
    """
    try:
        original_image = Image.open(image_path)
        with open(json_path1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(json_path2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
    except FileNotFoundError as e:
        print(f"  - 错误: 缺少文件 {e.filename}，跳过。")
        return
    except Exception as e:
        print(f"  - 加载文件时出错: {e}，跳过。")
        return
        
    # --- 1. 提取所需数据 ---
    engine1 = data1.get("engine", "N/A")
    analyze1 = data1.get("analyze", "N/A")
    type2 = data2.get("异常类型", "N/A")
    subtype2 = data2.get("异常子类型", "N/A")
    full_text2 = data2.get("full_analysis_text", "N/A")

    # --- 2. 定义画布和样式 ---
    PADDING = 40
    TEXT_AREA_HEIGHT = 600
    BACKGROUND_COLOR = (255, 255, 255)
    TEXT_COLOR = (0, 0, 0)
    KEY_COLOR = (80, 80, 80)
    TITLE_COLOR = (50, 50, 50)
    RED_COLOR = (204, 0, 0)
    LINE_SPACING = 5
    VERTICAL_GAP = 18

    # --- 3. 设置字体 (从本地加载) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path_regular = os.path.join(script_dir, FONT_REGULAR_NAME)
    font_path_bold = os.path.join(script_dir, FONT_BOLD_NAME)
    
    try:
        title_font = ImageFont.truetype(font_path_bold, size=24)
        key_font = ImageFont.truetype(font_path_bold, size=16)
        value_font_regular = ImageFont.truetype(font_path_regular, size=15)
        value_font_highlight = ImageFont.truetype(font_path_bold, size=18)
        print("  - 成功加载本地字体。")
    except IOError:
        print(f"  - 警告: 找不到字体文件 '{FONT_REGULAR_NAME}' 或 '{FONT_BOLD_NAME}'，将使用默认字体。")
        title_font = ImageFont.load_default()
        key_font = ImageFont.load_default()
        value_font_regular = ImageFont.load_default()
        value_font_highlight = ImageFont.load_default()

    # --- 4. 创建画布 ---
    canvas_width = max(original_image.width, 1200)
    canvas_height = original_image.height + TEXT_AREA_HEIGHT + PADDING * 2
    canvas = Image.new('RGB', (canvas_width, canvas_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    # --- 5. 绘制内容 ---
    img_x_offset = (canvas_width - original_image.width) // 2
    canvas.paste(original_image, (img_x_offset, 0))

    line_y = original_image.height + PADDING // 2
    draw.line([(0, line_y), (canvas_width, line_y)], fill=(200, 200, 200), width=2)
    
    text_y_start = original_image.height + PADDING
    left_box_x = PADDING
    right_box_x = canvas_width // 2 + PADDING // 2
    text_box_width = canvas_width // 2 - PADDING * 1.5

    # 绘制左侧 (Model 1)
    draw.text((left_box_x, text_y_start), "Model 1 Result", font=title_font, fill=TITLE_COLOR)
    y_pos = text_y_start + 50
    draw.text((left_box_x, y_pos), "Engine:", font=key_font, fill=KEY_COLOR)
    y_pos += key_font.size + LINE_SPACING
    draw.text((left_box_x, y_pos), engine1, font=value_font_highlight, fill=RED_COLOR)
    y_pos += value_font_highlight.size + VERTICAL_GAP
    draw.text((left_box_x, y_pos), "Analyze:", font=key_font, fill=KEY_COLOR)
    y_pos += key_font.size + LINE_SPACING
    wrapped_analyze = wrap_text(analyze1, value_font_regular, text_box_width)
    draw.multiline_text((left_box_x, y_pos), wrapped_analyze, font=value_font_regular, fill=TEXT_COLOR, spacing=4)

    # 绘制右侧 (Model 2)
    draw.text((right_box_x, text_y_start), "Model 2 Result", font=title_font, fill=TITLE_COLOR)
    y_pos = text_y_start + 50
    draw.text((right_box_x, y_pos), "异常类型:", font=key_font, fill=KEY_COLOR)
    y_pos += key_font.size + LINE_SPACING
    draw.text((right_box_x, y_pos), type2, font=value_font_highlight, fill=RED_COLOR)
    y_pos += value_font_highlight.size + VERTICAL_GAP
    draw.text((right_box_x, y_pos), "异常子类型:", font=key_font, fill=KEY_COLOR)
    y_pos += key_font.size + LINE_SPACING
    draw.text((right_box_x, y_pos), subtype2, font=value_font_regular, fill=TEXT_COLOR)
    y_pos += value_font_regular.size + VERTICAL_GAP
    draw.text((right_box_x, y_pos), "Full Analysis Text:", font=key_font, fill=KEY_COLOR)
    y_pos += key_font.size + LINE_SPACING
    wrapped_full_text = wrap_text(full_text2, value_font_regular, text_box_width)
    draw.multiline_text((right_box_x, y_pos), wrapped_full_text, font=value_font_regular, fill=TEXT_COLOR, spacing=4)

    # --- 6. 保存结果 ---
    canvas.save(output_path)
    print(f"  - ✅ 可视化结果已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="将图片和两个模型的JSON描述按特定规则可视化到一张对比图上。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/xn/250804/data/有效/images/突发状况", help="存放原始图片的目录。")
    parser.add_argument("-m1", "--model1_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/xn/250804/data/有效/doubao1p5/突发状况", help="存放模型1的JSON结果的目录。")
    parser.add_argument("-m2", "--model2_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/xn/250804/data/有效/gemini/突发状况", help="存放模型2的JSON结果的目录。")
    parser.add_argument("-o", "--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/xn/250804/data/有效/comparison_output_final", help="存放生成的可视化图片的目录。")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    total_images = len(image_files)
    print(f"发现 {total_images} 张图片，开始生成可视化结果...")

    if total_images == 0:
        print("在图片目录中未找到任何图片文件。")
        return

    for index, image_filename in enumerate(image_files, 1):
        print(f"\n--- 正在处理第 {index}/{total_images} 张图片: {image_filename} ---")
        
        base_name = os.path.splitext(image_filename)[0]
        
        img_path = os.path.join(args.image_dir, image_filename)
        json1_path = os.path.join(args.model1_dir, f"{base_name}.json")
        json2_path = os.path.join(args.model2_dir, f"{base_name}.json")
        output_image_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")

        create_comparison_image(img_path, json1_path, json2_path, output_image_path)

    print("\n--- 所有任务处理完毕 ---")

if __name__ == "__main__":
    main()