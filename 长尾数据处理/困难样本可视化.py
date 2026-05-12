import os
import json
from PIL import Image, ImageDraw, ImageFont

def text_wrap(text, font, max_width):
    """根据最大宽度自动换行文字"""
    lines = []
    if font.getlength(text) <= max_width:
        lines.append(text)
    else:
        words = list(text)  # 针对中文按字符拆分，若是英文可改用 text.split()
        current_line = ""
        for word in words:
            test_line = current_line + word
            if font.getlength(test_line) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    return lines

def visualize_data(img_root, json_root, output_root):
    # 支持的图片后缀
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 字体设置 (如果是中文，请确保路径指向一个支持中文的字体文件，如黑体或微软雅黑)
    # Windows: "msyh.ttc" (微软雅黑), Linux: "/usr/share/fonts/..."
    try:
        font_path = "/host/ssddata/disk2/zhanglu/SimHei.ttf"  # Windows 常用路径
        font = ImageFont.truetype(font_path, 20)
        title_font = ImageFont.truetype(font_path, 24)
    except:
        print("未找到指定中文字体，使用默认字体（可能无法显示中文）")
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # 1. 遍历 json_root 下的事件文件夹
    for event_name in os.listdir(json_root):
        event_path = os.path.join(json_root, event_name)
        
        if not os.path.isdir(event_path):
            continue
            
        # 创建对应的输出文件夹
        save_dir = os.path.join(output_root, event_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"正在处理类别: {event_name}")

        # 2. 遍历该事件文件夹下的所有 json 文件
        for json_file in os.listdir(event_path):
            if not json_file.endswith('.json'):
                continue
            
            json_full_path = os.path.join(event_path, json_file)
            file_basename = os.path.splitext(json_file)[0]
            
            # 3. 寻找对应的图片
            img_path = None
            for ext in img_extensions:
                temp_path = os.path.join(img_root, file_basename + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            
            if not img_path:
                print(f"警告: 找不到图片 {file_basename}")
                continue

            # 4. 读取 JSON 内容
            try:
                with open(json_full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 假设描述字段叫 'description' 或 'caption'，根据你的实际情况修改
                # 如果整个 json 就是一段话，直接用 str(data)
                description = data.get('description', data.get('caption', str(data)))
            except Exception as e:
                print(f"读取JSON失败: {json_full_path}, 错误: {e}")
                continue

            # 5. 可视化处理
            try:
                img = Image.open(img_path).convert("RGB")
                img_w, img_h = img.size
                
                # 设置文本区域宽度（固定 400 像素）
                text_area_w = 400
                padding = 20
                
                # 计算文字换行
                lines = text_wrap(description, font, text_area_w - 2 * padding)
                line_height = font.getbbox("Tg")[3] + 10 # 获取行高
                
                # 最终画布宽度 = 图片宽 + 文字区域宽
                canvas_w = img_w + text_area_w
                canvas_h = max(img_h, len(lines) * line_height + 100)
                
                # 创建白色背景画布
                new_img = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
                
                # 粘贴原图到左侧
                new_img.paste(img, (0, 0))
                
                # 在右侧绘制文字
                draw = ImageDraw.Draw(new_img)
                draw.text((img_w + padding, padding), f"Event: {event_name}", fill=(0, 0, 0), font=title_font)
                
                y_offset = padding + 50
                for line in lines:
                    draw.text((img_w + padding, y_offset), line, fill=(50, 50, 50), font=font)
                    y_offset += line_height
                
                # 6. 保存图片
                save_path = os.path.join(save_dir, f"{file_basename}_vis.jpg")
                new_img.save(save_path)
                
            except Exception as e:
                print(f"处理图片失败: {img_path}, 错误: {e}")

if __name__ == "__main__":
    # 配置路径
    config = {
        "img_root": "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/正样本/合并_deduplicated/large_images",        # 存放所有图片的文件夹
        "json_root": "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/正样本/分类and描述_正样本_删除",  # 存放带文件夹结构的JSON文件夹
        "output_root":"/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/正样本/分类and描述_正样本_删除_show"  # 输出结果文件夹
    }
    
    visualize_data(config["img_root"], config["json_root"], config["output_root"])
    print("任务完成！")
