import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# 设置参数
data_dir = '/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/test_data_quchong_filtered_out_v2'  # 替换为你的数据集路径
output_path = '/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/visualization_testquchong_过滤掉_v2.png'  # 输出文件路径
font_path = '/host/ssddata/disk2/zhanglu/SimHei.ttf'  # 替换为你的TTF字体文件路径


classes = sorted(os.listdir(data_dir))  # 获取所有类别（文件夹名）
images_per_row = 20  # 每行 20 张图片
rows_per_class = 2  # 每类 2 行
max_images_per_class = rows_per_class * images_per_row  # 每类最多显示 40 张图片
img_size = (120, 120)  # 图片大小 150x150
padding = 10  # 图片间距
text_height = 40  # 类别标签高度

# 计算画布大小（关键修正：确保高度足够）
n_classes = len(classes)
canvas_width = images_per_row * img_size[0] + (images_per_row + 1) * padding
canvas_height = n_classes * (text_height + rows_per_class * img_size[1] + (rows_per_class) * padding) + padding

# 创建空白画布（白色背景）
canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype(font_path, 24)  # 字体大小 24

current_y = padding  # 当前绘制位置的 Y 坐标

for i, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    image_files = sorted(os.listdir(class_dir))[:max_images_per_class]
    
    # 绘制类别名称（中文）
    draw.text((padding, current_y), f"类别: {class_name}", font=font, fill="black")
    current_y += text_height
    
    # 绘制该类别下的所有图片（每类 2 行，每行 20 张）
    for row in range(rows_per_class):
        current_x = padding
        for col in range(images_per_row):
            img_idx = row * images_per_row + col
            if img_idx < len(image_files):
                img_path = os.path.join(class_dir, image_files[img_idx])
                try:
                    img = Image.open(img_path).resize(img_size)
                    canvas.paste(img, (current_x, current_y))
                except Exception as e:
                    print(f"无法加载图像 {img_path}: {e}")
                    # 绘制黑色占位图
                    blank_img = Image.new("RGB", img_size, "black")
                    canvas.paste(blank_img, (current_x, current_y))
            else:
                # 如果图片不足，留白
                blank_img = Image.new("RGB", img_size, "white")
                canvas.paste(blank_img, (current_x, current_y))
            
            current_x += img_size[0] + padding
        
        # 换行（仅在行末添加间距）
        current_y += img_size[1] + padding
    
    # 不需要额外 class_spacing，仅保留默认的 padding

# 保存结果（不显示）
canvas.save(output_path, quality=95)
print(f"可视化结果已保存到: {output_path}")
