import os
import argparse
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage

# 定义支持的图片格式
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 定义Excel中图片的显示高度 (像素)
IMG_DISPLAY_HEIGHT = 100 

def parse_category(filename):
    """
    根据文件名解析类别。
    逻辑：提取第一个 '_' 之前的内容。
    例如: "非机动车乱停放_0668.jpg" -> "非机动车乱停放"
    """
    name_without_ext = os.path.splitext(filename)[0]
    if '_' in name_without_ext:
        # 分割一次，取第一部分
        return name_without_ext.split('_', 1)[0]
    else:
        # 如果没有下划线，返回"未知"或文件名本身，这里暂存为空字符串方便筛选
        return "未知类别"

def process_images_to_excel(input_dir, output_xlsx):
    # 1. 初始化 Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "评估结果"

    # 设置表头
    headers = ["图片名称", "图片预览", "类别", "状态 (正确/错误)"]
    ws.append(headers)

    # 设置列宽
    ws.column_dimensions['A'].width = 40  # 图片名称
    ws.column_dimensions['B'].width = 20  # 图片列 (稍后根据图片比例调整视感)
    ws.column_dimensions['C'].width = 20  # 类别
    ws.column_dimensions['D'].width = 15  # 状态

    # 定义要遍历的子文件夹及其对应的状态
    subfolders = {
        "correct": "正确",
        "incorrect": "错误"
    }

    current_row = 2  # 从第2行开始写入数据 (第1行是表头)

    print(f"开始处理，输入目录: {input_dir}")

    for folder_name, status_label in subfolders.items():
        folder_path = os.path.join(input_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在 {folder_path}，跳过。")
            continue

        files = sorted(os.listdir(folder_path))
        
        for filename in files:
            # 检查后缀
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            file_path = os.path.join(folder_path, filename)

            # --- 1. 写入文本信息 ---
            category = parse_category(filename)
            
            ws.cell(row=current_row, column=1, value=filename)   # A列: 名称
            # B列留给图片
            ws.cell(row=current_row, column=3, value=category)   # C列: 类别
            ws.cell(row=current_row, column=4, value=status_label) # D列: 状态

            # --- 2. 处理并插入图片 ---
            try:
                # 使用 openpyxl 加载图片
                img = ExcelImage(file_path)
                
                # 获取原始尺寸，计算缩放比例
                # 我们固定高度为 IMG_DISPLAY_HEIGHT，宽度等比缩放
                aspect_ratio = img.width / img.height
                new_width = IMG_DISPLAY_HEIGHT * aspect_ratio
                
                img.height = IMG_DISPLAY_HEIGHT
                img.width = new_width
                
                # 将图片锚定到 B 列的当前行
                anchor = f"B{current_row}"
                ws.add_image(img, anchor)
                
                # 调整行高以适应图片 (Excel行高单位约为像素的0.75倍，这里粗略设置)
                # height in points. 1 pixel ~= 0.75 points
                ws.row_dimensions[current_row].height = IMG_DISPLAY_HEIGHT * 0.75
                
                # 动态调整B列宽以适应最宽的图片 (粗略估算: 1个单位宽 ≈ 7像素)
                current_col_width = ws.column_dimensions['B'].width
                required_width = new_width / 7
                if required_width > current_col_width:
                     ws.column_dimensions['B'].width = required_width

            except Exception as e:
                print(f"图片读取失败: {filename}, 错误: {e}")
                ws.cell(row=current_row, column=2, value="图片无法加载")

            current_row += 1
            
            if (current_row - 1) % 50 == 0:
                print(f"已处理 {current_row - 1} 张图片...")

    # 保存文件
    try:
        wb.save(output_xlsx)
        print(f"\n成功！结果已保存至: {output_xlsx}")
        print(f"共处理 {current_row - 2} 张图片。")
    except Exception as e:
        print(f"保存Excel失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将correct/incorrect文件夹中的图片汇总到Excel")
    
    # 命令行参数
    parser.add_argument("--input_dir", type=str, default="/host/cephfs-data/zhanglu/qwen3_test_result/show_1209_changwei/changwei_single_ck213_1209", help="包含correct和incorrect文件夹的根目录路径")
    parser.add_argument("--output_file", type=str, default="/host/cephfs-data/zhanglu/qwen3_test_result/show_1209_changwei/changwei_single_ck213_1209/evaluation_result.xlsx", help="输出的xlsx文件名")
    
    args = parser.parse_args()
    
    process_images_to_excel(args.input_dir, args.output_file)