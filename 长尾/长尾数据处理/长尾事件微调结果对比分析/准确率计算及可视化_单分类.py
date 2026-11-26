import json
import os
import shutil
import argparse
from PIL import Image, ImageDraw, ImageFont
from prettytable import PrettyTable

# ================= 配置区域 =================

# 提示词关键词映射到类别名称
# 脚本会检查 prompt 中是否包含 key，若包含则归为 value 类
PROMPT_KEYWORD_MAP = {
    "商品售卖": "流动摊贩",
    "超出": "占道经营",
    "黄色停车线": "非机动车乱停放",
    "雨水井": "污水井倾倒",
    "垃圾桶外": "垃圾桶满溢",
    "建筑材料": "垃圾堆积",
    "废弃家具": "垃圾堆积",
    "倒地": "人员异常倒地",
    "聚集": "人员异常聚集",
    "火灾": "森林火灾"
}

# 默认中文字体路径 (请根据你的系统修改，或者确保运行目录下有 simhei.ttf)
# Linux常见: /usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf
# Windows常见: C:/Windows/Fonts/simhei.ttf
DEFAULT_FONT_PATH = "simhei.ttf" 

# ===========================================

def get_class_from_prompt(prompt_text):
    """根据提示词内容匹配类别"""
    for keyword, class_name in PROMPT_KEYWORD_MAP.items():
        if keyword in prompt_text:
            return class_name
    return "未知类别"

def draw_text_on_image(image_path, text, save_path, font_path=None):
    """读取图片，绘制文字，并保存"""
    try:
        # 打开图片
        if not os.path.exists(image_path):
            print(f"[Warn] 图片不存在: {image_path}")
            return
            
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # 加载字体
        font_size = max(20, int(min(img.size) * 0.05)) # 动态字体大小
        try:
            # 优先使用传入的字体路径
            font = ImageFont.truetype(font_path, font_size)
        except:
            try:
                # 尝试系统常见字体
                font = ImageFont.truetype("arial.ttf", font_size) # 英文兜底
            except:
                font = ImageFont.load_default() # 最终兜底
        
        # 绘制文字 (红色，加个半透明背景方便看清)
        text_lines = text.split('\n')
        x, y = 10, 10
        
        # 计算背景框大小
        bg_w = 0
        bg_h = 0
        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            bg_w = max(bg_w, bbox[2])
            bg_h += bbox[3] + 5
            
        # 绘制半透明背景
        draw.rectangle((x-5, y-5, x+bg_w+10, y+bg_h+10), fill=(0, 0, 0, 128))
        
        # 绘制文字
        current_y = y
        for line in text_lines:
            draw.text((x, current_y), line, fill=(255, 0, 0), font=font)
            bbox = draw.textbbox((0, 0), line, font=font)
            current_y += bbox[3] + 5
            
        img.save(save_path)
        
    except Exception as e:
        print(f"[Error] 处理图片失败 {image_path}: {e}")

def calculate_metrics_and_visualize(input_json, output_dir, font_path):
    # 1. 初始化统计数据
    # 结构: { "类别名": {"TP": 0, "FP": 0, "TN": 0, "FN": 0} }
    stats = {}
    
    # 准备输出目录
    correct_dir = os.path.join(output_dir, "correct")
    incorrect_dir = os.path.join(output_dir, "incorrect")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) #以此清空旧数据，小心使用
    os.makedirs(correct_dir)
    os.makedirs(incorrect_dir)
    
    print(f"正在读取结果文件: {input_json} ...")
    
    count = 0
    with open(input_json, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid json line: {line[:50]}...")
                continue
                
            # 解析字段
            pred = item.get('response', '').strip()
            gt = item.get('labels', '').strip()
            prompt = item['messages'][0]['content']
            img_path = item['images'][0]['path']
            
            # 确定类别
            cls_name = get_class_from_prompt(prompt)
            if cls_name not in stats:
                stats[cls_name] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
            
            # 统计逻辑 (二分类：是/否)
            is_correct = (pred == gt)
            
            if gt == "是":
                if pred == "是":
                    stats[cls_name]["TP"] += 1
                else:
                    stats[cls_name]["FN"] += 1
            elif gt == "否":
                if pred == "否":
                    stats[cls_name]["TN"] += 1
                else:
                    stats[cls_name]["FP"] += 1
            else:
                # 标签不是是/否的情况，跳过统计但打印警告
                print(f"[Warn] 未知标签格式: {gt}")
                continue
            
            # 可视化保存
            save_folder = correct_dir if is_correct else incorrect_dir
            file_name = f"{cls_name}_{os.path.basename(img_path)}"
            save_path = os.path.join(save_folder, file_name)
            
            # 图片上的文字内容
            draw_text = f"Class: {cls_name}\nGT: {gt}\nPred: {pred}"
            draw_text_on_image(img_path, draw_text, save_path, font_path)
            
            count += 1
            if count % 100 == 0:
                print(f"已处理 {count} 张图片...")

    # 2. 计算指标并打印表格
    table = PrettyTable()
    table.field_names = ["类别", "Total", "Acc", "Precision", "Recall", "TP", "FP", "FN"]
    
    print("\n" + "="*20 + " 评估结果 " + "="*20)
    
    total_samples = 0
    total_correct = 0
    
    for cls, s in stats.items():
        tp = s["TP"]
        fp = s["FP"]
        tn = s["TN"]
        fn = s["FN"]
        
        count_cls = tp + fp + tn + fn
        if count_cls == 0: continue
        
        # Accuracy
        acc = (tp + tn) / count_cls
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        total_samples += count_cls
        total_correct += (tp + tn)
        
        table.add_row([
            cls, 
            count_cls, 
            f"{acc:.2%}", 
            f"{precision:.2%}", 
            f"{recall:.2%}",
            tp, fp, fn
        ])
        
    print(table)
    
    if total_samples > 0:
        print(f"\n总体准确率 (Overall Accuracy): {total_correct/total_samples:.2%}")
    print(f"可视化结果已保存至: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算准确率并可视化")
    parser.add_argument("--json", type=str, default="/host/cephfs-data/zhanglu/qwen3_test_result/qwen3_4b_lora/changwei_single_ck213_1210_jintao.json", help="模型预测的JSON文件路径")
    parser.add_argument("--output", type=str, default="/host/cephfs-data/zhanglu/qwen3_test_result/show_1209_changwei/changwei_single_ck213_1210_jintao", help="可视化结果输出文件夹")
    parser.add_argument("--font", type=str, default="/host/ssddata/disk2/zhanglu/SimHei.ttf", help="中文字体路径(如 simhei.ttf)")
    
    args = parser.parse_args()
    
    calculate_metrics_and_visualize(args.json, args.output, args.font)