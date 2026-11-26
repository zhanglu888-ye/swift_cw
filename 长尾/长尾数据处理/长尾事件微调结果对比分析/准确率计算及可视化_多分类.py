import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib import font_manager

# ================= 配置区域 =================

# 字体路径 (请修改为您服务器的真实路径)
FONT_PATH = "/host/ssddata/disk2/zhanglu/SimHei.ttf" 

# 类别列表
ALL_CLASSES = [
    "流动摊贩", "森林火灾", "垃圾桶满溢", "人员异常聚集", 
    "垃圾堆积", "人员异常倒地", "占道经营", "非机动车乱停放", "污水井倾倒"
]

# ===========================================

def parse_labels(text):
    """从文本中解析出类别集合"""
    if not text: return set()
    text = str(text).strip().replace("。", "")
    
    # 否定或未检测
    if text.startswith("否") or "未检测" in text or "不存在" in text:
        return set()
    
    # 提取冒号后的内容
    clean_text = text
    if "检测到以下情况" in text:
        try:
            clean_text = re.split(r"检测到以下情况[：:]", text)[1]
        except: pass
            
    # 分割
    raw_labels = re.split(r'[、,，\s]+', clean_text)
    
    # 过滤有效类别
    found_labels = set()
    for lbl in raw_labels:
        lbl = lbl.strip()
        if lbl in ALL_CLASSES:
            found_labels.add(lbl)
            
    return found_labels

def plot_heatmap(cm, labels, title, output_path):
    # 归一化
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    # 字体加载
    try:
        my_font = font_manager.FontProperties(fname=FONT_PATH)
    except:
        my_font = None
        print("警告: 字体加载失败")

    plt.figure(figsize=(16, 14))
    
    # 构造标注
    annot_labels = []
    for i in range(cm.shape[0]):
        row_labels = []
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j]
            # 只有 >0 才显示
            txt = f"{val}\n{pct:.1%}" if val > 0 else "0"
            row_labels.append(txt)
        annot_labels.append(row_labels)

    # 绘图
    ax = sns.heatmap(cm, annot=np.array(annot_labels), fmt="", cmap="Blues", 
                     xticklabels=labels, yticklabels=labels,
                     linewidths=0.5, linecolor='gray')
    
    if my_font:
        plt.title(title, fontsize=20, fontproperties=my_font)
        plt.xlabel("预测类别 (Predicted)", fontsize=16, fontproperties=my_font)
        plt.ylabel("真实类别 (Actual)", fontsize=16, fontproperties=my_font)
        plt.xticks(rotation=45, ha='right', fontsize=12, fontproperties=my_font)
        plt.yticks(rotation=0, fontsize=12, fontproperties=my_font)
    else:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"混淆矩阵保存至: {output_path}")

def main(json_path, output_path):
    print(f"正在处理: {json_path}")
    
    # 矩阵维度: N+1 (包含 None)
    n = len(ALL_CLASSES)
    # class_to_idx
    c2i = {cls: i for i, cls in enumerate(ALL_CLASSES)}
    none_idx = n
    
    # 初始化矩阵 (行: GT, 列: Pred)
    cm = np.zeros((n + 1, n + 1), dtype=int)
    
    data = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
            else:
                f.seek(0)
                for line in f:
                    if line.strip(): data.append(json.loads(line))
    except Exception as e:
        print(f"读取失败: {e}")
        return

    for item in data:
        # 1. 解析 GT 和 Pred
        gt_text = item.get('labels', '')
        pred_text = item.get('response', '')
        
        gt_set = parse_labels(gt_text)
        pred_set = parse_labels(pred_text)
        
        # 2. 核心逻辑：填充矩阵
        
        # 2.1 共同部分 (Correct / Diagonal)
        common = gt_set.intersection(pred_set)
        for cls in common:
            idx = c2i[cls]
            cm[idx, idx] += 1
            
        # 2.2 剩余部分 (Errors)
        rem_gt = list(gt_set - pred_set)    # 漏检的
        rem_pred = list(pred_set - gt_set)  # 多报的
        
        # 逻辑：尽可能将“漏检”和“多报”配对，视为“错判”
        # 例如：GT={A}, Pred={B} -> 认为 A 被错判为 B
        
        while rem_gt and rem_pred:
            g = rem_gt.pop(0)
            p = rem_pred.pop(0)
            cm[c2i[g], c2i[p]] += 1 # A 错判为 B (非对角线数据!)
            
        # 处理剩余的 GT (纯漏检 -> None)
        for g in rem_gt:
            cm[c2i[g], none_idx] += 1
            
        # 处理剩余的 Pred (纯误报 -> None 变 Pred)
        for p in rem_pred:
            cm[none_idx, c2i[p]] += 1
            
        # 处理纯背景图 (TN)
        if not gt_set and not pred_set:
            cm[none_idx, none_idx] += 1

    # 绘图
    display_labels = ALL_CLASSES + ["None"]
    title = f"Confusion Matrix: {os.path.basename(json_path)}"
    plot_heatmap(cm, display_labels, title, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="多分类结果JSON路径")
    parser.add_argument("--output", type=str, default="multi_cm.png", help="输出图片路径")
    args = parser.parse_args()
    main(args.json, args.output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多标签分类评估脚本")
    parser.add_argument("--json", type=str, default="/host/cephfs-data/zhanglu/qwen3_test_result/qwen3_4b_nofinetune/changwei_multlabel_nofinetune_1210_jintao.json", help="模型预测的JSON文件路径")
    parser.add_argument("--output", type=str, default="/host/cephfs-data/zhanglu/qwen3_test_result/show_1209_changwei/changwei_multlabel_nofinetune_1210_jintao", help="可视化结果输出文件夹")
    parser.add_argument("--font", type=str, default="/host/ssddata/disk2/zhanglu/SimHei.ttf", help="中文字体路径(如 simhei.ttf)")
          
    args = parser.parse_args()
    
    main(args.json, args.output, args.font)