import matplotlib.pyplot as plt
from matplotlib import font_manager

# ================= 配置区域 =================
FONT_PATH = "/host/ssddata/disk2/zhanglu/SimHei.ttf" 

# ---------------- 数据录入 ----------------

# 1. 验证集数据 (Validation Set)
val_data = {
    # 格式: [Pre_Acc, Pre_P, Pre_R,  Ckpt213_Acc, Ckpt213_P, Ckpt213_R,  Ckpt1065_Acc, Ckpt1065_P, Ckpt1065_R]
    "流动摊贩": [90.45, 89.84, 96.64,  98.31, 97.54, 100.00,  98.88, 99.16, 99.16],
    "森林火灾": [81.33, 100.00, 46.72, 94.63, 93.28, 91.24,  95.40, 94.07, 92.70],
    "垃圾桶满溢": [81.15, 97.28, 76.06, 98.85, 100.00, 98.40, 99.23, 99.47, 99.47],
    "人员异常聚集": [75.24, 90.11, 65.60, 100.00, 100.00, 100.00, 99.52, 99.21, 100.00],
    "垃圾堆积": [82.59, 93.81, 75.83,  96.02, 96.67, 96.67,  96.52, 97.48, 96.67],
    "人员异常倒地": [97.69, 98.43, 98.74, 98.46, 98.75, 99.37,  99.23, 99.68, 99.37],
    "占道经营": [59.12, 91.04, 52.81,  95.61, 95.42, 99.13,  96.96, 97.44, 98.70],
    "非机动车乱停放": [58.05, 67.31, 38.46, 96.55, 95.70, 97.80,  98.85, 98.90, 98.90],
    "污水井倾倒": [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]
}
val_overall = [80.53, 97.28, 98.00]

# 2. 测试集数据 (Test Set)
test_data = {
    # 注意：森林火灾在您的测试集表格中缺失，这里设为0
    "流动摊贩": [97.16, 99.14, 97.66,  12.97, 91.11, 2.66,  13.09, 88.24, 2.92],
    "森林火灾": [0, 0, 0,  0, 0, 0,  0, 0, 0], 
    "垃圾桶满溢": [93.27, 98.46, 92.52, 96.24, 98.01, 96.98, 97.03, 97.78, 98.29],
    "人员异常聚集": [65.16, 95.85, 45.34, 99.40, 100.00, 99.02, 99.40, 99.51, 99.51],
    "垃圾堆积": [86.88, 82.14, 92.37, 96.77, 94.27, 99.20, 96.96, 94.30, 99.60],
    "人员异常倒地": [97.55, 97.63, 98.26, 97.94, 97.64, 98.92, 93.17, 100.00, 88.50],
    "占道经营": [70.97, 64.55, 74.89, 72.68, 61.92, 97.84, 75.33, 64.07, 99.57],
    "非机动车乱停放": [64.44, 79.71, 55.28, 57.14, 93.28, 31.41, 42.55, 83.33, 6.28],
    "污水井倾倒": [76.74, 100.00, 52.38, 78.71, 99.36, 56.78, 79.07, 100.00, 57.14]
}
test_overall = [85.22, 67.07, 65.43]

# ===========================================

def generate_table(data_dict, overall_acc, title, filename):
    # 列标题
    col_labels = [
        "Pre\nAcc", "Pre\nPrec", "Pre\nRecall",
        "Ckpt213\nAcc", "Ckpt213\nPrec", "Ckpt213\nRecall",
        "Ckpt1065\nAcc", "Ckpt1065\nPrec", "Ckpt1065\nRecall"
    ]
    row_labels = list(data_dict.keys()) + ["Overall Acc"]
    
    cell_text = []
    cell_colors = []
    
    # 填充类别数据
    for cls in data_dict.keys():
        vals = data_dict[cls]
        # 文本
        txt_row = [f"{v:.2f}%" if v > 0 else "-" for v in vals]
        cell_text.append(txt_row)
        
        # 颜色 (红黄绿)
        c_row = []
        for v in vals:
            if v == 0: color = "#ffffff"
            elif v < 60: color = "#ffcccc" # 红
            elif v < 85: color = "#ffffcc" # 黄
            else: color = "#ccffcc" # 绿
            c_row.append(color)
        cell_colors.append(c_row)
        
    # 填充 Overall 行
    over_txt = []
    over_col = []
    acc_indices = [0, 3, 6]
    for i in range(9):
        if i in acc_indices:
            val = overall_acc[acc_indices.index(i)]
            over_txt.append(f"{val:.2f}%")
            if val < 60: c = "#ffcccc"
            elif val < 85: c = "#ffffcc"
            else: c = "#ccffcc"
            over_col.append(c)
        else:
            over_txt.append("")
            over_col.append("#eeeeee")
            
    cell_text.append(over_txt)
    cell_colors.append(over_col)
    
    # 绘图
    try:
        my_font = font_manager.FontProperties(fname=FONT_PATH)
    except:
        my_font = None

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('off')
    
    # 创建表格
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    # 样式调整
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5) # 增加行高
    
    # 设置表头样式
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')
        # 设置中文行标签字体
        if col == -1 and my_font:
            cell.set_text_props(fontproperties=my_font)

    plt.title(title, fontsize=18, pad=20, fontproperties=my_font)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"表格已保存: {filename}")

if __name__ == "__main__":
    # 生成验证集表格
    generate_table(val_data, val_overall, 
                  "单分类任务 - 验证集结果汇总 (Single-Label Validation Summary)", 
                  "val_summary.png")
                  
    # 生成测试集表格
    generate_table(test_data, test_overall, 
                  "单分类任务 - 测试集结果汇总 (Single-Label Test Summary)", 
                  "test_summary.png")