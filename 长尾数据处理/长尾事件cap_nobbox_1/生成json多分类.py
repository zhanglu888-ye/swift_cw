import os
import json
import random
import argparse

# 固定随机种子
random.seed(42)

# ================= 类别定义配置 =================

# 字典格式： "类别名称": "该类别的判断标准描述"
# 注意：去掉了原定义中“输出是/否”的指令，只保留描述，因为现在是汇总回答。
CLASS_DEFINITIONS = {
    "森林火灾": "重点关注林区、草地或植被茂密区域是否有异常的烟柱（白色、灰色或黑色）、火焰光亮或大面积燃烧痕迹。",
    
    "流动摊贩": "在城市主要大街、十字路口、公共道路设施（消防通道、盲道、拐角、天桥、地下通道）场景，识别是否有人在三轮车/小货车/小推车/菜篮子/桌子/地摊儿旁边进行商品售卖。",
    
    "占道经营": "在城市沿街商铺、店铺门口，识别店家的商品、广告牌或餐桌是否超出了店铺门口台阶，或超出了电子围栏标记线框。",
    
    "非机动车乱停放": "在城市主要大街、商圈店铺门口场景，识别自行车、电动车、摩托车、二轮或三轮车辆是否未按规定停在黄色停车线框内。",
    
    "污水井倾倒": "在城市主要大街、小巷、沿街店铺门口，识别是否有人正在往雨水井中倾倒液体或废弃物。",
    
    "垃圾桶满溢": "在城市街道场景下，识别垃圾桶外地面是否有较多垃圾影响市容，或者垃圾桶桶壁外沿上是否有垃圾堆积。",
    
    "垃圾堆积": "在城市街道场景下，识别地面是否有较大量的建筑材料、生活垃圾、废弃家具占用了盲道或机动车道，且该处并非垃圾桶满溢的情况。",
    
    "人员异常倒地": "识别图像中人员是否因意外、疾病或外力导致身体呈水平或接近水平状态接触地面（如平躺、俯卧、侧卧），姿态异于正常休息。",
    
    "人员异常聚集": "识别公共区域内人群密度是否异常增高、行为是否异常集中，综合判断是否存在异常聚集现象。"
}

# 获取有序的类别列表，保证生成的提示词顺序一致
ORDERED_CLASSES = list(CLASS_DEFINITIONS.keys())

# 系统前缀
SYSTEM_PREFIX = "你是一个专业的视觉分析助手。"

# 预先构造通用的问题模板（除了图片外，问题文本对所有图片都是一样的）
# 格式：
# ...
# 1. [类别]: [定义]
# 2. [类别]: [定义]
# ...
# 请分析图像...
def build_question_text():
    intro = f"{SYSTEM_PREFIX}请分析图像，检测是否存在以下违规或异常情况，判断标准如下：\n"
    items = []
    for idx, cls_name in enumerate(ORDERED_CLASSES, 1):
        desc = CLASS_DEFINITIONS[cls_name]
        items.append(f"{idx}. 【{cls_name}】：{desc}")
    
    conditions = "\n".join(items)
    suffix = "\n\n请根据以上标准分析图像，回答是否存在上述情况？"
    return intro + conditions + suffix

# 全局变量保存问题文本
FULL_QUESTION_TEXT = build_question_text()

# ===========================================

def process_file_list(file_list, image_dir, label_dir):
    dataset = []
    
    # 统计信息
    stats = {
        "total_images": 0,
        "clean_images": 0,    # 无任何标签
        "anomaly_images": 0,  # 有至少一个标签
        "class_counts": {k: 0 for k in ORDERED_CLASSES}
    }
    
    for label_file in file_list:
        label_path = os.path.join(label_dir, label_file)
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取文件 {label_file} 失败: {e}")
            continue
            
        file_name = data.get('file_name')
        labels = data.get('labels', [])
        
        # 检查图片是否存在
        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            continue
            
        abs_image_path = os.path.abspath(image_path)
        stats["total_images"] += 1
        
        # 1. 整理有效标签
        # 过滤出我们在 ORDERED_CLASSES 中定义的标签
        valid_labels = [l for l in labels if l in CLASS_DEFINITIONS]
        # 去重并排序（按我们在 ORDERED_CLASSES 中的顺序），保证回答文本的一致性
        valid_labels = sorted(list(set(valid_labels)), key=lambda x: ORDERED_CLASSES.index(x))
        
        # 2. 构造回答 (Answer)
        if valid_labels:
            labels_str = "、".join(valid_labels)
            answer = f"是，检测到以下情况：{labels_str}。"
            
            stats["anomaly_images"] += 1
            for l in valid_labels:
                stats["class_counts"][l] += 1
        else:
            answer = "否，未检测到上述情况。"
            stats["clean_images"] += 1
            
        # 3. 组合最终数据项
        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": FULL_QUESTION_TEXT
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ],
            "images": [abs_image_path]
        }
        dataset.append(entry)
            
    return dataset, stats

def main(image_dir, label_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    total_files = len(all_files)
    
    if total_files == 0:
        print("未找到任何标签文件。")
        return

    print(f"共发现 {total_files} 个标签文件。正在处理...")
    
    # 打印生成的 Prompt 预览
    print("\n[Prompt 预览 (截取前500字符)]:")
    print("-" * 50)
    print(FULL_QUESTION_TEXT[:500] + "...")
    print("-" * 50 + "\n")

    # 1. 打乱并切分 (90% Train, 10% Val)
    random.shuffle(all_files)
    # split_index = int(total_files * 0.9)
    split_index = int(total_files * 1)    
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    # 2. 生成数据集
    print("正在生成训练集...")
    train_dataset, train_stats = process_file_list(train_files, image_dir, label_dir)
    
    print("正在生成验证集...")
    val_dataset, val_stats = process_file_list(val_files, image_dir, label_dir)

    # 3. 保存文件
    train_output_path = os.path.join(output_dir, "train_multilabel_def.json")
    val_output_path = os.path.join(output_dir, "val_multilabel_def.json")

    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)

    with open(val_output_path, 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, ensure_ascii=False, indent=2)

    # 4. 打印统计信息
    def print_stats_table(name, s):
        print(f"\n>>> 数据集: {name} <<<")
        print(f"{'指标':<20} | {'数值':<10}")
        print("-" * 35)
        print(f"{'总图片数':<20} | {s['total_images']:<10}")
        print(f"{'有异常(Positive)':<20} | {s['anomaly_images']:<10}")
        print(f"{'无异常(Negative)':<20} | {s['clean_images']:<10}")
        print("-" * 35)
        print("各类别检出数量:")
        for cls_name in ORDERED_CLASSES:
            count = s['class_counts'][cls_name]
            # 只打印有数据的类别，让表格更整洁
            if count > 0: 
                print(f"  - {cls_name:<16}: {count}")

    print("="*60)
    print_stats_table("训练集 (Train)", train_stats)
    print_stats_table("验证集 (Val)", val_stats)
    print("="*60)
    print(f"文件已保存:")
    print(f"  Train: {train_output_path}")
    print(f"  Val  : {val_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成带详细定义的多标签问答数据集")
    # parser.add_argument("--img_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并_deduplicated/large_images", help="图片文件夹路径")
    # parser.add_argument("--label_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并_deduplicated/label_json", help="标签JSON文件夹路径")
    # parser.add_argument("--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data", help="输出文件保存目录")
     
     
    parser.add_argument("--img_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao/合并数据/large_images", help="图片文件夹路径")
    parser.add_argument("--label_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao/合并数据/label_json", help="标签JSON文件夹路径")
    parser.add_argument("--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao", help="输出文件保存目录")
          
      
    args = parser.parse_args()
    
    main(args.img_dir, args.label_dir, args.output_dir)