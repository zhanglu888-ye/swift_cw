import os
import json
import random
import argparse

# 固定随机种子
random.seed(42)

# ================= 配置区域 =================

# 1. 定义森林火灾的专用配置
FIRE_CLASS = "森林火灾"
FIRE_PROMPT = "请分析图像中是否发生火灾。"

# 2. 定义城管/通用场景的配置
# 注意：这里不包含森林火灾，只包含城管类
CITY_PROMPTS_MAP = {
    "流动摊贩": "在城市主要大街、十字路口、公共道路设施（消防通道、盲道、十字路口拐角、天桥、地下通道、楼梯/电梯出入口）场景，识别是否有人三轮车汽车/小货车/小推车/菜篮子/桌子/地摊儿旁边进行商品售卖的情况，存在则输出识别结果为是，否则输出识别结果为否；",
    "占道经营": "识别画面中场景是否在城市马路边上的沿街商铺、店铺门口，识别出是否存在店家的商品、广告牌或者就餐餐桌超出了店铺门口台阶的情况，或者存在店家的商品、广告牌或者就餐餐桌超出了画面中的电子围栏标记线框的情况，存在则输出识别结果为是，否则输出识别结果为否；",
    "非机动车乱停放": "在城市主要大街、商圈店铺门口场景，识别画面中是否有自行车、电动车、摩托车、二轮或三轮车辆未按规定停在黄色停车线框内的情况，存在则输出判定结果为是，否则输出判定结果为否；",
    "污水井倾倒": "在城市主要大街、小巷、沿街店铺门口画面中，识别出是否有人正在往雨水井中倾倒的动作，存在则输出判定结果为是，否则输出判定结果为否；",
    "垃圾桶满溢": "在城市主要大街、十字路口、小巷、沿街商铺场景下，识别出垃圾桶外地面有较大量的垃圾影响市容、通行的情况或者垃圾桶桶壁外沿儿上有垃圾的情况，存在则输出判定结果为是，不存在输出判定结果为否；",
    "垃圾堆积": "在城市主要大街、十字路口、小巷、沿街商铺场景下，识别地面有较大量的建筑材料/垃圾、生活垃圾、废弃家具占用了盲道、机动车道的情况且附近没有垃圾桶不属于垃圾桶满溢的情况，这些垃圾存在影响市容、通行情况；存在则输出判定结果为是，不存在输出判定结果为否；",
    "人员异常倒地": "请分析图像中是否存在人员倒地现象，即图像中人员因意外、疾病、外力等原因，身体呈水平或接近水平状态接触地面（如平躺、俯卧、侧卧），且姿态明显异于正常休息（如非自主倒地、无支撑物辅助）。",
    "人员异常聚集": "请分析图像中是否存在异常人群聚集现象，要求检测公共区域内人群密度是否异常增高、行为是否异常集中。特别关注那些服装、发型或其他视觉标志显示出明显不同文化特征的群体，但请注意仅将此作为辅助信息，综合判断异常聚集情况，避免将单一特征作为唯一依据。"
}

# 提取城管类的所有类别名列表
CITY_CLASSES_LIST = list(CITY_PROMPTS_MAP.keys())

# 系统提示词前缀
SYSTEM_PREFIX = "你是一个专业的视觉分析助手。"

# ===========================================

def process_file_list(file_list, image_dir, label_dir):
    """
    处理文件列表，返回数据集列表和统计信息
    """
    dataset = []
    
    # 统计计数器
    stats = {
        "images_used": 0,
        "fire_pos": 0,    # 火灾-正样本
        "fire_neg": 0,    # 火灾-负样本 (来自正常森林图)
        "city_pos": 0,    # 城管-正样本
        "city_neg": 0     # 城管-负样本
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
        stats["images_used"] += 1
        
        # === 逻辑分支 A：森林火灾数据 ===
        # 判定依据：标签包含"森林火灾" 或者 标签为空(根据之前的处理逻辑，正常森林图标签为空)
        # 注意：这里假设城管数据尽量不为空，或者如果为空也视为“无火灾”的负样本也是合理的
        
        is_fire_data = False
        if FIRE_CLASS in labels:
            is_fire_data = True
        elif not labels:
            # 如果标签为空，根据你的需求，这属于"人工风险等级为正常"，
            # 对应的问答对就是"否"，且专门作为森林火灾的负样本。
            is_fire_data = True
        
        if is_fire_data:
            # --- 处理森林火灾 ---
            if FIRE_CLASS in labels:
                # 1. 正样本：是火灾
                prompt_text = f"{SYSTEM_PREFIX}{FIRE_PROMPT}"
                dataset.append({
                    "messages": [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": "是"}
                    ],
                    "images": [abs_image_path]
                })
                stats["fire_pos"] += 1
                # 火灾的正样本图片，不作为负样本（不生成“不是火灾”）
            
            else:
                # 2. 负样本：正常图片 (标签为空)
                # 题目：请分析图像中是否发生火灾。 回答：否
                prompt_text = f"{SYSTEM_PREFIX}{FIRE_PROMPT}"
                dataset.append({
                    "messages": [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": "否"}
                    ],
                    "images": [abs_image_path]
                })
                stats["fire_neg"] += 1
        
        else:
            # === 逻辑分支 B：城管/通用数据 ===
            # 判定依据：标签里有城管的类别
            
            # 1. 构造城管正样本
            valid_pos_labels = [l for l in labels if l in CITY_PROMPTS_MAP]
            
            for label in valid_pos_labels:
                prompt_text = f"{SYSTEM_PREFIX}{CITY_PROMPTS_MAP[label]}"
                dataset.append({
                    "messages": [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": "是"}
                    ],
                    "images": [abs_image_path]
                })
                stats["city_pos"] += 1
            
            # 2. 构造城管负样本
            # 规则：只从城管类别中选负样本，不涉及森林火灾
            neg_candidates = [cls for cls in CITY_CLASSES_LIST if cls not in labels]
            
            if valid_pos_labels and neg_candidates:
                num_pos = len(valid_pos_labels)
                target_neg_count = num_pos * 0.5
                base_count = int(target_neg_count)
                final_neg_count = base_count + (1 if random.random() < (target_neg_count - base_count) else 0)
                
                count_to_sample = min(final_neg_count, len(neg_candidates))
                
                if count_to_sample > 0:
                    selected_negs = random.sample(neg_candidates, count_to_sample)
                    for label in selected_negs:
                        prompt_text = f"{SYSTEM_PREFIX}{CITY_PROMPTS_MAP[label]}"
                        dataset.append({
                            "messages": [
                                {"role": "user", "content": prompt_text},
                                {"role": "assistant", "content": "否"}
                            ],
                            "images": [abs_image_path]
                        })
                        stats["city_neg"] += 1

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

    # 1. 打乱并切分
    random.shuffle(all_files)
    # split_index = int(total_files * 0.9)
    split_index = int(total_files * 1)    
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    # 2. 生成训练集
    print("正在生成训练集...")
    train_dataset, train_stats = process_file_list(train_files, image_dir, label_dir)
    
    # 3. 生成验证集
    print("正在生成验证集...")
    val_dataset, val_stats = process_file_list(val_files, image_dir, label_dir)

    # 4. 保存文件
    train_output_path = os.path.join(output_dir, "train.json")
    val_output_path = os.path.join(output_dir, "val.json")

    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)

    with open(val_output_path, 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, ensure_ascii=False, indent=2)

    # 5. 打印统计表格
    def print_stat_row(name, stats, dataset_len):
        total_pos = stats['fire_pos'] + stats['city_pos']
        total_neg = stats['fire_neg'] + stats['city_neg']
        print(f"{name:<8} | {stats['images_used']:<8} | {total_pos:<8} ({stats['fire_pos']}火/{stats['city_pos']}城) | {total_neg:<8} ({stats['fire_neg']}火/{stats['city_neg']}城) | {dataset_len:<8}")

    print("\n" + "="*80)
    print(f"{'数据集生成统计':^78}")
    print("="*80)
    print(f"{'集合':<8} | {'图片数':<8} | {'正样本(是)':<20} | {'负样本(否)':<20} | {'总QA数':<8}")
    print("-" * 80)
    
    print_stat_row("Train", train_stats, len(train_dataset))
    print_stat_row("Val", val_stats, len(val_dataset))
    
    print("-" * 80)
    
    # 合并统计
    total_fire_pos = train_stats['fire_pos'] + val_stats['fire_pos']
    total_fire_neg = train_stats['fire_neg'] + val_stats['fire_neg']
    total_city_pos = train_stats['city_pos'] + val_stats['city_pos']
    total_city_neg = train_stats['city_neg'] + val_stats['city_neg']
    
    print(f"总计火灾样本: 正 {total_fire_pos} / 负 {total_fire_neg}")
    if total_fire_neg > 0:
        print(f"  -> 火灾正负比: {total_fire_pos/total_fire_neg:.2f}:1")
        
    print(f"总计城管样本: 正 {total_city_pos} / 负 {total_city_neg}")
    if total_city_neg > 0:
        print(f"  -> 城管正负比: {total_city_pos/total_city_neg:.2f}:1")
        
    print("="*80)
    print(f"结果已保存至: {output_dir}")

if __name__ == "__main__":
    # 默认路径配置
    default_img_dir = "./large_images"
    default_label_dir = "./label_json"
    default_output_dir = "./"  # 输出到当前目录
    
    parser = argparse.ArgumentParser(description="生成训练集和验证集数据")
    # parser.add_argument("--img_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并_deduplicated/large_images", help="图片文件夹路径")
    # parser.add_argument("--label_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并_deduplicated/label_json", help="标签JSON文件夹路径")
    # parser.add_argument("--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data", help="输出文件保存目录")
    
    parser.add_argument("--img_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao/合并数据/large_images", help="图片文件夹路径")
    parser.add_argument("--label_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao/合并数据/label_json", help="标签JSON文件夹路径")
    parser.add_argument("--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao", help="输出文件保存目录")
        
    
    args = parser.parse_args()
    
    main(args.img_dir, args.label_dir, args.output_dir)