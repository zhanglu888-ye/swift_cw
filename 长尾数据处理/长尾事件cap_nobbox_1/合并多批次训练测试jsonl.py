import os
import json
import random
from pathlib import Path
import argparse
import logging
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Any
import re 

# --- 配置区域 ---
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 事件核心配置，主要用于查找Prompt文件路径
EVENT_CONFIG: Dict[str, Dict[str, Any]] = {
    ##p0
    "垃圾事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_垃圾事件.md")},
    "倒地事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_倒地事件.md")},
    "占道经营事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_占道经营事件.md")},
    "人群聚集事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_人群异常聚集.md")},
    "雨水井倾倒污水事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_雨水井倾倒污水.md")},
    "非机动车乱停放事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_非机动车乱停放.md")},
    "森林火灾": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_森林火灾.md")}, 
    ##p1
    "暴力冲突事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_暴力冲突事件.md")},
    "公共场所躺卧事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_公共场所躺卧.md")},
    "横幅事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_横幅事件.md")},
    "城市火灾与违规焚烧事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_火灾烟雾与违规焚烧事件.md")},
    "垃圾倾倒事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_垃圾倾倒事件.md")},
    "雨水井垃圾监测事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_雨水井垃圾监测.md")},     
    "渣土车遗洒事件": {"prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_渣土车遗洒事件.md")},      
}

# 图片目录名 -> 标准事件名 (EVENT_CONFIG的键) 的映射
IMG_CATEGORY_DIR_TO_CONFIG_KEY: Dict[str, str] = {
    #p0
    "人员倒地": "倒地事件",
    "人员聚集": "人群聚集事件",
    "倾倒污水": "雨水井倾倒污水事件",
    "出店经营": "占道经营事件",
    "流动摊贩": "占道经营事件",
    "垃圾堆积": "垃圾事件",
    "垃圾桶满溢": "垃圾事件",
    "非机动车乱停放": "非机动车乱停放事件",
    "森林火灾": "森林火灾",
    ##p1
    "不文明祭祀":"城市火灾与违规焚烧事件",
    "垃圾倾倒":"垃圾倾倒事件",
    "拉横幅":"横幅事件",
    "绿地公共场所躺卧":"公共场所躺卧事件",
    "肢体冲突持械威胁":"暴力冲突事件",
    "城市烟雾火焰":"城市火灾与违规焚烧事件",
    "垃圾焚烧":"城市火灾与违规焚烧事件",
    "渣土车遗洒":"渣土车遗洒事件",
    "雨水井垃圾":"雨水井垃圾监测事件",    
}

# --- 辅助函数 ---

def load_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    """加载所有事件的Prompt模板"""
    templates = {}
    for event_type, details in config.items():
        try:
            templates[event_type] = details["prompt_path"].read_text(encoding='utf-8')
        except FileNotFoundError:
            logging.warning(f"警告: Prompt文件 '{details['prompt_path']}' 未找到！相关事件将无法处理。")
    return templates

def load_jsonl(file_path: Path) -> list:
    """从.jsonl文件加载数据"""
    if not file_path.exists():
        logging.warning(f"文件 {file_path} 不存在，将返回空列表。")
        return []
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"跳过无效的JSON行 in {file_path}: {line.strip()}")
    return data

def save_jsonl(data: list, file_path: Path):
    """将数据保存到.jsonl文件"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # logging.info(f"成功保存 {len(data)} 条记录到 {file_path}") # 在统计函数中打印信息

def find_corresponding_image(label_path: Path, labels_root: Path, images_root: Path) -> Path | None:
    """根据标签路径找到对应的图片路径"""
    try:
        relative_path = label_path.relative_to(labels_root)
        image_path_base = images_root / relative_path.with_suffix('')
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_path = image_path_base.with_suffix(ext)
            if image_path.exists():
                return image_path
        return None
    except Exception as e:
        logging.error(f"查找图片时发生错误 for {label_path}: {e}")
        return None

def split_data(data: list, train_ratio=0.8, val_ratio=0.05, seed=42) -> tuple[list, list, list]:
    """将数据按比例划分为训练、验证和测试集"""
    if not data:
        return [], [], []
    random.seed(seed)
    random.shuffle(data)
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_set = data[:n_train]
    val_set = data[n_train : n_train + n_val]
    test_set = data[n_train + n_val:]
    return train_set, val_set, test_set

# --- 新增统计函数 ---
def count_events_by_prompt(data: list) -> Dict[str, int]:
    """根据user prompt内容统计数据中各事件类型的数量"""
    counts = defaultdict(int)
    # 正则表达式，用于从prompt中提取事件名称，例如 “垃圾事件”
    pattern = re.compile(r'与“([^”]+)”相关的情况')
    for item in data:
        try:
            user_prompt = item['messages'][0]['content']
            match = pattern.search(user_prompt)
            if match:
                # 您的prompt中可能是"人员异常倒地"，而我们的内部键是"倒地事件"，
                # 为了统计清晰，直接使用prompt中的名称。
                event_name = match.group(1)
                counts[event_name] += 1
            else:
                counts["[未知事件]"] += 1
        except (KeyError, IndexError):
            counts["[格式错误]"] += 1
    return counts

def print_detailed_statistics(title: str, data: list, output_path: Path):
    """保存文件并打印详细的分类统计信息"""
    # --- 修改处：在保存前打乱数据顺序 ---
    random.shuffle(data) 
    # ----------------------------------
    
    save_jsonl(data, output_path)
    logging.info(f"  {title}: {len(data)} 条")
    event_counts = count_events_by_prompt(data)
    if not event_counts:
        return
    # 排序以保证输出顺序一致
    for event, count in sorted(event_counts.items()):
        logging.info(f"    - {event}: {count} 条")


# --- 主逻辑 ---

def main():
    parser = argparse.ArgumentParser(description="处理第二批数据，划分并与第一批_cleaned数据合并。")
    # parser.add_argument("--labels_v2_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/第一批_cleaned/labels_kept", help="第二批数据的标签根目录。")
    # parser.add_argument("--images_v2_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/第一批_cleaned/images_kept", help="第二批数据的图片根目录。")
    # parser.add_argument("--data_v1_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述", help="第一批_cleanedjsonl数据所在的目录。")
    # parser.add_argument("--output_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data", help="合并后v2版本jsonl的输出目录。")
    
    # parser.add_argument("--labels_v2_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p0_第三批_cleaned/labels_kept", help="第二批数据的标签根目录。")
    # parser.add_argument("--images_v2_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p0_第三批_cleaned/images_kept", help="第二批数据的图片根目录。")
    # parser.add_argument("--data_v1_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data", help="第一批_cleanedjsonl数据所在的目录。")
    # parser.add_argument("--output_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data", help="合并后v2版本jsonl的输出目录。")    
    
    
    parser.add_argument("--labels_v2_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第二批_260209_cleaned/labels_kept", help="第二批数据的标签根目录。")
    parser.add_argument("--images_v2_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第二批_260209_cleaned/images_kept", help="第二批数据的图片根目录。")
    parser.add_argument("--data_v1_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p1", help="第一批_cleanedjsonl数据所在的目录。")
    parser.add_argument("--output_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p1", help="合并后v2版本jsonl的输出目录。")    
       
    
    
    
    
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载所有事件的prompt
    prompts = load_prompts(EVENT_CONFIG)
    if not prompts:
        logging.critical("未能加载任何prompt，无法继续。")
        return

    # 2. 遍历并处理第二批数据
    logging.info(f"开始处理第二批数据，源目录: {args.labels_v2_dir}")
    positive_samples_v2 = []
    negative_samples_v2 = []
    all_json_files = list(args.labels_v2_dir.rglob('*.json'))
    
    for label_path in tqdm(all_json_files, desc="处理第二批标签"):
        is_positive = "正样本" in label_path.parts
        is_negative = "负样本" in label_path.parts
        if not (is_positive or is_negative): continue

        try:
            with open(label_path, 'r', encoding='utf-8') as f: label_data = json.load(f)
        except (json.JSONDecodeError, IOError): continue

        image_path = find_corresponding_image(label_path, args.labels_v2_dir, args.images_v2_dir)
        if not image_path: continue

        try:
            event_dir_name = label_path.relative_to(args.labels_v2_dir).parts[0]
            config_key = IMG_CATEGORY_DIR_TO_CONFIG_KEY.get(event_dir_name)
            if not config_key or config_key not in prompts: continue
            user_prompt = prompts[config_key]
        except (IndexError, KeyError): continue
            
        assistant_content_str = json.dumps(label_data, ensure_ascii=False, indent=2)
        jsonl_record = {
            "messages": [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": assistant_content_str}],
            "images": [str(image_path)]
        }
        
        if is_positive: positive_samples_v2.append(jsonl_record)
        else: negative_samples_v2.append(jsonl_record)

    logging.info(f"第二批数据处理完成：正样本 {len(positive_samples_v2)} 条, 负样本 {len(negative_samples_v2)} 条。")

    # 3. 划分第二批数据
    logging.info("开始划分第二批数据...")
    pos_train, pos_val, pos_test = split_data(positive_samples_v2)
    neg_train, neg_val, neg_test = split_data(negative_samples_v2)
    logging.info("划分完成。")
    
    # 4. 加载第一批_cleaned数据
    logging.info(f"开始加载第一批_cleaned数据，源目录: {args.data_v1_dir}")
    # train_pos_v1 = load_jsonl(args.data_v1_dir / "train_zhengyb_converted_longprompt.jsonl")
    # train_neg_v1 = load_jsonl(args.data_v1_dir / "train_fuyb_converted_longprompt.jsonl")
    # val_pos_v1 = load_jsonl(args.data_v1_dir / "val_zhengyb_converted_longprompt.jsonl")
    # val_neg_v1 = load_jsonl(args.data_v1_dir / "val_fuyb_converted_longprompt.jsonl")

    train_pos_v1 = load_jsonl(args.data_v1_dir / "p1_train_zhengyangben_v2.jsonl")
    train_neg_v1 = load_jsonl(args.data_v1_dir / "p1_train_fuyangben_v2.jsonl")
    val_pos_v1 = load_jsonl(args.data_v1_dir / "p1_val_zhengyangben_v2.jsonl")
    val_neg_v1 = load_jsonl(args.data_v1_dir / "p1_val_fuyangben_v2.jsonl")
    test_v1 = load_jsonl(args.data_v1_dir / "p1_test_v2.jsonl")



    # 5. 合并数据
    logging.info("开始合并数据...")
    train_pos_v2 = train_pos_v1 + pos_train
    train_neg_v2 = train_neg_v1 + neg_train
    val_pos_v2 = val_pos_v1 + pos_val
    val_neg_v2 = val_neg_v1 + neg_val
    test_v2 = pos_test + neg_test + test_v1

    # 6. 保存所有v2版本的文件并打印详细统计
    logging.info("="*50)
    logging.info(f"所有处理已完成！最终文件统计 (保存至 {args.output_dir}):")
    
    print_detailed_statistics(
        "p1_train_zhengyangben_v2.jsonl",
        train_pos_v2,
        args.output_dir / "p1_train_zhengyangben_v3.jsonl"
    )
    print_detailed_statistics(
        "p1_train_fuyangben_v2.jsonl",
        train_neg_v2,
        args.output_dir / "p1_train_fuyangben_v3.jsonl"
    )
    print_detailed_statistics(
        "p1_val_zhengyangben_v2.jsonl",
        val_pos_v2,
        args.output_dir / "p1_val_zhengyangben_v3.jsonl"
    )
    print_detailed_statistics(
        "p1_val_fuyangben_v2.jsonl",
        val_neg_v2,
        args.output_dir / "p1_val_fuyangben_v3.jsonl"
    )
    print_detailed_statistics(
        "p1_test_v2.jsonl",
        test_v2,
        args.output_dir / "p1_test_v3.jsonl"
    )
    
    logging.info("="*50)


if __name__ == "__main__":
    main()