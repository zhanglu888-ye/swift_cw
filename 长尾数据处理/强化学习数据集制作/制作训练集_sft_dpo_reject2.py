import json
import os
import argparse
import random
import glob
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域：Prompt 读取逻辑 =================

CATEGORY_TO_MD = {
    "垃圾桶满溢": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_垃圾事件.md",
    "垃圾堆积": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_垃圾事件.md",
    "人员异常倒地": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_倒地事件.md",
}

PROMPT_CACHE = {}

def load_prompt_from_md(file_path):
    if file_path in PROMPT_CACHE:
        return PROMPT_CACHE[file_path]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            PROMPT_CACHE[file_path] = content
            return content
    except Exception as e:
        print(f"Error loading prompt {file_path}: {e}")
        return ""

def get_prompt_content(category, context_task=None):
    target_md_path = None
    if category in CATEGORY_TO_MD:
        target_md_path = CATEGORY_TO_MD[category]
    else:
        if context_task == "garbage":
            target_md_path = CATEGORY_TO_MD["垃圾堆积"]
        elif context_task == "fall":
            target_md_path = CATEGORY_TO_MD["人员异常倒地"]
            
    if target_md_path:
        return load_prompt_from_md(target_md_path)
    return ""

# ================= 核心逻辑 =================

def wrap_with_event_analyses(obj):
    """确保输出符合 {"event_analyses": [...]} 的结构"""
    if isinstance(obj, dict) and "event_analyses" in obj:
        return json.dumps(obj, ensure_ascii=False)
    # 如果直接是内容字典，则包装一层列表
    return json.dumps({"event_analyses": [obj]}, ensure_ascii=False)

def extract_core_info(json_data):
    file_name = json_data.get("file_name")
    if "analyses" in json_data:
        response_list = json_data["analyses"][0].get("event_analyses", [])
    elif "event_analyses" in json_data:
        response_list = json_data["event_analyses"]
    else:
        return None, None, None
        
    if not response_list: return None, None, None
    category = response_list[0].get("comprehensive_assessment", {}).get("异常类型", "无异常")
    return file_name, response_list, category

def load_sft_from_dir(json_dir, image_dir, task_type):
    dataset = []
    files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"[{task_type}] 正在加载 SFT 数据: {len(files)} 条...")
    
    for f_path in tqdm(files):
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            file_name, response_list, category = extract_core_info(data)
            if not file_name or not response_list: continue
            image_path = os.path.join(image_dir, file_name)
            if not os.path.exists(image_path): continue
            prompt_content = get_prompt_content(category, context_task=task_type)
            if not prompt_content: continue
            
            entry = {
                "images": [image_path],
                "messages": [
                    {"role": "user", "content": prompt_content},
                    {"role": "assistant", "content": wrap_with_event_analyses({"event_analyses": response_list})}
                ],
                "source": task_type
            }
            dataset.append(entry)
        except: continue
    return dataset

def load_dpo_from_dir(json_dir, task_type):
    """
    针对 1 Chosen + 2 Rejected 结构进行解析
    """
    dataset = []
    # 查找所有 _rejected.json 后缀的文件
    files = glob.glob(os.path.join(json_dir, "*_rejected.json"))
    print(f"[{task_type}] 正在加载 DPO 数据 (1:2 模式): {len(files)} 条源文件...")
    
    for f_path in tqdm(files):
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_path = data.get("image_path")
            if not image_path or not os.path.exists(image_path): continue
            
            # 提取正例
            chosen_obj = data.get("chosen")
            # 提取负例列表
            rejected_versions = data.get("rejected_versions", [])
            
            if not chosen_obj or not rejected_versions: continue
            
            # 获取类别用于匹配 Prompt (优先使用 original_category)
            cat = data.get("original_category")
            if not cat:
                # 兜底：从 chosen 内部结构找
                if "comprehensive_assessment" in chosen_obj:
                    cat = chosen_obj["comprehensive_assessment"]["异常类型"]
                elif "event_analyses" in chosen_obj:
                    cat = chosen_obj["event_analyses"][0]["comprehensive_assessment"]["异常类型"]

            prompt_content = get_prompt_content(cat, context_task=task_type)
            if not prompt_content: continue

            # 处理 Chosen 字符串
            chosen_str = wrap_with_event_analyses(chosen_obj)

            # --- 关键修改：遍历所有 rejected 版本 (通常为2个) ---
            for rej_obj in rejected_versions:
                rej_str = wrap_with_event_analyses(rej_obj)

                # 每个 Rejected 对应生成一行 JSONL 条目
                entry = {
                    "images": [image_path],
                    "messages": [
                        {"role": "user", "content": prompt_content}
                    ],
                    "chosen": {
                        "role": "assistant", 
                        "content": chosen_str
                    },
                    "rejected": {
                        "role": "assistant", 
                        "content": rej_str
                    }
                }
                dataset.append(entry)
                
        except Exception as e:
            continue
            
    return dataset

def save_jsonl(data, filepath):
    print(f"正在保存至 {filepath} ...")
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"完成。共计 {len(data)} 条记录。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_garbage_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/垃圾堆积_labels")
    parser.add_argument("--sft_fall_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/人员倒地_labels")
    parser.add_argument("--img_garbage_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/图片/垃圾堆积")
    parser.add_argument("--img_fall_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/图片/人员倒地")
    parser.add_argument("--dpo_garbage_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/垃圾堆积_labels_reject")
    parser.add_argument("--dpo_fall_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/人员倒地_labels_reject")
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载 SFT
    sft_garbage = load_sft_from_dir(args.sft_garbage_dir, args.img_garbage_dir, "garbage")
    sft_fall = load_sft_from_dir(args.sft_fall_dir, args.img_fall_dir, "fall")
    all_sft = sft_garbage + sft_fall
    random.shuffle(all_sft)
    
    sft_split_idx = int(len(all_sft) * 0.9)
    sft_train = all_sft[:sft_split_idx]
    sft_val = all_sft[sft_split_idx:]
    
    # 2. 加载 DPO (1:2 逻辑)
    print("\n=== 加载 DPO 数据 (包含 2 个 Rejected) ===")
    dpo_garbage = load_dpo_from_dir(args.dpo_garbage_dir, "garbage")
    dpo_fall = load_dpo_from_dir(args.dpo_fall_dir, "fall")
    
    all_dpo_pairs = dpo_garbage + dpo_fall
    random.shuffle(all_dpo_pairs)
    
    dpo_split_idx = int(len(all_dpo_pairs) * 0.9)
    dpo_train_pairs = all_dpo_pairs[:dpo_split_idx]
    dpo_val_pairs = all_dpo_pairs[dpo_split_idx:]
    
    # 3. 混合 SFT (20% 占比)
    num_dpo = len(dpo_train_pairs)
    num_sft_to_mix = int(num_dpo * 0.25)
    
    if len(sft_train) > 0:
        sft_mix_samples = random.choices(sft_train, k=num_sft_to_mix)
        dpo_sft_entries = []
        for item in sft_mix_samples:
            dpo_entry = {
                "images": item["images"],
                "messages": [item["messages"][0]],
                "chosen": item["messages"][1],
                "rejected": None
            }
            dpo_sft_entries.append(dpo_entry)
        final_dpo_train = dpo_train_pairs + dpo_sft_entries
        random.shuffle(final_dpo_train)
    else:
        final_dpo_train = dpo_train_pairs
    
    # 保存结果
    save_jsonl(final_dpo_train, os.path.join(args.output_dir, "dpo_train_mixed_strict_reject2.jsonl"))
    save_jsonl(dpo_val_pairs, os.path.join(args.output_dir, "dpo_val_strict_reject2.jsonl"))

    print("\n=== 统计报告 ===")
    print(f"原始 DPO 样本数 (1 Chosen : 2 Rejected 展开后): {len(all_dpo_pairs)}")
    print(f"最终混合训练集大小 (DPO 对数 + 混入的 SFT): {len(final_dpo_train)}")

if __name__ == "__main__":
    main()