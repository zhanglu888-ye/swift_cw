import json
import os
import argparse
import random
import glob
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域：Prompt 读取逻辑 =================

# 你的 Prompt 文件映射表
CATEGORY_TO_MD = {
    "垃圾桶满溢": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_垃圾事件.md",
    "垃圾堆积": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_垃圾事件.md",
    "人员异常倒地": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_倒地事件.md",
}

# 预加载 Prompt 内容缓存
PROMPT_CACHE = {}

def load_prompt_from_md(file_path):
    """读取并缓存 markdown 内容"""
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
    """
    根据类别返回对应的 Prompt。
    
    参数:
    - category: 当前 JSON 中的异常类型 (如 '垃圾堆积', '无异常')
    - context_task: 上下文任务类型 ('garbage' 或 'fall')，用于处理 '无异常' 的情况
    """
    
    target_md_path = None
    
    # 1. 尝试直接匹配
    if category in CATEGORY_TO_MD:
        target_md_path = CATEGORY_TO_MD[category]
        
    # 2. 处理 '无异常' 或 未知类别的兜底逻辑
    else:
        # 如果是垃圾任务目录下的无异常 -> 用垃圾 Prompt
        if context_task == "garbage":
            target_md_path = CATEGORY_TO_MD["垃圾堆积"] # 借用垃圾堆积的MD
        # 如果是倒地任务目录下的无异常 -> 用倒地 Prompt
        elif context_task == "fall":
            target_md_path = CATEGORY_TO_MD["人员异常倒地"]
            
    if target_md_path:
        return load_prompt_from_md(target_md_path)
    
    print(f"Warning: No prompt found for category '{category}' in task '{context_task}'")
    return ""

# ================= 核心逻辑 =================

def extract_core_info(json_data):
    """从不同结构的 JSON 中提取 file_name 和 response"""
    file_name = json_data.get("file_name")
    
    # 提取 assistant 回复
    if "analyses" in json_data:
        response_list = json_data["analyses"][0].get("event_analyses", [])
    elif "event_analyses" in json_data:
        response_list = json_data["event_analyses"]
    else:
        return None, None, None
        
    if not response_list: return None, None, None

    # 获取异常类型用于匹配 Prompt
    # 假设取第一个事件的类型
    category = response_list[0].get("comprehensive_assessment", {}).get("异常类型", "无异常")
    
    return file_name, response_list, category

def load_sft_from_dir(json_dir, image_dir, task_type):
    """
    从指定目录加载 SFT 数据
    """
    dataset = []
    # 递归查找所有 .json 文件
    files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"[{task_type}] 正在加载 SFT 数据: {len(files)} 条...")
    
    for f_path in tqdm(files):
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_name, response_list, category = extract_core_info(data)
            if not file_name or not response_list: continue
            
            # 检查图片是否存在
            image_path = os.path.join(image_dir, file_name)
            if not os.path.exists(image_path):
                continue
            
            # 获取对应的 Prompt
            prompt_content = get_prompt_content(category, context_task=task_type)
            if not prompt_content: continue
            
            # 构造标准 SFT 条目
            entry = {
                "images": [image_path],
                "messages": [
                    {"role": "user", "content": prompt_content},
                    {"role": "assistant", "content": json.dumps({"event_analyses": response_list}, ensure_ascii=False)}
                ],
                "source": task_type
            }
            dataset.append(entry)
            
        except Exception as e:
            continue
            
    return dataset

def load_dpo_from_dir(json_dir, task_type):
    """
    从指定目录加载 DPO (Rejected) 数据
    """
    dataset = []
    files = glob.glob(os.path.join(json_dir, "*_rejected.json"))
    print(f"[{task_type}] 正在加载 DPO 数据: {len(files)} 条...")
    
    for f_path in tqdm(files):
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_path = data.get("image_path")
            if not os.path.exists(image_path): continue
            
            chosen_obj = data.get("chosen")
            rejected_versions = data.get("rejected_versions", [])
            
            if not chosen_obj or not rejected_versions: continue
            
            # 获取 Prompt
            # 注意：rejected json 中的 chosen 通常是单个对象或含 event_analyses
            # 这里我们需要提取出类别
            if "event_analyses" in chosen_obj: 
                # 如果是完整结构
                cat = chosen_obj["event_analyses"][0]["comprehensive_assessment"]["异常类型"]
                chosen_str = json.dumps(chosen_obj, ensure_ascii=False)
            elif "comprehensive_assessment" in chosen_obj:
                # 如果是简化结构（直接是对象）
                cat = chosen_obj["comprehensive_assessment"]["异常类型"]
                # 包装回 event_analyses 列表结构以符合 SFT 格式
                chosen_str = json.dumps({"event_analyses": [chosen_obj]}, ensure_ascii=False)
            else:
                continue
                
            prompt_content = get_prompt_content(cat, context_task=task_type)
            
            # 为每个 reject 版本生成一条训练数据
            for rej in rejected_versions:
                # 同样处理 reject 的结构
                if "comprehensive_assessment" in rej and "event_analyses" not in rej:
                     rej_str = json.dumps({"event_analyses": [rej]}, ensure_ascii=False)
                else:
                     rej_str = json.dumps(rej, ensure_ascii=False)

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
            # print(f"Error processing DPO {f_path}: {e}")
            continue
            
    return dataset

def save_jsonl(data, filepath):
    print(f"正在保存至 {filepath} ...")
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"完成。共 {len(data)} 条。")

def main():
    parser = argparse.ArgumentParser()
    
    # SFT 输入目录 (双目录)
    parser.add_argument("--sft_garbage_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/垃圾堆积_labels")
    parser.add_argument("--sft_fall_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/人员倒地_labels")
    
    # 图片输入目录 (双目录)
    parser.add_argument("--img_garbage_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/图片/垃圾堆积")
    parser.add_argument("--img_fall_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/图片/人员倒地")
    
    # DPO 输入目录 (双目录)
    parser.add_argument("--dpo_garbage_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/垃圾堆积_labels_reject")
    parser.add_argument("--dpo_fall_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/人员倒地_labels_reject")
    
    # 输出目录
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # -------------------------------------------------
    # 1. 构建 SFT 数据集
    # -------------------------------------------------
    print("\n=== Step 1: Loading SFT Data ===")
    sft_garbage = load_sft_from_dir(args.sft_garbage_dir, args.img_garbage_dir, "garbage")
    sft_fall = load_sft_from_dir(args.sft_fall_dir, args.img_fall_dir, "fall")
    
    all_sft = sft_garbage + sft_fall
    random.shuffle(all_sft)
    
    # 90/10 划分
    sft_split_idx = int(len(all_sft) * 0.9)
    sft_train = all_sft[:sft_split_idx]
    sft_val = all_sft[sft_split_idx:]
    
    save_jsonl(sft_train, os.path.join(args.output_dir, "sft_train.jsonl"))
    save_jsonl(sft_val, os.path.join(args.output_dir, "sft_val.jsonl"))
    
    # -------------------------------------------------
    # 2. 构建 DPO 数据集
    # -------------------------------------------------
    print("\n=== Step 2: Loading DPO Data ===")
    dpo_garbage = load_dpo_from_dir(args.dpo_garbage_dir, "garbage")
    dpo_fall = load_dpo_from_dir(args.dpo_fall_dir, "fall")
    
    all_dpo_pairs = dpo_garbage + dpo_fall
    random.shuffle(all_dpo_pairs)
    
    # 90/10 划分
    dpo_split_idx = int(len(all_dpo_pairs) * 0.9)
    dpo_train_pairs = all_dpo_pairs[:dpo_split_idx]
    dpo_val_pairs = all_dpo_pairs[dpo_split_idx:]
    
    # -------------------------------------------------
    # 3. SFT 混排 (20% 占比)
    # -------------------------------------------------
    print("\n=== Step 3: Mixing SFT into DPO Train Set ===")
    
    # 计算需要混入的 SFT 数量 (DPO的25%，即总量的20%)
    num_dpo = len(dpo_train_pairs)
    num_sft_to_mix = int(num_dpo * 0.25)
    
    if len(sft_train) > 0:
        if len(sft_train) >= num_sft_to_mix:
            sft_mix_samples = random.sample(sft_train, num_sft_to_mix)
        else:
            sft_mix_samples = random.choices(sft_train, k=num_sft_to_mix)
            
        # 转换为 DPO 格式 (rejected=None)
        dpo_sft_entries = []
        for item in sft_mix_samples:
            dpo_entry = {
                "images": item["images"],
                "messages": [item["messages"][0]], # User Prompt
                "chosen": item["messages"][1],     # Assistant Response
                "rejected": None
            }
            dpo_sft_entries.append(dpo_entry)
            
        final_dpo_train = dpo_train_pairs + dpo_sft_entries
        random.shuffle(final_dpo_train)
    else:
        final_dpo_train = dpo_train_pairs
        print("Warning: No SFT data available for mixing.")
    
    save_jsonl(final_dpo_train, os.path.join(args.output_dir, "dpo_train_mixed.jsonl"))
    save_jsonl(dpo_val_pairs, os.path.join(args.output_dir, "dpo_val.jsonl"))

    print("\n=== Summary ===")
    print(f"SFT Train: {len(sft_train)}")
    print(f"DPO Train (Mixed): {len(final_dpo_train)}")

if __name__ == "__main__":
    main()