import json
import os
import argparse
import time
import glob
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 (API & 类别定义) =================

VALID_CATEGORIES_MAP = {
    "garbage": ["垃圾堆积", "垃圾桶满溢", "无异常"],
    "fall": ["人员异常倒地", "无异常"]
}

def get_isolated_prompt(original_category, task_type):
    """
    根据任务类型生成对抗 Prompt (保持之前的逻辑)
    """
    is_normal = "无异常" in original_category
    
    if task_type == "garbage":
        if is_normal:
            strategy = "【强行幻觉模式】原图无异常。请无中生有，将阴影/杂物描述为“垃圾堆积”或“垃圾桶满溢”。禁止输出“无异常”。"
        else:
            strategy = f"【混淆模式】原始是{original_category}。请生成：1.类别混淆(如堆积变满溢)；2.强制漏报(说成无异常)。"
        
        return f"""你是一个垃圾检测对抗助手。世界里只有垃圾概念。
        异常类型仅限：{json.dumps(VALID_CATEGORIES_MAP['garbage'], ensure_ascii=False)}。
        {strategy}
        请返回包含【两个】JSON对象的列表。"""

    elif task_type == "fall":
        if is_normal:
            strategy = "【强行幻觉模式】原图无异常。请指鹿为马，将休息的人或地上的物体描述为“人员异常倒地”。禁止输出“无异常”。"
        else:
            strategy = f"【混淆模式】原始是{original_category}。请生成：1.强行合理化(说是休息)；2.视觉无视(说无人)。"

        return f"""你是一个倒地检测对抗助手。世界里只有人和倒地概念。
        异常类型仅限：{json.dumps(VALID_CATEGORIES_MAP['fall'], ensure_ascii=False)}。
        {strategy}
        请返回包含【两个】JSON对象的列表。"""
    
    return ""

def call_api_generate(args, chosen_json_str, original_category, task_type):
    """API 生成兜底函数"""
    prompt = get_isolated_prompt(original_category, task_type)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}", "x-proxy-secret": args.proxy_secret}
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": f"基于此生成负样本:\n{chosen_json_str}"}],
        "temperature": 0.85, "max_tokens": 4096
    }

    for _ in range(args.retries):
        try:
            response = requests.post(args.api_base_url.rstrip('/')+"/chat/completions", headers=headers, json=payload, timeout=60)
            if response.status_code != 200: continue
            content = response.json()['choices'][0]['message']['content'].strip()
            
            if "```json" in content: content = content.replace("```json", "").replace("```", "")
            if "[" in content and "]" in content:
                json_str = content[content.find('['):content.rfind(']')+1]
                return json.loads(json_str)
        except:
            time.sleep(1)
    return []

# ================= 核心处理逻辑 =================

def extract_core_data(json_data):
    """提取核心分析数据的通用函数"""
    # 适配不同层级的 JSON 结构
    if "event_analyses" in json_data:
        items = json_data["event_analyses"]
        return items[0] if items else None
    
    analyses = json_data.get("analyses", [])
    if analyses:
        item = analyses[0] if isinstance(analyses, list) else analyses
        event_list = item.get("event_analyses", [])
        return event_list[0] if event_list else None
    return None

def get_base_id(filename):
    """
    从文件名中提取基础ID。
    例如: 
    - aaab1_model1.json -> aaab1
    - aaab1_model2.json -> aaab1
    - aaab1.json -> aaab1
    """
    base = os.path.basename(filename)
    # 移除 .json 后缀
    name_no_ext = os.path.splitext(base)[0]
    # 移除 _modelX 后缀 (如果有)
    # 使用正则匹配 _model 后跟数字的情况，或者直接 split
    if "_model" in name_no_ext:
        return name_no_ext.split("_model")[0]
    return name_no_ext

def find_hard_negatives(base_id, correct_category, candidate_dir, task_type, correct_filename):
    """
    在候选文件夹中查找【判定结果不同】的 JSON
    """
    found_rejects = []
    
    # 搜索模式：base_id + 任意后缀 + .json
    # 例如 base_id="aaab1", 匹配 "aaab1_model1.json", "aaab1_model3.json"
    search_pattern = os.path.join(candidate_dir, f"{base_id}*.json")
    candidates = glob.glob(search_pattern)
    
    for cand_path in candidates:
        # 跳过自己 (正确的文件本身不参与比较)
        if os.path.basename(cand_path) == correct_filename:
            continue
            
        try:
            with open(cand_path, 'r', encoding='utf-8') as f:
                cand_data = json.load(f)
            
            cand_core = extract_core_data(cand_data)
            if not cand_core: continue
            
            cand_cat = cand_core.get("comprehensive_assessment", {}).get("异常类型", "无异常")
            
            # 【核心逻辑】：只有当 候选类别 != 正确类别 时，才是 Hard Negative
            if cand_cat != correct_category:
                # 校验：候选类别必须在当前任务的白名单内 (防止跨任务数据污染)
                allowed = VALID_CATEGORIES_MAP.get(task_type, [])
                
                # 特殊逻辑：如果是"无异常"，任何任务都允许
                if cand_cat in allowed or cand_cat == "无异常":
                    found_rejects.append(cand_core)
                    
        except Exception as e:
            # print(f"Skipping candidate {cand_path}: {e}")
            continue
            
    return found_rejects

def process_item(args, correct_json_path):
    try:
        # 1. 读取【正确】的 JSON
        with open(correct_json_path, 'r', encoding='utf-8') as f:
            correct_data = json.load(f)
        
        # 提取文件名和ID
        correct_filename = os.path.basename(correct_json_path)
        base_id = get_base_id(correct_filename)
        
        # 提取图片路径：优先从 json 获取，如果没有则通过 ID 拼接
        file_name = correct_data.get("file_name", f"{base_id}.jpg")
        image_path = os.path.join(args.image_dir, file_name)
        
        correct_core = extract_core_data(correct_data)
        if not correct_core: return False
        
        original_category = correct_core.get("comprehensive_assessment", {}).get("异常类型")
        
        # 2. 策略A：从历史错误中查找 (Hard Negatives)
        final_rejects = find_hard_negatives(
            base_id, 
            original_category, 
            args.candidate_json_dir, 
            args.task_type,
            correct_filename
        )
        
        # 3. 策略B：数量不足则调用 API 生成 (Synthetic Negatives)
        needed_count = 2 - len(final_rejects)
        if needed_count > 0:
            api_rejects = call_api_generate(
                args, 
                json.dumps(correct_core, ensure_ascii=False), 
                original_category, 
                args.task_type
            )
            if api_rejects:
                final_rejects.extend(api_rejects[:needed_count])
        
        # 如果还是空的，跳过
        if not final_rejects: return False

        # 4. 构造 DPO 数据
        output_data = {
            "file_name": file_name,
            "original_category": original_category,
            "task_type": args.task_type,
            "image_path": image_path,
            "chosen": correct_core,
            "rejected_versions": final_rejects[:2] # 限制最多2个
        }
        
        # 5. 保存结果
        output_filename = f"{base_id}_rejected.json"
        save_path = os.path.join(args.output_dir, output_filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        return True

    except Exception as e:
        print(f"Error processing {correct_json_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Any-Model Hybrid DPO Generator")
    
    # 路径配置
    parser.add_argument("--correct_json_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/人员倒地部分错_修正", help="存放人工挑选出的【正确】JSON的目录")
    parser.add_argument("--candidate_json_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/人员倒地部分错_待复查", help="存放【所有】模型输出JSON的目录（用于查找错误版本）")
    parser.add_argument("--image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/人员倒地部分错_修正", help="图片文件夹路径")
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/人员倒地部分错_修正_reject", help="结果保存目录")
    
    
    parser.add_argument("--task_type", type=str, choices=["garbage", "fall"], default="garbage")
    
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument("--api_key", type=str, default="McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ", help="API密钥。")                                                                                                                                                                             
    parser.add_argument("--api_base_url", type=str, default="https://travel-belong-disable-excellence.trycloudflare.com/v1/", help="API的基础URL。")
    parser.add_argument("--proxy_secret", type=str, default="my_secret_password_sensoro_123", help="x-proxy-secret 头的值。")

    parser.add_argument("--workers", type=int, default=1, help="并发处理的线程数。")
    parser.add_argument("--retries", type=int, default=3, help="API调用失败后的最大重试次数。")    
    
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 扫描正确目录下的所有JSON
    correct_files = [
        os.path.join(args.correct_json_dir, f) 
        for f in os.listdir(args.correct_json_dir) 
        if f.endswith('.json')
    ]
    
    print(f"=== 混合负样本生成器 ===")
    print(f"任务类型: {args.task_type}")
    print(f"正确样本源: {len(correct_files)} 个 (来自 {args.correct_json_dir})")
    print(f"备选错误源: {args.candidate_json_dir}")
    print(f"正在处理...")

    count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_item, args, p): p for p in correct_files}
        for future in tqdm(as_completed(futures), total=len(correct_files)):
            if future.result(): count += 1

    print(f"\n生成完成! 共 {count} 条数据保存至 {args.output_dir}")

if __name__ == "__main__":
    main()