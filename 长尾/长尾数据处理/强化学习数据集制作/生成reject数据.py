import json
import os
import argparse
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================

# 定义严格的类别白名单
VALID_CATEGORIES_MAP = {
    "garbage": ["垃圾堆积", "垃圾桶满溢", "无异常"],
    "fall": ["人员异常倒地", "无异常"]
}

def get_isolated_prompt(original_category, task_type):
    """
    根据任务类型和原始类别，生成针对性的对抗 Prompt
    """
    is_normal = "无异常" in original_category

    # ================= 1. 垃圾检测专用 Prompt =================
    if task_type == "garbage":
        
        # 情况 A: 原图是“无异常”（干净） -> 负样本必须是“幻觉”（有垃圾）
        if is_normal:
            strategy = """
【强行幻觉模式】
原始类别是：【无异常】（即画面干净/合规）。
你的任务是**无中生有**，必须声称画面中存在违规垃圾。
请生成两个版本的负样本：
- 版本A（视觉幻觉）：将地面上的阴影、落叶、井盖或合规的车辆/设备，强行描述为“垃圾堆积”。
- 版本B（逻辑扭曲）：承认画面是施工/作业现场，但忽略排除条件，强行判定为“垃圾堆积”或“垃圾桶满溢”。
**注意：Rejected 的异常类型必须是“垃圾堆积”或“垃圾桶满溢”，绝对禁止输出“无异常”。**
"""
        # 情况 B: 原图是“有垃圾” -> 负样本必须是“漏报”或“混淆”
        else:
            strategy = f"""
【漏报与混淆模式】
原始类别是：【{original_category}】。
请生成两个版本的负样本：
- 版本A（类别混淆）：将“垃圾堆积”说成“垃圾桶满溢”（或反之）。
- 版本B（强制漏报）：无视明显的垃圾，强行辩解称“未达到堆积标准”或“有人看管”，判定为“无异常”。
"""

        return f"""
你是一个【垃圾检测场景】的对抗训练助手。你的任务是基于给定的正确分析，生成具有迷惑性的负样本。

【严格约束】
1. 你的世界里只有垃圾相关的概念。
2. comprehensive_assessment 中的 "异常类型" 只能从以下列表中选择：
   {json.dumps(VALID_CATEGORIES_MAP['garbage'], ensure_ascii=False)}
3. **绝对禁止**出现“人员倒地”等无关判定。

{strategy}

【输出格式】
返回一个包含两个JSON对象的列表。不要输出Markdown。
"""

    # ================= 2. 倒地检测专用 Prompt =================
    elif task_type == "fall":
        
        # 情况 A: 原图是“无异常”（正常人） -> 负样本必须是“幻觉”（有人倒地）
        if is_normal:
            strategy = """
【强行幻觉模式】
原始类别是：【无异常】（即人员正常活动/休息）。
你的任务是**指鹿为马**，必须声称画面中存在“人员异常倒地”。
请生成两个版本的负样本：
- 版本A（行为误判）：将画面中正在【坐着、蹲着、躺在长椅上、做瑜伽/运动】的正常人，强行描述为“突发疾病倒地”或“异常摔倒”。
- 版本B（视觉误认）：如果画面无人，将地上的【衣服、影子、人形立牌、雕塑】强行描述为“倒地的人员”。
**注意：Rejected 的异常类型必须是“人员异常倒地”，绝对禁止输出“无异常”。**
"""
        # 情况 B: 原图是“有人倒地” -> 负样本必须是“漏报”
        else:
            strategy = f"""
【漏报与混淆模式】
原始类别是：【{original_category}】。
请生成两个版本的负样本：
- 版本A（强行合理化）：承认人躺在地上，但辩称“他只是在休息/睡觉/摆拍”，判定为“无异常”。
- 版本B (无视主体)：忽略倒地的人，声称画面中“只有正常行走的行人”，判定为“无异常”。
"""

        return f"""
你是一个【人员倒地检测场景】的对抗训练助手。你的任务是基于给定的正确分析，生成具有迷惑性的负样本。

【严格约束】
1. 你的世界里只有“人”和“异常倒地”的概念。
2. comprehensive_assessment 中的 "异常类型" 只能从以下列表中选择：
   {json.dumps(VALID_CATEGORIES_MAP['fall'], ensure_ascii=False)}
3. **绝对禁止**输出“垃圾”相关类别。

{strategy}

【输出格式】
返回一个包含两个JSON对象的列表。不要输出Markdown。
"""
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def validate_and_fix_categories(rejected_list, task_type, original_category):
    """
    后处理：
    1. 强制清洗跨领域的错误类别。
    2. 如果原图是“无异常”，强制 Rejected 不能是“无异常”。
    """
    valid_set = set(VALID_CATEGORIES_MAP.get(task_type, []))
    fixed_list = []
    is_original_normal = "无异常" in original_category
    
    for item in rejected_list:
        try:
            comp_assess = item.get("comprehensive_assessment", {})
            raw_type = comp_assess.get("异常类型", "无异常")
            
            # --- 修正逻辑 1: 跨领域清洗 ---
            if raw_type not in valid_set:
                # 倒地任务里出现了垃圾 -> 这种在倒地任务里算无异常
                if task_type == "fall":
                    item["comprehensive_assessment"]["异常类型"] = "无异常"
                # 垃圾任务里出现了倒地 -> 这种在垃圾任务里算无异常
                elif task_type == "garbage":
                    item["comprehensive_assessment"]["异常类型"] = "无异常"
                # 更新 raw_type 以便后续检查
                raw_type = item["comprehensive_assessment"]["异常类型"]

            # --- 修正逻辑 2: 针对“无异常”原图的强制反转 ---
            if is_original_normal and raw_type == "无异常":
                # 如果原图正常，API 却生成了“无异常”（说明API没骗成功），强制修改标签
                if task_type == "fall":
                    item["comprehensive_assessment"]["异常类型"] = "人员异常倒地"
                    # 这里甚至可以注入一段通用的错误推理，防止内容不匹配，但通常DPO对Label更敏感
                elif task_type == "garbage":
                    item["comprehensive_assessment"]["异常类型"] = "垃圾堆积" # 默认选堆积
            
            fixed_list.append(item)
            
        except Exception:
            continue
            
    return fixed_list

def call_api(args, chosen_json_str, original_category, task_type):
    # 获取针对性的 Prompt
    prompt = get_isolated_prompt(original_category, task_type)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
        "x-proxy-secret": args.proxy_secret
    }
    
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"请基于以下正确分析，生成两个对抗样本:\n{chosen_json_str}"}
        ],
        "temperature": 0.85, 
        "max_tokens": 4096
    }

    url = args.api_base_url.rstrip('/') + "/chat/completions"

    for _ in range(args.retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            if response.status_code != 200:
                continue
                
            content = response.json()['choices'][0]['message']['content'].strip()
            
            # 清洗
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            elif "```" in content:
                content = content.replace("```", "")
            
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                rejected_list = json.loads(json_str)
                
                if isinstance(rejected_list, list) and len(rejected_list) >= 1:
                    # 传入 original_category 进行逻辑校验
                    return validate_and_fix_categories(rejected_list, task_type, original_category)
            
        except Exception as e:
            time.sleep(1)
            
    return None

def process_item(args, json_path, image_dir, output_dir):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_name = data.get("file_name")
        image_path = os.path.join(image_dir, file_name)
        
        if not os.path.exists(image_path):
            return False

        analyses = data.get("analyses", [])
        if not analyses: return False
        
        target_obj = analyses[0].get("event_analyses", [])[0]
        original_category = target_obj.get("comprehensive_assessment", {}).get("异常类型")
        
        if not original_category: return False
        
        # API 调用
        rejected_versions = call_api(args, json.dumps(target_obj, ensure_ascii=False), original_category, args.task_type)
        
        if not rejected_versions:
            return False
            
        output_data = {
            "file_name": file_name,
            "original_category": original_category,
            "task_type": args.task_type,
            "image_path": image_path,
            "chosen": target_obj,
            "rejected_versions": rejected_versions
        }
        
        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{base_name}_rejected.json"
        save_path = os.path.join(output_dir, output_filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        return True

    except Exception as e:
        print(f"处理失败 {json_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="DPO Negative Sample Generator (Manual Task Mode)")

    parser.add_argument("--image_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/图片/垃圾堆积", help="图片文件夹路径")
    parser.add_argument("--json_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/垃圾堆积都对_model1", help="JSON文件夹路径")
    parser.add_argument("--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/垃圾堆积都对_model1_reject")
    parser.add_argument("--task_type", type=str, choices=["garbage", "fall"], default="garbage")
    
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument("--api_key", type=str, default="McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ", help="API密钥。")                                                                                                                                                                             
    parser.add_argument("--api_base_url", type=str, default="https://travel-belong-disable-excellence.trycloudflare.com/v1/", help="API的基础URL。")
    parser.add_argument("--proxy_secret", type=str, default="my_secret_password_sensoro_123", help="x-proxy-secret 头的值。")

    parser.add_argument("--workers", type=int, default=4, help="并发处理的线程数。")
    parser.add_argument("--retries", type=int, default=3, help="API调用失败后的最大重试次数。")    
   
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_files = [os.path.join(args.json_dir, f) for f in os.listdir(args.json_dir) if f.endswith('.json')]
    
    print(f"模式: 强行幻觉生成 | 当前任务: {args.task_type}")
    print(f"扫描到 {len(json_files)} 个 JSON 文件，开始处理...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_item, args, jf, args.image_dir, args.output_dir): jf 
            for jf in json_files
        }
        
        for future in tqdm(as_completed(futures), total=len(json_files)):
            if future.result():
                success_count += 1

    print(f"\n处理完成！成功生成: {success_count} / {len(json_files)}")

if __name__ == "__main__":
    main()