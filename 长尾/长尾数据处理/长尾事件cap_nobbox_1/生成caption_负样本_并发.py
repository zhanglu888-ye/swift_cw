import os
import json
import re
import base64
import argparse
import logging
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any, Optional
import concurrent.futures
from tqdm import tqdm
import time
import random

# --- 配置区域 ---
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 事件配置
# 注意：现在脚本会用这里的 "categories" 来匹配你的文件夹名称
EVENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "垃圾事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_垃圾事件.md"),
        "categories": {"垃圾桶满溢", "垃圾堆积"}
    },
    "倒地事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_倒地事件.md"),
        "categories": {"人员异常倒地", "人员倒地"} # 兼容 "人员倒地" 文件夹名
    },
    "占道经营事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_占道经营事件.md"),
        "categories": {"流动摊贩", "占道经营"}
    },
    "人群聚集事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_人群聚集事件.md"),
        "categories": {"人员异常聚集", "人群聚集"} # 兼容 "人群聚集" 文件夹名
    },
    "雨水井倾倒污水事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_雨水井倾倒污水事件.md"),
        "categories": {"污水井倾倒"}
    },
    "非机动车乱停放事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_非机动车乱停放事件.md"),
        "categories": {"非机动车乱停放", "非机动车"} # 兼容 "非机动车" 文件夹名
    },
}

# --- 辅助函数 (基本不变) ---
def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file: 
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_all_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    templates = {}
    for event_type, details in config.items():
        try:
            templates[event_type] = details["prompt_path"].read_text(encoding='utf-8')
            logging.info(f"成功加载Prompt for '{event_type}' from '{details['prompt_path']}'")
        except FileNotFoundError:
            logging.error(f"致命错误: Prompt文件 '{details['prompt_path']}' 未找到！")
            raise
    return templates

def build_category_map(config: Dict[str, Any]) -> Dict[str, str]:
    """ 从配置构建一个从 类别名 -> 事件名 的映射 """
    category_map = {}
    for event_type, details in config.items():
        for category in details["categories"]: 
            category_map[category] = event_type
    return category_map

def extract_json_from_response(text: str) -> Optional[Any]:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = None
    if match:
        json_str = match.group(1)
    else:
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
    if json_str:
        try: 
            return json.loads(json_str)
        except json.JSONDecodeError: 
            return None
    return None

def call_vision_api(client: OpenAI, model: str, prompt: str, base64_large_image: str) -> Any:
    messages_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_large_image}"}}
    ]
    raw_content = ""
    try:
        response = client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": messages_content}],
            temperature=0.2, 
            max_tokens=4096,
        )
        raw_content = response.choices[0].message.content
        parsed_json = extract_json_from_response(raw_content)
        if parsed_json:
            return parsed_json
        else:
            logging.warning(f"无法从模型响应中提取有效的JSON，将尝试重试。原始输出: {raw_content[:200]}...")
            return None
    except Exception as e:
        logging.warning(f"API调用失败，将尝试重试。错误: {e}")
        return None

# --- 核心处理逻辑 (重构) ---
def process_sample(image_path: Path, event_type: str, output_dir: Path, prompt_templates: Dict[str, str], client: OpenAI, model: str, max_retries: int = 3):
    try:
        base64_large_image = encode_image_to_base64(image_path)
    except Exception as e:
        logging.error(f"编码图片 {image_path} 失败: {e}")
        return

    analysis_result = None
    # 获取此事件类型对应的所有具体类别
    category_list = list(EVENT_CONFIG[event_type]["categories"])
        
    for attempt in range(max_retries):
        logging.info(f"样本 {image_path.name} -> '{event_type}': 尝试调用API (第 {attempt + 1}/{max_retries} 次)...")
        
        prompt_template = prompt_templates[event_type]
        # 将此事件下的所有类别作为占位符内容
        category_list_str = json.dumps(category_list, ensure_ascii=False)
        final_prompt = prompt_template.replace("__待检测类别占位符__", category_list_str)
        
        analysis_result = call_vision_api(client, model, final_prompt, base64_large_image)
        
        if analysis_result:
            logging.info(f"样本 {image_path.name} -> '{event_type}': 第 {attempt + 1} 次尝试成功。")
            break
        else:
            if attempt < max_retries - 1:
                logging.warning(f"样本 {image_path.name} -> '{event_type}': 第 {attempt + 1} 次尝试失败，将在2秒后重试...")
                time.sleep(2)
            else:
                logging.error(f"样本 {image_path.name} -> '{event_type}': 所有 {max_retries} 次尝试均失败。")
    
    if not analysis_result:
        logging.error(f"样本 {image_path.name} 的 '{event_type}' API调用最终失败，不保存结果。")
        return

    analysis_result["source_event_type"] = event_type
    analysis_result["source_categories"] = category_list
    
    # 最终输出的JSON包含一个分析列表（虽然这里只有一个元素，但保持结构一致性）
    final_output = {
        "image_path": str(image_path.resolve()),
        "analyses": [analysis_result]
    }
    
    output_subdir = output_dir / event_type
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    output_json_path = output_subdir / f"{image_path.stem}.json"
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    logging.info(f"成功保存结果到 {output_json_path}")

# --- 主函数 (重构) ---
def main():
    parser = argparse.ArgumentParser(description="通过扫描目录结构生成图片描述JSON，无需标签文件。")
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-A22B-Instruct", help="模型名称。")
    parser.add_argument("--image_dir", default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/负样本", help="图片根目录。其下一级子目录名应为类别名 (如'垃圾堆积')。") 
    parser.add_argument("--output_dir", default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/负样本_分类and描述", help="保存JSON分析结果的根目录。")   
    parser.add_argument("--api_key", type=str, default="McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ", help="API密钥。")                                                                                                                                                                             
    parser.add_argument("--api_base_url", type=str, default="http://10.21.65.20:8080/v1/", help="API的基础URL。")
    parser.add_argument("--workers", type=int, default=4, help="并发处理的线程数。")
    parser.add_argument("--retries", type=int, default=3, help="API调用失败后的最大重试次数。")
    parser.add_argument("--max_per_category", type=int, default=500, help="每个事件类别最多处理的图片数量。默认500张，设为0或负数表示不限制。")

    args = parser.parse_args()
    
    # --- 准备工作 ---
    prompt_templates = load_all_prompts(EVENT_CONFIG)
    category_map = build_category_map(EVENT_CONFIG)
    client = OpenAI(
        api_key=args.api_key, 
        base_url=args.api_base_url,
        timeout=60.0,
        max_retries=2 
    )
    
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        logging.critical(f"错误: 图片输入目录 '{image_dir}' 不存在或不是一个目录。")
        return

    # --- 扫描目录，发现并筛选任务 ---
    all_potential_tasks = []
    category_counts = {event_type: 0 for event_type in EVENT_CONFIG.keys()}

    logging.info(f"正在从 '{image_dir}' 扫描类别文件夹...")
    for category_dir in image_dir.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        event_type = category_map.get(category_name)
        
        if not event_type:
            logging.warning(f"跳过文件夹 '{category_name}'，因为它未在 EVENT_CONFIG 中被映射到任何事件。")
            continue
        
        logging.info(f"发现类别 '{category_name}' -> 映射到事件 '{event_type}'。正在查找图片...")
        
        # 递归查找所有支持的图片格式
        image_paths = [p for p in category_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not image_paths:
            logging.warning(f"在 '{category_dir}' 中未找到图片。")
            continue
        
        logging.info(f"在 '{category_name}' 中找到 {len(image_paths)} 张图片。")
        
        for img_path in image_paths:
            all_potential_tasks.append((img_path, event_type))

    # --- 应用类别数量限制 ---
    if args.max_per_category is not None and args.max_per_category > 0:
        logging.info(f"将对每个事件类别应用数量上限: {args.max_per_category}")
        tasks_by_event = {event: [] for event in EVENT_CONFIG.keys()}
        for img_path, event_type in all_potential_tasks:
            tasks_by_event[event_type].append((img_path, event_type))
        
        limited_tasks = []
        for event_type, tasks in tasks_by_event.items():
            if len(tasks) > args.max_per_category:
                logging.info(f"事件 '{event_type}' 有 {len(tasks)} 张图片, 将随机抽样 {args.max_per_category} 张。")
                limited_tasks.extend(random.sample(tasks, args.max_per_category))
            else:
                limited_tasks.extend(tasks)
        all_potential_tasks = limited_tasks
    
    # --- 断点续传：过滤已完成的任务 ---
    tasks_to_run = []
    logging.info("正在检查已完成的任务以实现断-点续传...")
    for img_path, event_type in tqdm(all_potential_tasks, desc="检查进度"):
        output_json_path = output_dir / event_type / f"{img_path.stem}.json"
        if not output_json_path.exists():
            tasks_to_run.append((img_path, event_type))
            category_counts[event_type] += 1

    logging.info(f"扫描和筛选完成。总共发现 {len(all_potential_tasks)} 个符合条件的样本。")
    logging.info(f"其中 {len(all_potential_tasks) - len(tasks_to_run)} 个已处理，剩余 {len(tasks_to_run)} 个任务待执行。")
    logging.info("各事件类别待处理任务数量:")
    for event, count in sorted(category_counts.items()):
        if count > 0:
            logging.info(f"  - {event}: {count}")

    if not tasks_to_run:
        logging.info("所有任务已完成！")
        return

    # --- 执行并发处理 ---
    logging.info(f"开始使用 {args.workers} 个线程进行并发处理...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_sample, 
                img_path, event_type, output_dir, 
                prompt_templates, client, args.model,
                max_retries=args.retries
            ) 
            for img_path, event_type in tasks_to_run
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="并发处理中"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"一个处理任务在执行中失败: {e}", exc_info=True)

    logging.info("所有剩余任务处理完成！")

if __name__ == "__main__":
    main()
