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

# --- 配置区域 ---
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 事件配置
EVENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "垃圾事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_垃圾事件.md"),
        "categories": {"垃圾桶满溢", "垃圾堆积"}
    },
    "倒地事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_倒地事件.md"),
        "categories": {"人员异常倒地"}
    },
    "占道经营事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_占道经营事件.md"),
        "categories": {"流动摊贩", "占道经营"}
    },
    #"森林火灾事件": {
    #    "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_森林火灾.md"),
    #    "categories": {"森林火灾"}
    #},
    "人群聚集事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_人群聚集事件.md"),
        "categories": {"人员异常聚集"}
    },
    "雨水井倾倒污水事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_雨水井倾倒污水事件.md"),
        "categories": {"污水井倾倒"}
    },
    "非机动车乱停放事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_非机动车乱停放事件.md"),
        "categories": {"非机动车乱停放"}
    },

}

# --- 辅助函数 ---
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

# --- 核心处理逻辑 ---
def process_sample(label_path: Path, image_dir: Path, output_dir: Path, prompt_templates: Dict[str, str], category_map: Dict[str, str], client: OpenAI, model: str, max_retries: int = 3):
    try:
        with open(label_path, 'r', encoding='utf-8') as f: 
            label_data = json.load(f)
        file_name = label_data.get("file_name")
        labels = label_data.get("labels", [])
        if not file_name or not labels:
            logging.warning(f"跳过 {label_path.name}: 'file_name' 或 'labels' 字段缺失/为空。")
            return
        image_path = image_dir / file_name
        if not image_path.exists():
            logging.warning(f"跳过 {file_name}: 在图片目录 '{image_dir}' 中找不到对应的图片文件。")
            return
    except Exception as e:
        logging.error(f"读取或解析标签文件 {label_path} 失败: {e}")
        return

    events_to_process: Dict[str, List[str]] = {}
    for label in labels:
        event_type = category_map.get(label)
        if event_type:
            if event_type not in events_to_process: 
                events_to_process[event_type] = []
            events_to_process[event_type].append(label)
    if not events_to_process:
        return

    try:
        base64_large_image = encode_image_to_base64(image_path)
    except Exception as e:
        logging.error(f"编码大图 {image_path} 失败: {e}")
        return

    all_analyses, processed_event_types = [], []
    for event_type, category_list in events_to_process.items():
        
        analysis_result = None
        for attempt in range(max_retries):
            # 在并发模式下，为日志添加文件名以区分
            logging.info(f"样本 {file_name} -> '{event_type}': 尝试调用API (第 {attempt + 1}/{max_retries} 次)...")
            
            prompt_template = prompt_templates[event_type]
            category_list_str = json.dumps(category_list, ensure_ascii=False)
            final_prompt = prompt_template.replace("__待检测类别占位符__", category_list_str)
            
            analysis_result = call_vision_api(client, model, final_prompt, base64_large_image)
            
            if analysis_result:
                logging.info(f"样本 {file_name} -> '{event_type}': 第 {attempt + 1} 次尝试成功。")
                break
            else:
                if attempt < max_retries - 1:
                    logging.warning(f"样本 {file_name} -> '{event_type}': 第 {attempt + 1} 次尝试失败，将在2秒后重试...")
                    time.sleep(2)
                else:
                    logging.error(f"样本 {file_name} -> '{event_type}': 所有 {max_retries} 次尝试均失败。")
        
        if analysis_result:
            analysis_result["source_event_type"] = event_type
            analysis_result["source_categories"] = category_list
            all_analyses.append(analysis_result)
            processed_event_types.append(event_type)
        else:
            logging.error(f"样本 {file_name} 的 '{event_type}' API调用最终失败。")
            
    if not all_analyses:
        logging.warning(f"样本 {file_name} 的所有事件组都处理失败，不保存任何结果。")
        return

    final_output = { "file_name": file_name, "analyses": all_analyses }
    processed_event_types.sort()
    save_event_type_dir = processed_event_types[0]
    output_subdir = output_dir / save_event_type_dir
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    image_stem = Path(file_name).stem
    output_json_path = output_subdir / f"{image_stem}.json"
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

# --- 主函数 ---
# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="并发处理大图和标签JSON，支持断点续传和API调用重试。")
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-A22B-Instruct", help="模型名称。")
    parser.add_argument("--image_dir", default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/合并_deduplicated/large_images", help="包含所有图片的文件夹路径。") 
    parser.add_argument("--label_dir", default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/合并_deduplicated/label_json", help="包含所有标签json文件的文件夹路径。")            
    parser.add_argument("--output_dir", default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/分类and描述", help="保存JSON分析结果的根目录。")   
    parser.add_argument("--api_key", type=str, default="McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ", help="API密钥。")                                                                                                                                                                             
    parser.add_argument("--api_base_url", type=str, default="http://10.21.65.20:8080/v1/", help="API的基础URL。")
    parser.add_argument("--workers", type=int, default=4, help="并发处理的线程数。")
    parser.add_argument("--retries", type=int, default=3, help="API调用失败后的最大重试次数。")

    args = parser.parse_args()
    if not args.api_key: 
        logging.critical("致命错误: 未提供API密钥。")
        return

    prompt_templates = load_all_prompts(EVENT_CONFIG)
    category_map = build_category_map(EVENT_CONFIG)
    client = OpenAI(
        api_key=args.api_key, 
        base_url=args.api_base_url,
        timeout=60.0,
        max_retries=2 
    )
    
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_label_files = sorted(list(label_dir.glob('*.json')))
    if not all_label_files:
        logging.warning(f"在目录 '{label_dir}' 中未找到任何 .json 文件。")
        return
    
    tasks_to_run = []
    # --- 新增代码: 初始化一个集合来存储未找到的标签 ---
    unmapped_labels_set = set()
    
    logging.info("正在检查已完成的任务，实现断点续传...")
    for label_path in tqdm(all_label_files, desc="检查进度"):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            file_name = label_data.get("file_name")
            labels = label_data.get("labels", [])
            if not file_name or not labels: 
                continue

            event_types_for_sample = sorted(list(set(
                category_map.get(label) for label in labels if category_map.get(label)
            )))

            if not event_types_for_sample:
                # --- 修改点: 如果文件被跳过，记录其未配置的标签 ---
                for label in labels:
                    if label not in category_map:
                        unmapped_labels_set.add(label)
                continue

            save_event_type_dir = event_types_for_sample[0]
            image_stem = Path(file_name).stem
            output_json_path = output_dir / save_event_type_dir / f"{image_stem}.json"

            if not output_json_path.exists():
                tasks_to_run.append(label_path)
        except Exception as e:
            logging.error(f"检查文件 {label_path.name} 失败: {e}")

    # --- 新增代码: 在检查后打印所有未找到的标签 ---
    if unmapped_labels_set:
        logging.warning("="*80)
        logging.warning("发现以下标签类别未在 EVENT_CONFIG 中配置，导致部分文件被跳过：")
        # 排序以获得一致的输出
        for label_name in sorted(list(unmapped_labels_set)):
            logging.warning(f"  - {label_name}")
        logging.warning("请检查您的 EVENT_CONFIG 配置，将这些标签添加到相应的 'categories' 中。")
        logging.warning("="*80)
    
    total_tasks = len(all_label_files)
    remaining_tasks = len(tasks_to_run)
    logging.info(f"总计 {total_tasks} 个样本，已完成 {total_tasks - remaining_tasks} 个(包含跳过)，剩余 {remaining_tasks} 个待处理。")

    if not tasks_to_run:
        logging.info("所有任务已完成！")
        return

    logging.info(f"开始使用 {args.workers} 个线程进行并发处理...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_sample, 
                label_file_path, image_dir, output_dir, 
                prompt_templates, category_map, client, args.model,
                max_retries=args.retries
            ) 
            for label_file_path in tasks_to_run
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="并发处理中"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"一个处理任务在执行中失败: {e}", exc_info=True)

    logging.info("所有剩余任务处理完成！")

if __name__ == "__main__":
    main()
