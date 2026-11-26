import os
import json
import re
import base64
import argparse
import logging
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
from tqdm import tqdm
import time

# --- 配置区域 ---
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 事件配置 (保持不变)
EVENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "垃圾事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注/cap_垃圾事件.md"),
        "categories": {"垃圾桶满溢", "垃圾堆积"}
    },
    "倒地事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注/cap_倒地事件.md"),
        "categories": {"人员异常倒地"}
    },
    "占道经营事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注/cap_占道经营事件.md"),
        "categories": {"流动摊贩", "占道经营"}
    },
    "人群聚集事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注/cap_人群聚集事件.md"),
        "categories": {"人员异常聚集"}
    },
    "雨水井倾倒污水事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注/cap_雨水井倾倒污水事件.md"),
        "categories": {"污水井倾倒"}
    },
    "非机动车乱停放事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注/cap_非机动车乱停放事件.md"),
        "categories": {"非机动车乱停放"}
    },
}



# --- 文件名标签到Config类别的映射别名 ---
# key: 文件名中出现的字符串 (中文括号内容 或 英文关键词)
# value: EVENT_CONFIG["categories"] 里的标准名称
TAG_ALIAS = {
    # --- 中文标签 (括号内内容) ---
    "出店经营": "占道经营",          
    "流动摊贩": "流动摊贩",
    "人员倒地": "人员异常倒地",
    "人员聚集": "人员异常聚集",
    "倾倒污水": "污水井倾倒",
    "垃圾桶满溢": "垃圾桶满溢",
    "垃圾堆积": "垃圾堆积",
    "垃圾堆积 - 无桶": "垃圾堆积",
    "非机动车乱停放": "非机动车乱停放",

    # --- 英文标签 (文件名包含内容) ---
    "Trash_Overflow": "垃圾桶满溢",
    "Trash_Accumulation": "垃圾堆积", 
    "Mobile_Vendor": "流动摊贩",
    "Store_Outside": "占道经营",    # 对应 Config 中的 占道经营
    "Person_Falling": "人员异常倒地",  
    "Abnormal_Gathering": "人员异常聚集", 
    "Pouring_Sewage": "污水井倾倒" 
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
            logging.warning(f"警告: Prompt文件 '{details['prompt_path']}' 未找到！相关事件将无法处理。")
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

def extract_info_from_filename(file_name: str, category_map: Dict[str, str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    解析文件名，返回 (原始标签, 映射后的类别, 事件大类)
    策略1: 优先匹配括号 () 内的中文标签
    策略2: 其次匹配 TAG_ALIAS 中的英文关键词
    """
    found_tag = None
    
    # 策略 1: 括号正则匹配
    match = re.search(r'\((.*?)\)', file_name)
    if match:
        found_tag = match.group(1).strip()
        # 如果括号里的内容在映射表里，直接采纳
        if found_tag in TAG_ALIAS:
            category = TAG_ALIAS[found_tag]
            event_type = category_map.get(category)
            if event_type:
                return found_tag, category, event_type

    # 策略 2: 英文关键词包含匹配 (如果没有括号，或括号内容无法映射)
    # 遍历映射表的所有Key，看是否包含在文件名中
    for key in TAG_ALIAS.keys():
        if key in file_name:
            # 找到一个匹配的关键词
            found_tag = key
            category = TAG_ALIAS[key]
            event_type = category_map.get(category)
            if event_type:
                return found_tag, category, event_type
    
    # 如果都失败
    return None, None, None


def call_vision_api(client: OpenAI, model: str, prompt: str, base64_large_image: str) -> Any:
    messages_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_large_image}"}}
    ]
    try:
        response = client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": messages_content}],
            temperature=0.1, 
            max_tokens=4096,
        )
        raw_content = response.choices[0].message.content
        parsed_json = extract_json_from_response(raw_content)
        if parsed_json:
            return parsed_json
        else:
            logging.warning(f"无法从模型响应中提取有效的JSON，将尝试重试。原始输出简略: {raw_content[:100]}...")
            return None
    except Exception as e:
        logging.warning(f"API调用失败，将尝试重试。错误: {e}")
        return None

# --- 核心处理逻辑 ---
def process_sample(image_path: Path, output_dir: Path, prompt_templates: Dict[str, str], category_map: Dict[str, str], client: OpenAI, model: str, max_retries: int = 3):
    """
    处理单个图片：从文件名提取标签 -> 匹配Prompt -> 调用API -> 保存结果
    """
    file_name = image_path.name
    
    # 1. 提取信息 (使用统一的辅助函数)
    raw_tag, category, event_type = extract_info_from_filename(file_name, category_map)

    if not event_type:
        logging.warning(f"跳过 {file_name}: 无法匹配到已知类别 (原始标签: {raw_tag if raw_tag else '未识别'})")
        return

    # 2. 检查是否有对应的 Prompt
    if event_type not in prompt_templates:
        logging.warning(f"跳过 {file_name}: 事件类型 '{event_type}' (来自 '{raw_tag}') 没有加载到 Prompt 模板。")
        return

    # 3. 准备图像
    try:
        base64_large_image = encode_image_to_base64(image_path)
    except Exception as e:
        logging.error(f"编码大图 {image_path} 失败: {e}")
        return

    # 4. 调用 API
    analysis_result = None
    for attempt in range(max_retries):
        logging.info(f"处理 {file_name} -> '{category}': 尝试调用API (第 {attempt + 1}/{max_retries} 次)...")
        
        prompt_template = prompt_templates[event_type]
        category_list = [category] 
        category_list_str = json.dumps(category_list, ensure_ascii=False)
        final_prompt = prompt_template.replace("__待检测类别占位符__", category_list_str)
        
        analysis_result = call_vision_api(client, model, final_prompt, base64_large_image)
        
        if analysis_result:
            break
        else:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logging.error(f"处理 {file_name} 失败，已达最大重试次数。")

    # 5. 保存结果
    if analysis_result:
        analysis_result["source_event_type"] = event_type
        analysis_result["source_category"] = category
        analysis_result["source_tag_in_filename"] = raw_tag # 记录一下是从哪个标签识别出来的
        
        final_output = { 
            "file_name": file_name, 
            "analyses": [analysis_result] 
        }
        
        output_subdir = output_dir / event_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        image_stem = image_path.stem
        output_json_path = output_subdir / f"{image_stem}.json"
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="遍历图片文件夹，根据文件名标签并发调用VLM生成分析结果。")
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-A22B-Instruct", help="模型名称。")
    parser.add_argument("--image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/待标注", help="包含所有图片的文件夹路径。") 
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/待标注_qwen3vl235B", help="保存JSON分析结果的根目录。")   


  
    # === 更新后的 API 配置 ===
    parser.add_argument("--api_key", type=str, default="McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ", help="API密钥。")                                                                                                                                                                             
    parser.add_argument("--api_base_url", type=str, default="https://arms-referring-matthew-appendix.trycloudflare.com/v1/", help="API的基础URL。")
    parser.add_argument("--proxy_secret", type=str, default="my_secret_password_sensoro_123", help="x-proxy-secret 头的值。")

    parser.add_argument("--workers", type=int, default=4, help="并发处理的线程数。")
    parser.add_argument("--retries", type=int, default=3, help="API调用失败后的最大重试次数。")

    args = parser.parse_args()
    if not args.api_key: 
        logging.critical("致命错误: 未提供API密钥。")
        return

    # 加载配置
    prompt_templates = load_all_prompts(EVENT_CONFIG)
    category_map = build_category_map(EVENT_CONFIG)
    
    # === 初始化客户端 (包含自定义 Header) ===
    client = OpenAI(
        api_key=args.api_key, 
        base_url=args.api_base_url,
        default_headers={"x-proxy-secret": args.proxy_secret}, 
        timeout=60.0,
        max_retries=2 
    )
    
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        logging.critical(f"错误: 图片目录不存在 {image_dir}")
        return

    # 1. 扫描所有图片
    logging.info(f"正在扫描目录: {image_dir} ...")
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = [
        p for p in image_dir.rglob('*') 
        if p.suffix.lower() in valid_extensions
    ]
    
    if not all_images:
        logging.warning(f"在目录 '{image_dir}' 中未找到任何图片文件。")
        return

    # 2. 过滤已完成的任务 (断点续传) 与检查标签有效性
    tasks_to_run = []
    unrecognized_files = []
    skipped_count = 0
    
    logging.info("正在检查已完成任务和文件名标签...")
    for img_path in tqdm(all_images, desc="预处理"):
        file_name = img_path.name
        
        # 使用新的解析函数
        raw_tag, category, event_type = extract_info_from_filename(file_name, category_map)
        
        if not event_type:
            unrecognized_files.append(file_name)
            skipped_count += 1
            continue

        # 检查是否已存在结果
        image_stem = img_path.stem
        output_json_path = output_dir / event_type / f"{image_stem}.json"
        
        if not output_json_path.exists():
            tasks_to_run.append(img_path)

    # 3. 打印警告信息
    if unrecognized_files:
        logging.warning("="*80)
        logging.warning(f"发现 {len(unrecognized_files)} 个文件的标签无法匹配到配置，将被跳过。部分示例：")
        for f in unrecognized_files[:10]: # 只打印前10个
            logging.warning(f"  - {f}")
        logging.warning("="*80)

    total_images = len(all_images)
    ready_tasks = len(tasks_to_run)
    logging.info(f"扫描到 {total_images} 张图片。")
    logging.info(f"需处理 {ready_tasks} 张 (排除已完成/无法识别)。")

    if not tasks_to_run:
        logging.info("所有任务已完成！")
        return

    # 4. 并发执行
    logging.info(f"开始使用 {args.workers} 个线程进行并发处理...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_sample, 
                img_path, output_dir, 
                prompt_templates, category_map, client, args.model,
                max_retries=args.retries
            ) 
            for img_path in tasks_to_run
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="并发处理中"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"任务执行异常: {e}", exc_info=True)

    logging.info("所有任务处理完成！")

if __name__ == "__main__":
    main()