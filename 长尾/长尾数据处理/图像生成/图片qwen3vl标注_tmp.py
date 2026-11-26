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

# --- 修改点 1: 定义固定的事件类型和类别 ---
# 所有输入的图片都将被视为这个类别，不再从文件名解析
FIXED_EVENT_TYPE = "森林火灾事件"
FIXED_CATEGORY = "森林火灾"


# 事件配置 (保持不变，脚本会从中查找 FIXED_EVENT_TYPE 的配置)
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
    "森林火灾事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注/cap_森林火灾.md"),
        "categories": {"森林火灾"}
    },    
}

# --- 修改点 2: 移除不再需要的配置和函数 ---
# TAG_ALIAS 和 extract_info_from_filename 已被移除，因为不再需要从文件名解析类别。
# build_category_map 函数也不再需要。

# --- 辅助函数 (部分保留) ---
def encode_image_to_base64(image_path: Path) -> str:
    """将图片文件编码为Base64字符串"""
    with open(image_path, "rb") as image_file: 
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_all_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    """加载所有在EVENT_CONFIG中定义的Prompt文件"""
    templates = {}
    for event_type, details in config.items():
        try:
            templates[event_type] = details["prompt_path"].read_text(encoding='utf-8')
            logging.info(f"成功加载Prompt for '{event_type}' from '{details['prompt_path']}'")
        except FileNotFoundError:
            logging.warning(f"警告: Prompt文件 '{details['prompt_path']}' 未找到！相关事件将无法处理。")
    return templates

def extract_json_from_response(text: str) -> Optional[Any]:
    """从模型的文本响应中提取JSON对象"""
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
            logging.warning(f"JSON解析失败: {json_str[:200]}...")
            return None
    return None

def call_vision_api(client: OpenAI, model: str, prompt: str, base64_image: str) -> Any:
    """调用视觉大模型API"""
    messages_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
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

# --- 核心处理逻辑 (已修改) ---
def process_sample(image_path: Path, output_dir: Path, prompt_templates: Dict[str, str], client: OpenAI, model: str, max_retries: int = 3):
    """
    处理单个图片：使用固定的类别和Prompt调用API，并保存结果。
    """
    file_name = image_path.name

    # --- 修改点 3: 直接使用固定的事件类型和类别 ---
    event_type = FIXED_EVENT_TYPE
    category = FIXED_CATEGORY

    # 1. 检查是否有对应的 Prompt
    prompt_template = prompt_templates.get(event_type)
    if not prompt_template:
        logging.error(f"跳过 {file_name}: 未能为事件类型 '{event_type}' 加载到 Prompt 模板。请检查 EVENT_CONFIG 配置。")
        return

    # 2. 准备图像
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        logging.error(f"编码图片 {image_path} 失败: {e}")
        return

    # 3. 调用 API (带重试逻辑)
    analysis_result = None
    for attempt in range(max_retries):
        logging.info(f"处理 {file_name}: 尝试调用API (第 {attempt + 1}/{max_retries} 次)...")
        
        # 准备最终的Prompt
        category_list_str = json.dumps([category], ensure_ascii=False)
        final_prompt = prompt_template.replace("__待检测类别占位符__", category_list_str)
        
        analysis_result = call_vision_api(client, model, final_prompt, base64_image)
        
        if analysis_result:
            break
        else:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logging.error(f"处理 {file_name} 失败，已达最大重试次数。")

    # 4. 保存结果
    if analysis_result:
        # 在结果中记录源信息
        analysis_result["source_event_type"] = event_type
        analysis_result["source_category"] = category
        
        final_output = { 
            "file_name": file_name, 
            "analyses": [analysis_result] 
        }
        
        # 根据固定的事件类型创建子目录
        output_subdir = output_dir / event_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        image_stem = image_path.stem
        output_json_path = output_subdir / f"{image_stem}.json"
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        logging.info(f"成功处理并保存结果到 {output_json_path}")


# --- 主函数 (已修改) ---
def main():
    parser = argparse.ArgumentParser(description="遍历图片文件夹，对所有图片按'非机动车乱停放'类别进行分析。")
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-A22B-Instruct", help="模型名称。")
    parser.add_argument("--image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p0_第三批_cleaned/images_moved/森林火灾/正样本", help="包含所有图片的文件夹路径。") 
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p0_第三批_cleaned/labels_moved/森林火灾_new", help="保存JSON分析结果的根目录。")       
    parser.add_argument("--api_key", type=str, default="McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ", help="API密钥。")                                                                                                                                                                             
    parser.add_argument("--api_base_url", type=str, default="https://arms-referring-matthew-appendix.trycloudflare.com/v1/", help="API的基础URL。")
    parser.add_argument("--proxy_secret", type=str, default="my_secret_password_sensoro_123", help="x-proxy-secret 头的值。")
    parser.add_argument("--workers", type=int, default=4, help="并发处理的线程数。")
    parser.add_argument("--retries", type=int, default=3, help="API调用失败后的最大重试次数。")

    args = parser.parse_args()
    if not args.api_key: 
        logging.critical("致命错误: 未提供API密钥。")
        return

    # 加载所有Prompts
    prompt_templates = load_all_prompts(EVENT_CONFIG)
    
    # 初始化API客户端
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
    all_images = [p for p in image_dir.rglob('*') if p.suffix.lower() in valid_extensions]
    
    if not all_images:
        logging.warning(f"在目录 '{image_dir}' 中未找到任何图片文件。")
        return

    # --- 修改点 4: 简化任务过滤逻辑 ---
    # 只检查输出文件是否存在，不再根据文件名判断类别
    tasks_to_run = []
    logging.info("正在检查已完成的任务...")
    for img_path in tqdm(all_images, desc="预处理"):
        image_stem = img_path.stem
        # 输出路径根据固定的事件类型来确定
        output_json_path = output_dir / FIXED_EVENT_TYPE / f"{image_stem}.json"
        
        if not output_json_path.exists():
            tasks_to_run.append(img_path)

    total_images = len(all_images)
    ready_tasks = len(tasks_to_run)
    logging.info(f"扫描到 {total_images} 张图片。")
    logging.info(f"需处理 {ready_tasks} 张 (已排除 {total_images - ready_tasks} 个已完成的任务)。")

    if not tasks_to_run:
        logging.info("所有任务已完成！")
        return

    # 3. 并发执行
    logging.info(f"开始使用 {args.workers} 个线程进行并发处理...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 更新传递给 process_sample 的参数，不再需要 category_map
        futures = [
            executor.submit(
                process_sample, 
                img_path, 
                output_dir, 
                prompt_templates, 
                client, 
                args.model,
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