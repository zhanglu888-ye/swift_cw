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
import shutil
from collections import defaultdict

# --- 配置区域 ---
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 1. 事件核心配置
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
        "categories": {"流动摊贩", "出店经营"}
    },
    "人群聚集事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注/cap_人群聚集事件.md"),
        "categories": {"人员异常聚集"}
    },
    "雨水井倾倒污水事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注/cap_雨水井倾倒污水事件.md"),
        "categories": {"倾倒污水"}
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

# 2. 新图片目录名 -> 标准事件名 (EVENT_CONFIG的键) 的映射
IMG_CATEGORY_DIR_TO_CONFIG_KEY: Dict[str, str] = {
    "人员倒地": "倒地事件",
    "人员聚集": "人群聚集事件",
    "倾倒污水": "雨水井倾倒污水事件",
    "出店经营": "占道经营事件",
    "流动摊贩": "占道经营事件",
    "垃圾堆积": "垃圾事件",
    "垃圾桶满溢": "垃圾事件",
    "非机动车乱停放": "非机动车乱停放事件",
    "森林火灾":"森林火灾事件",
}

# 3. 标准事件名 -> 旧标签目录名 的映射
CONFIG_KEY_TO_SOURCE_LABEL_DIR: Dict[str, str] = {
    "倒地事件": "人员倒地",
    "人群聚集事件": "人员聚集",
    "雨水井倾倒污水事件": "倾倒污水",
    "占道经营事件": "占道经营事件",
    "垃圾事件": "垃圾事件",
    "非机动车乱停放事件":"非机动车乱停放事件",
    "森林火灾事件":"森林火灾事件",
}

# 4. 图片目录名 -> 标准类别名 (用于Prompt) 的映射
DIR_NAME_TO_CATEGORY: Dict[str, str] = {
    "人员倒地": "人员异常倒地",
    "人员聚集": "人员异常聚集",
    "倾倒污水": "倾倒污水",
    "出店经营": "出店经营",
    "流动摊贩": "流动摊贩",
    "垃圾堆积": "垃圾堆积",
    "垃圾桶满溢": "垃圾桶满溢",
    "非机动车乱停放":"非机动车乱停放",
    "森林火灾":"森林火灾",
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
            logging.info(f"成功加载Prompt for '{event_type}'")
        except FileNotFoundError:
            logging.warning(f"警告: Prompt文件 '{details['prompt_path']}' 未找到！")
    return templates

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
            logging.warning(f"JSON解析失败: {json_str[:200]}...")
            return None
    return None

def get_info_from_path(image_path: Path, image_root_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        relative_path = image_path.relative_to(image_root_dir)
        category_dir_name = relative_path.parts[0]

        config_key = IMG_CATEGORY_DIR_TO_CONFIG_KEY.get(category_dir_name)
        category = DIR_NAME_TO_CATEGORY.get(category_dir_name)

        if not config_key or not category:
            logging.warning(f"路径 {image_path} 中的目录 '{category_dir_name}' 未在配置中找到映射。")
            return None, None
        return config_key, category
    except (ValueError, IndexError):
        logging.warning(f"无法从路径 {image_path} 解析事件信息。")
        return None, None

def is_valid_json_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        logging.warning(f"文件 {path} 存在但JSON格式无效或编码错误，将重新生成。")
        return False

def call_vision_api(client: OpenAI, model: str, prompt: str, base64_image: str) -> Any:
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
        return extract_json_from_response(raw_content)
    except Exception as e:
        logging.warning(f"API调用失败: {e}")
        return None

# --- 核心处理逻辑 ---
def process_single_image(
    image_path: Path,
    output_json_path: Path,
    config_key: str,
    category: str,
    prompt_templates: Dict[str, str],
    client: OpenAI,
    model: str,
    max_retries: int = 3
):
    prompt_template = prompt_templates.get(config_key)
    if not prompt_template:
        logging.error(f"跳过 {image_path.name}: 事件类型 '{config_key}' 没有加载到 Prompt 模板。")
        return

    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        logging.error(f"编码图片 {image_path} 失败: {e}")
        return

    analysis_result = None
    for attempt in range(max_retries):
        logging.info(f"处理 {image_path.name} -> '{category}': 调用API (第 {attempt + 1}/{max_retries} 次)...")

        category_list_str = json.dumps([category], ensure_ascii=False)
        final_prompt = prompt_template.replace("__待检测类别占位符__", category_list_str)

        analysis_result = call_vision_api(client, model, final_prompt, base64_image)

        if analysis_result:
            break
        else:
            logging.warning(f"从模型响应中提取JSON失败，准备重试...")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logging.error(f"处理 {image_path.name} 失败，已达最大重试次数。")

    if analysis_result:
        analysis_result["source_event_type"] = config_key
        analysis_result["source_category"] = category
        analysis_result["source_image_path"] = str(image_path)

        final_output = {
            "file_name": image_path.name,
            "analyses": [analysis_result]
        }
        
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        logging.info(f"成功生成并保存标注到 {output_json_path}")

# --- 主函数 ---

def main():
    parser = argparse.ArgumentParser(description="增量式标注图片，强制重新生成负样本和重复样本。")
    # parser.add_argument("--image_root_dir", default="/host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/清洗标注结果/p0_第三批/images", help="图片根目录。")
    # parser.add_argument("--source_label_dir", default="/host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/生成_results_v3_0123_p0_qwen3vl235B", help="存放旧JSON标签的目录。")
    # parser.add_argument("--output_label_dir", default="/host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/清洗标注结果/p0_第三批/labels", help="保存最终JSON标签的根目录。")
    
    parser.add_argument("--image_root_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/待标注", help="包含所有图片的文件夹路径。") 
    parser.add_argument("--source_label_dir", default="", help="存放旧JSON标签的目录。")
    parser.add_argument("--output_label_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/待标注_qwen3vl235B_tmp", help="保存JSON分析结果的根目录。")   

    
    # --- API及模型参数 ---
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-A22B-Instruct", help="模型名称。")
    parser.add_argument("--api_key", type=str, default="McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ", help="API密钥。")                                                                                                                                                                             
    parser.add_argument("--api_base_url", type=str, default="https://travel-belong-disable-excellence.trycloudflare.com/v1/", help="API的基础URL。")
    parser.add_argument("--proxy_secret", type=str, default="my_secret_password_sensoro_123", help="x-proxy-secret 头的值。")
    parser.add_argument("--workers", type=int, default=4, help="并发处理的线程数。")
    parser.add_argument("--retries", type=int, default=3, help="API调用失败后的最大重试次数。")
    args = parser.parse_args()

    # --- 初始化 ---
    prompt_templates = load_all_prompts(EVENT_CONFIG)
    headers = {"x-proxy-secret": args.proxy_secret} if args.proxy_secret else {}
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base_url,
        default_headers=headers,
        timeout=60.0,
        max_retries=2
    )
    image_root_dir = Path(args.image_root_dir)
    source_label_dir = Path(args.source_label_dir)
    output_label_dir = Path(args.output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    if not image_root_dir.exists() or not source_label_dir.exists():
        logging.critical(f"错误: 图片目录 '{image_root_dir}' 或源标签目录 '{source_label_dir}' 不存在。")
        return

    # --- 1. 扫描所有图片并识别重复项 ---
    logging.info(f"正在扫描图片目录: {image_root_dir} ...")
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = [p for p in image_root_dir.rglob('*') if p.suffix.lower() in valid_extensions]
    
    image_locations = defaultdict(list)
    for p in all_images:
        image_locations[p.name].append(p)

    duplicate_image_names = {name for name, paths in image_locations.items() if len(paths) > 1}
    if duplicate_image_names:
        logging.info(f"发现 {len(duplicate_image_names)} 个重复的图片文件名，它们将被强制重新生成。")

    # --- 2. 确定每张图片的处理策略 ---
    tasks_to_generate = []
    copied_count = 0
    skipped_count = 0
    forced_regeneration_count = 0

    logging.info("正在检查已有标签并应用规则，确定需要重新生成的任务...")
    for image_path in tqdm(all_images, desc="预处理图片"):
        relative_path = image_path.relative_to(image_root_dir)
        output_json_path = (output_label_dir / relative_path).with_suffix('.json')
        
        # 规则0: 如果目标文件已存在，则跳过 (支持断点续传)
        if output_json_path.exists():
            skipped_count += 1
            continue

        config_key, category = get_info_from_path(image_path, image_root_dir)
        if not config_key or not category:
            continue
            
        # 规则1: 是否为负样本?
        is_negative = "负样本" in image_path.parts
        # 规则2: 是否为重复图片?
        is_duplicate = image_path.name in duplicate_image_names

        # 如果满足任一强制生成规则
        if is_negative or is_duplicate:
            reason = "负样本" if is_negative else "重复图片"
            # logging.info(f"任务加入生成队列: {image_path.name} (原因: {reason})")
            tasks_to_generate.append((image_path, output_json_path, config_key, category))
            forced_regeneration_count += 1
            continue

        # 规则3: 对于唯一的正样本，执行原有逻辑
        source_label_dir_name = CONFIG_KEY_TO_SOURCE_LABEL_DIR.get(config_key)
        if not source_label_dir_name:
            tasks_to_generate.append((image_path, output_json_path, config_key, category))
            continue
        
        source_json_path = source_label_dir / source_label_dir_name / f"{image_path.stem}.json"

        if is_valid_json_file(source_json_path):
            try:
                output_json_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_json_path, output_json_path)
                copied_count += 1
            except Exception as e:
                logging.error(f"复制文件 {source_json_path} 到 {output_json_path} 失败: {e}，将尝试重新生成。")
                tasks_to_generate.append((image_path, output_json_path, config_key, category))
        else:
            tasks_to_generate.append((image_path, output_json_path, config_key, category))

    # --- 3. 打印任务统计信息 ---
    logging.info("="*50)
    logging.info("预处理完成！")
    logging.info(f"总图片数: {len(all_images)}")
    logging.info(f"已跳过 (目标JSON已存在): {skipped_count}")
    logging.info(f"成功复制旧标签 (唯一正样本): {copied_count}")
    logging.info(f"需要生成新标签: {len(tasks_to_generate)}")
    logging.info(f"  - 其中强制生成 (负样本/重复): {forced_regeneration_count}")
    logging.info(f"  - 其中常规生成 (唯一正样本无有效旧标签): {len(tasks_to_generate) - forced_regeneration_count}")
    logging.info("="*50)

    if not tasks_to_generate:
        logging.info("所有任务已完成！")
        return

    # --- 4. 并发执行生成任务 ---
    logging.info(f"开始使用 {args.workers} 个线程并发生成新标签...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_single_image,
                img_path, output_path, cfg_key, cat,
                prompt_templates, client, args.model, args.retries
            )
            for img_path, output_path, cfg_key, cat in tasks_to_generate
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="生成标注中"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"一个标注任务执行时发生严重错误: {e}", exc_info=True)

    logging.info("所有生成任务处理完成！")


if __name__ == "__main__":
    main()
