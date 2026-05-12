# -*- coding: utf-8 -*-
"""
图像事件描述生成脚本（VLM版）

功能：
1. 读取包含图片文件名和事件标签的JSON文件。
2. 根据事件标签，从配置文件中动态选择对应的Prompt。
3. 调用多模态大模型（VLM）API，传入图片和Prompt，生成对事件的详细描述。
4. 支持API调用失败后自动重试。
5. 支持并发处理以提高效率。
6. 实现断点续传：自动跳过已经成功处理并生成结果的图片，可以直接重复运行以处理失败的样本。
7. 新增特性：在启动时检查并报告所有未在配置中定义的标签，帮助用户发现配置问题。
"""

import os
import json
import re
import base64
import argparse
import logging
import time
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any, Optional, Set
import concurrent.futures
from tqdm import tqdm

# --- 1. 配置区域 ---

# 日志配置：设置日志格式，方便调试和追踪
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 事件类型配置：核心配置区域
# 每个事件类型定义了：
# - prompt_path: 用于指导VLM生成描述的Prompt模板文件路径。
# - categories: 属于该事件类型的具体标签集合。
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
    # 示例：如果需要添加新事件，在这里添加即可
    # "森林火灾事件": {
    #    "prompt_path": Path("/path/to/your/prompts/cap_森林火灾.md"),
    #    "categories": {"森林火灾", "山火"}
    # },
}

# --- 2. 辅助函数 ---

def encode_image_to_base64(image_path: Path) -> str:
    """将图片文件编码为Base64字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"无法读取或编码图片 '{image_path}': {e}")
        raise

def load_all_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    """从配置文件中加载所有事件类型的Prompt模板。"""
    templates = {}
    for event_type, details in config.items():
        try:
            templates[event_type] = details["prompt_path"].read_text(encoding='utf-8')
            logging.info(f"成功加载 '{event_type}' 的Prompt: '{details['prompt_path']}'")
        except FileNotFoundError:
            logging.error(f"致命错误: Prompt文件 '{details['prompt_path']}' 未找到！脚本无法继续。")
            raise
    return templates

def build_category_map(config: Dict[str, Any]) -> Dict[str, str]:
    """构建从具体标签到事件类型的映射字典，方便快速查找。"""
    category_map = {}
    for event_type, details in config.items():
        for category in details["categories"]:
            if category in category_map:
                logging.warning(f"配置警告: 标签 '{category}' 在多个事件类型中重复定义！将使用 '{event_type}'。")
            category_map[category] = event_type
    return category_map

def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型返回的文本中提取JSON对象。
    优先匹配被```json ... ```包裹的代码块，如果找不到，则尝试匹配最外层的花括号。
    """
    if not text:
        return None
    # 优先匹配Markdown格式的JSON代码块
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = None
    if match:
        json_str = match.group(1)
    else:
        # 如果没有找到代码块，则尝试寻找第一个'{'和最后一个'}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]

    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.warning(f"JSON解析失败: {e}. 原始字符串片段: '{json_str[:200]}...'")
            return None
    return None

def call_vision_api(client: OpenAI, model: str, prompt: str, base64_image: str) -> Optional[Dict[str, Any]]:
    """
    调用VLM API并返回解析后的JSON结果。
    如果API调用失败或返回内容无法解析为JSON，则返回None。
    """
    try:
        messages_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": messages_content}],
            temperature=0.2,
            max_tokens=4096,
        )
        raw_content = response.choices[0].message.content
        return extract_json_from_response(raw_content)
    except Exception as e:
        # 捕获所有可能的API异常，如超时、连接错误等
        logging.warning(f"API调用时发生异常: {e}")
        return None

# --- 3. 核心处理逻辑 ---

def process_sample(label_path: Path, image_dir: Path, output_dir: Path,
                   prompt_templates: Dict[str, str], category_map: Dict[str, str],
                   client: OpenAI, model: str, max_retries: int = 3):
    """
    处理单个样本的完整流程。
    包括读取标签、查找图片、调用API、处理结果和保存文件。
    """
    try:
        # 1. 读取并验证标签文件
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

    # 2. 根据标签确定需要处理的事件类型
    events_to_process: Dict[str, List[str]] = {}
    for label in labels:
        event_type = category_map.get(label)
        if event_type:
            if event_type not in events_to_process:
                events_to_process[event_type] = []
            events_to_process[event_type].append(label)

    if not events_to_process:
        # 这种情况通常在主函数的预检查阶段就被过滤了，但作为安全措施保留
        logging.warning(f"跳过 {file_name}: 标签 {labels} 均未在EVENT_CONFIG中配置。")
        return

    # 3. 编码图片
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception:
        # 错误已在 encode_image_to_base64 中记录
        return

    # 4. 对每个事件类型调用API并收集结果
    all_analyses = []
    for event_type, category_list in events_to_process.items():
        analysis_result = None
        for attempt in range(max_retries):
            # 准备最终的Prompt
            prompt_template = prompt_templates[event_type]
            category_list_str = json.dumps(category_list, ensure_ascii=False)
            final_prompt = prompt_template.replace("__待检测类别占位符__", category_list_str)

            # 调用API
            log_prefix = f"样本 {file_name} -> '{event_type}' (尝试 {attempt + 1}/{max_retries})"
            logging.info(f"{log_prefix}: 开始调用API...")
            analysis_result = call_vision_api(client, model, final_prompt, base64_image)

            if analysis_result:
                logging.info(f"{log_prefix}: 成功获取并解析了JSON结果。")
                break
            else:
                if attempt < max_retries - 1:
                    logging.warning(f"{log_prefix}: 未能获取有效JSON，将在2秒后重试...")
                    time.sleep(2)
                else:
                    logging.error(f"{log_prefix}: 所有 {max_retries} 次尝试均告失败。")

        if analysis_result:
            # 为结果添加源信息，方便追溯
            analysis_result["source_event_type"] = event_type
            analysis_result["source_categories"] = category_list
            all_analyses.append(analysis_result)
        else:
            logging.error(f"样本 {file_name} 的 '{event_type}' 事件处理最终失败。")

    # 5. 保存结果
    if not all_analyses:
        logging.warning(f"样本 {file_name} 的所有事件组都处理失败，不保存任何结果。")
        return

    # 为了保持输出目录的一致性和可预测性，我们根据事件类型排序，并使用第一个作为子目录名
    processed_event_types = sorted([res["source_event_type"] for res in all_analyses])
    save_event_type_dir = processed_event_types[0]
    output_subdir = output_dir / save_event_type_dir
    output_subdir.mkdir(parents=True, exist_ok=True)

    image_stem = Path(file_name).stem
    output_json_path = output_subdir / f"{image_stem}.json"

    final_output = {"file_name": file_name, "analyses": all_analyses}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    logging.info(f"成功保存结果到: {output_json_path}")

# --- 4. 主函数 ---

def main():
    """脚本主入口，负责参数解析、任务调度和并发执行。"""
    parser = argparse.ArgumentParser(
        description="并发处理图片与标签，调用VLM生成事件描述，支持断点续传和API重试。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-A22B-Instruct", help="调用的模型名称。")
    parser.add_argument("--image_dir", type=Path, default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/合并_deduplicated/large_images", help="包含所有图片的文件夹路径。")
    parser.add_argument("--label_dir", type=Path, default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/合并_deduplicated/label_json", help="包含所有标签JSON文件的文件夹路径。")
    parser.add_argument("--output_dir", type=Path, default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/分类and描述", help="保存JSON分析结果的根目录。")
    parser.add_argument("--api_key", type=str, default="McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ", help="API密钥。")
    parser.add_argument("--api_base_url", type=str, default="http://10.21.65.20:8080/v1/", help="API的基础URL。")
    parser.add_argument("--workers", type=int, default=2, help="并发处理的线程数。")
    parser.add_argument("--retries", type=int, default=3, help="API调用失败后的最大重试次数。")
    args = parser.parse_args()

    # --- 1. 初始化 ---
    if not args.api_key:
        logging.critical("致命错误: 未通过 --api_key 参数提供API密钥。")
        return

    try:
        prompt_templates = load_all_prompts(EVENT_CONFIG)
    except FileNotFoundError:
        return # 如果Prompt文件找不到，直接退出

    category_map = build_category_map(EVENT_CONFIG)
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base_url,
        timeout=60.0, # 设置较长的超时时间
        max_retries=0 # 在我们自己的代码中处理重试逻辑
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_label_files = sorted(list(args.label_dir.glob('*.json')))
    if not all_label_files:
        logging.warning(f"在目录 '{args.label_dir}' 中未找到任何 .json 文件。")
        return

    # --- 2. 发现需要处理的任务 (实现断点续传和配置检查) ---
    tasks_to_run: List[Path] = []
    unmapped_labels: Set[str] = set()

    logging.info("正在扫描标签目录，检查任务状态和配置...")
    for label_path in tqdm(all_label_files, desc="扫描与检查"):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)

            file_name = label_data.get("file_name")
            labels = label_data.get("labels", [])
            if not file_name or not labels:
                continue

            # 检查此文件的标签是否都已配置
            has_mapped_event = False
            for label in labels:
                if label in category_map:
                    has_mapped_event = True
                else:
                    unmapped_labels.add(label)

            if not has_mapped_event:
                # 如果这个文件的所有标签都没有配置，则跳过
                continue

            # 计算预期的输出路径
            event_types_for_sample = sorted(list({
                category_map[label] for label in labels if label in category_map
            }))
            save_event_type_dir = event_types_for_sample[0]
            image_stem = Path(file_name).stem
            output_json_path = args.output_dir / save_event_type_dir / f"{image_stem}.json"

            # 如果输出文件不存在，则加入待办列表
            if not output_json_path.exists():
                tasks_to_run.append(label_path)
        except Exception as e:
            logging.error(f"预检查文件 {label_path.name} 时出错: {e}")

    # --- 3. 报告检查结果 ---
    if unmapped_labels:
        logging.warning("="*80)
        logging.warning("!!! 配置检查警告 !!!")
        logging.warning("发现以下标签类别未在 EVENT_CONFIG 中配置，包含这些标签的文件可能被跳过或部分处理：")
        for label_name in sorted(list(unmapped_labels)):
            logging.warning(f"  - {label_name}")
        logging.warning("请检查您的 EVENT_CONFIG 配置，将这些标签添加到相应的 'categories' 集合中。")
        logging.warning("="*80)

    total_tasks = len(all_label_files)
    remaining_tasks = len(tasks_to_run)
    logging.info(f"扫描完成。总计 {total_tasks} 个样本，已处理/跳过 {total_tasks - remaining_tasks} 个，剩余 {remaining_tasks} 个待处理。")

    if not tasks_to_run:
        logging.info("所有任务均已完成！程序退出。")
        return

    # --- 4. 并发执行任务 ---
    logging.info(f"开始使用 {args.workers} 个线程并发处理剩余任务...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 创建future列表
        futures = {
            executor.submit(
                process_sample,
                label_file_path, args.image_dir, args.output_dir,
                prompt_templates, category_map, client, args.model,
                max_retries=args.retries
            ): label_file_path
            for label_file_path in tasks_to_run
        }

        # 使用tqdm显示进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks_to_run), desc="并发处理中"):
            label_file = futures[future]
            try:
                future.result()  # 获取任务结果，如果任务中发生异常，这里会重新抛出
            except Exception as e:
                logging.error(f"处理文件 {label_file.name} 时发生严重错误: {e}", exc_info=True)

    logging.info("所有剩余任务处理完成！")


if __name__ == "__main__":
    main()
