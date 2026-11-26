# 文件名: vqa_concurrent_annotator.py
import os
import json
import base64
import argparse
import logging
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import demjson3 as demjson # 用于修复不完美的JSON

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ==============================================================================
# --- VQA事件配置中心 ---
# ==============================================================================
VQA_EVENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "VQA_垃圾事件": {
        "prompt_path": Path("4类长尾事件vqa_垃圾堆积.md"),
        "categories": {"垃圾桶满溢", "垃圾堆积"}
    },
    "VQA_倒地事件": {
        "prompt_path": Path("4类长尾事件vqa_倒地.md"),
        "categories": {"人员异常倒地"}
    },
    "VQA_占道经营事件": {
        "prompt_path": Path("4类长尾事件vqa_占道经营.md"),
        "categories": {"流动摊贩", "出店经营"}
    },
    "VQA_人群聚集事件": {
        "prompt_path": Path("4类长尾事件vqa_聚集.md"),
        "categories": {"异质文化异常聚集", "开放空间异常聚集"}
    }
}
# ==============================================================================

# --- 核心函数 ---

def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_all_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    templates = {}
    for event_type, details in config.items():
        try:
            template = details["prompt_path"].read_text(encoding='utf-8')
            if "{{category_name}}" not in template:
                logging.warning(f"警告: Prompt '{details['prompt_path']}' 中缺少 '{{category_name}}' 占位符。")
            templates[event_type] = template
        except Exception as e:
            logging.error(f"读取Prompt '{details['prompt_path']}' 失败: {e}")
            raise
    return templates

def call_vqa_api(client: OpenAI, model: str, prompt: str, base64_large_image: str, base64_crop_image: str) -> Any:
    """调用API，返回原始文本或解析后的字典。"""
    messages_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_large_image}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_crop_image}"}}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": messages_content}],
            max_tokens=2048, temperature=0.0, stream=False
        )
        response_text = response.choices[0].message.content
        if not response_text or not response_text.strip():
            return "API_EMPTY_RESPONSE"

        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"): cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            logging.warning("标准JSON解析失败，尝试demjson修复...")
            try:
                return demjson.decode(cleaned_text)
            except demjson.JSONDecodeError:
                logging.error("demjson修复失败，将保存原始文本。")
                return cleaned_text # 返回无法解析的原始文本
    except Exception as e:
        logging.error(f"调用API时发生严重错误: {e}")
        return None

def process_single_crop(
    crop_info: Dict[str, Any],
    large_image_b64: str,
    prompt_templates: Dict[str, str],
    client: OpenAI,
    model: str,
    output_dir: Path
):
    """
    处理单个抠图的核心逻辑，将被并发调用。
    """
    case_name = crop_info["case_name"]
    crop_path = crop_info["path"]
    category_name = crop_info["category_name"]
    bbox = crop_info["bbox"]
    event_type = crop_info["event_type"]

    try:
        # 1. 编码小图
        base64_crop_image = encode_image_to_base64(crop_path)
        
        # 2. 准备Prompt
        prompt_template = prompt_templates[event_type]
        final_prompt = prompt_template.replace("{{category_name}}", category_name)

        # 3. 调用API
        vqa_result = call_vqa_api(
            client=client, model=model, prompt=final_prompt,
            base64_large_image=large_image_b64, base64_crop_image=base64_crop_image
        )

        # 4. 处理并保存结果
        output_subdir = output_dir / event_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_filename_base = f"{case_name}_{crop_path.stem}"

        if isinstance(vqa_result, dict):
            # 注入Bbox
            vqa_result["original_bbox"] = " ".join(map(str, bbox))
            output_path = output_subdir / f"{output_filename_base}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(vqa_result, f, ensure_ascii=False, indent=2)
            return f"Success (JSON): {output_path.name}"
        elif isinstance(vqa_result, str):
            # 保存无法解析的文本
            output_path = output_subdir / f"{output_filename_base}_bad_response.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(vqa_result)
            return f"Failed (Bad Format): {output_path.name}"
        else: # is None or empty
            return f"Failed (API Error): {case_name}/{crop_path.name}"

    except Exception as e:
        logging.error(f"处理抠图 {crop_path} 时发生未知错误: {e}")
        return f"Failed (Exception): {case_name}/{crop_path.name}"


def main():
    parser = argparse.ArgumentParser(description="【VQA模式】并发处理VLM问答任务。")
    parser.add_argument("--large_image_dir", required=True, help="包含原始大图的文件夹路径。")
    parser.add_argument("--cropped_data_dir", required=True, help="包含抠图和元数据的根目录。")
    parser.add_argument("--output_dir", required=True, help="保存VQA结果的根目录。")
    parser.add_argument("--api_key", type=str, required=True, help="API密钥。")
    parser.add_argument("--api_base_url", type=str, default="http://0.0.0.0:23333/v1/", help="API的基础URL。")
    parser.add_argument("--model", type=str, required=True, help="模型名称。")
    parser.add_argument("--workers", type=int, default=8, help="并发处理的线程数。")
    args = parser.parse_args()

    try:
        # 1. 预加载资源
        prompt_templates = load_all_prompts(VQA_EVENT_CONFIG)
        client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)
        
        # 2. 收集所有待处理的任务（以小图为单位）
        all_tasks = []
        all_case_dirs = [d for d in Path(args.cropped_data_dir).iterdir() if d.is_dir()]

        logging.info("正在扫描并收集所有待处理的抠图任务...")
        for case_dir in tqdm(all_case_dirs, desc="扫描案例"):
            case_name = case_dir.name
            large_image_path = next(Path(args.large_image_dir).glob(f"{case_name}.*"), None)
            if not large_image_path:
                logging.warning(f"找不到大图 for '{case_name}'，跳过此案例。")
                continue
            
            # 每个案例只需编码一次大图
            large_image_b64 = encode_image_to_base64(large_image_path)
            
            for sub_dir in case_dir.iterdir():
                if not sub_dir.is_dir(): continue
                for meta_path in sub_dir.glob('*.json'):
                    img_path = meta_path.with_suffix('.jpg')
                    if not img_path.exists(): continue
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    category_name = meta_data["category_name"]
                    
                    # 匹配VQA事件类型
                    matched_event_type = next((et for et, d in VQA_EVENT_CONFIG.items() if category_name in d["categories"]), None)
                    if matched_event_type:
                        all_tasks.append({
                            "crop_info": {
                                "case_name": case_name,
                                "path": img_path,
                                "category_name": category_name,
                                "bbox": meta_data["original_bbox"],
                                "event_type": matched_event_type
                            },
                            "large_image_b64": large_image_b64
                        })
        
        if not all_tasks:
            logging.warning("未找到任何可处理的抠图任务。")
            return

        logging.info(f"共发现 {len(all_tasks)} 个VQA任务，将使用 {args.workers} 个并发线程进行处理...")

        # 3. 使用ThreadPoolExecutor进行并发处理
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(
                    process_single_crop, 
                    task["crop_info"],
                    task["large_image_b64"],
                    prompt_templates,
                    client,
                    args.model,
                    Path(args.output_dir)
                ): task["crop_info"]["path"].name 
                for task in all_tasks
            }
            
            progress_bar = tqdm(as_completed(future_to_task), total=len(all_tasks), desc="执行VQA任务")
            for future in progress_bar:
                crop_name = future_to_task[future]
                try:
                    result = future.result()
                    progress_bar.set_postfix_str(f"{crop_name}: {result}")
                except Exception as exc:
                    logging.error(f"任务 '{crop_name}' 在执行期间产生了一个错误: {exc}")

        logging.info("所有VQA任务处理完成！")

    except Exception as e:
        logging.critical(f"主流程发生严重错误: {e}")

if __name__ == "__main__":
    main()