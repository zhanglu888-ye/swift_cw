# 文件名: your_vlm_script.py (或您给它起的名字)
import os
import json
import base64
import argparse
import logging
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

EVENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "垃圾事件": {
        "prompt_path": Path("4类长尾事件cap_垃圾堆积.md"),
        "categories": {"垃圾桶满溢", "垃圾堆积"}
    },
    "倒地事件": {
        "prompt_path": Path("4类长尾事件cap_倒地.md"),
        "categories": {"人员异常倒地"}
    },
    "占道经营事件": {
        "prompt_path": Path("4类长尾事件cap_占道经营.md"),
        "categories": {"流动摊贩", "出店经营"}
    },
    "人群聚集事件": {
        "prompt_path": Path("4类长尾事件cap_聚集.md"),
        "categories": {"异质文化异常聚集", "开放空间异常聚集"}
    }
}

def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_all_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    templates = {}
    for event_type, details in config.items():
        try:
            templates[event_type] = details["prompt_path"].read_text(encoding='utf-8')
        except Exception as e:
            logging.error(f"读取Prompt '{details['prompt_path']}' 失败: {e}")
            raise
    return templates

def call_vision_api(client: OpenAI, model: str, prompt: str, base64_large_image: str, sub_events_for_api: List[Dict[str, Any]]) -> Any:
    messages_content = [{"type": "text", "text": prompt}]
    messages_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_large_image}"}})
    for event in sub_events_for_api:
        messages_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{event['base64_image']}"}})
    
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": messages_content}],
            max_tokens=6144, temperature=0.0, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"调用API时发生错误: {e}")
        return None

def process_case(case_dir: Path, large_image_dir: Path, output_dir: Path, prompt_templates: Dict[str, str], client: OpenAI, model: str):
    case_name = case_dir.name
    logging.info(f"--- 正在处理案例: {case_name} ---")

    large_image_path = next(large_image_dir.glob(f"{case_name}.*"), None)
    if not large_image_path:
        logging.warning(f"找不到大图 for '{case_name}'，跳过。")
        return

    events_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for sub_dir in case_dir.iterdir():
        if not sub_dir.is_dir(): continue
        
        # 新增：从元数据文件中读取类别和Bbox
        for meta_path in sub_dir.glob('*.json'):
            img_path = meta_path.with_suffix('.jpg')
            if not img_path.exists(): continue
            
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            category_name = meta_data["category_name"]
            bbox = meta_data["original_bbox"]
            
            matched_event_type = None
            for event_type, details in EVENT_CONFIG.items():
                if category_name in details["categories"]:
                    matched_event_type = event_type
                    break
            
            if matched_event_type:
                if matched_event_type not in events_by_type:
                    events_by_type[matched_event_type] = []
                events_by_type[matched_event_type].append({
                    "path": img_path,
                    "category_name": category_name,
                    "bbox": bbox
                })

    if not events_by_type:
        logging.info(f"案例 '{case_name}' 中未找到已配置的事件类别，跳过。")
        return

    try:
        base64_large_image = encode_image_to_base64(large_image_path)
    except Exception as e:
        logging.error(f"无法处理大图 {large_image_path} for '{case_name}'。")
        return

    for event_type, sub_events in events_by_type.items():
        logging.info(f"  -> 正在为 '{case_name}' 处理 '{event_type}' ({len(sub_events)} 个抠图)...")

        sub_events_for_prompt, sub_events_for_api = [], []
        original_bboxes = [] # 暂存Bbox，顺序与API发送的图片一致
        for i, event in enumerate(sub_events):
            try:
                sub_events_for_prompt.append({"crop_image": f"[Image_Crop_{i+1}]", "category_name": event["category_name"]})
                sub_events_for_api.append({"base64_image": encode_image_to_base64(event["path"])})
                original_bboxes.append(event["bbox"])
            except Exception as e:
                logging.warning(f"处理小图 {event['path']} 失败: {e}")

        if not sub_events_for_api: continue

        prompt_template = prompt_templates[event_type]
        sub_event_list_str = json.dumps(sub_events_for_prompt, ensure_ascii=False, indent=2)
        final_prompt = prompt_template.split("#### # 输入信息")[0] + (
            f"#### # 输入信息\n\n*   **主图片**: [Image_Main]\n*   **子事件列表**: `{sub_event_list_str}`"
        )
        
        analysis_result = call_vision_api(client, model, final_prompt, base64_large_image, sub_events_for_api)

        if analysis_result:
            # 新增：将Bbox注入到返回结果中
            # 假设VLM返回的 event_analyses 列表与我们发送的子事件列表顺序一致
            if "event_analyses" in analysis_result and len(analysis_result["event_analyses"]) == len(original_bboxes):
                for i, analysis in enumerate(analysis_result["event_analyses"]):
                    bbox = original_bboxes[i]
                    bbox_str = " ".join(map(str, bbox))
                    if "comprehensive_assessment" in analysis:
                        analysis["comprehensive_assessment"]["bbox"] = bbox_str
            else:
                logging.warning(f"VLM返回的分析数量与发送的抠图数量不匹配，无法注入Bbox。")

            output_subdir = output_dir / event_type
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_json_path = output_subdir / f"{case_name}_{event_type}.json"
            
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            logging.info(f"    [+] 分析成功！结果已保存至: {output_json_path}")
        else:
            logging.error(f"  -> 案例 '{case_name}' 的 '{event_type}' API调用失败。")

def main():
    parser = argparse.ArgumentParser(description="根据图片中的目标类别，动态调用不同的VLM Prompt进行分析。")
    parser.add_argument("--large_image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并_deduplicated/large_images", help="包含原始大图的文件夹路径 (由prepare_vlm_input.py生成)。")
    parser.add_argument("--cropped_data_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并_deduplicated/cropped_data", help="包含抠图和元数据的根目录 (由prepare_vlm_input.py生成)。")
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm", help="保存JSON分析结果的根目录。")
    parser.add_argument(
        "--api_key", type=str, default='1h1K6lRCB4M2s7n94y14Yr7C9i9D2C8ilr4J3wSq2H4q8T2C',
        help="OpenAI API密钥。如果未提供，将尝试从环境变量 'OPENAI_API_KEY' 读取。"
    )
    parser.add_argument(
        "--api_base_url", type=str, default="http://0.0.0.0:23333/v1/",
        help="OpenAI API的基础URL。可用于指向兼容的第三方服务。"
    )
    parser.add_argument(
        "--model", type=str, 
        # default="doubao-pro-vision-2",
        default="/nvme0n1/model_repository/Qwen/Qwen2.5-VL-72B-Instruct",
        help="要使用的模型名称。"
    )
    args = parser.parse_args()

    if not args.api_key:
        logging.critical("致命错误: 未提供API密钥。")
        return

    try:
        prompt_templates = load_all_prompts(EVENT_CONFIG)
        client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)
        
        for case_dir in Path(args.cropped_data_dir).iterdir():
            if case_dir.is_dir():
                process_case(case_dir, Path(args.large_image_dir), Path(args.output_dir), prompt_templates, client, args.model)
        logging.info("所有任务处理完成！")
    except Exception as e:
        logging.critical(f"主流程发生错误: {e}")

if __name__ == "__main__":
    main()