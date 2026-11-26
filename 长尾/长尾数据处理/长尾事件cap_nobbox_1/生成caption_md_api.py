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

# --- 1. API 与 模型配置 (来自代码2) ---
# ModelScope API 配置
MS_API_KEY = "ms-5acdd5f7-23ad-49eb-9bee-842863de3578" # 请确保这里是你完整的 Key，如果原代码中是截断的，请替换为真实 Key
MS_BASE_URL = 'https://api-inference.modelscope.cn/v1'
MS_MODEL_ID = 'qwen/Qwen2.5-VL-72B-Instruct'
# --- 2. 业务配置 (来自代码1) ---
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 事件 Prompt 配置
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

# --- 3. 辅助函数 ---

def encode_image_to_base64(image_path: Path) -> str:
    """将图片转为base64字符串"""
    try:
        with open(image_path, "rb") as image_file: 
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"图片读取失败: {image_path}, 错误: {e}")
        return ""

def load_all_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    """加载所有MD Prompt模板"""
    templates = {}
    for event_type, details in config.items():
        try:
            templates[event_type] = details["prompt_path"].read_text(encoding='utf-8')
            logging.info(f"成功加载Prompt: '{event_type}'")
        except FileNotFoundError:
            logging.error(f"致命错误: Prompt文件 '{details['prompt_path']}' 未找到！")
            raise
    return templates

def extract_json_from_response(text: str) -> Optional[Any]:
    """从大模型返回的文本中提取JSON"""
    # 尝试匹配 ```json { ... } ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = None
    if match:
        json_str = match.group(1)
    else:
        # 尝试匹配最外层的大括号
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
    
    if json_str:
        try: 
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 简单的容错尝试
            try:
                # 有时候模型会在末尾多加逗号
                return json.loads(json_str.replace(",\n}", "\n}"))
            except:
                pass
    return None

def call_vision_api(client: OpenAI, prompt: str, base64_image: str) -> Any:
    """调用 ModelScope API"""
    if not base64_image:
        return None

    messages_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MS_MODEL_ID, 
            messages=[{"role": "user", "content": messages_content}],
            temperature=0.2, # 保持较低的温度以获得稳定输出
            max_tokens=4096,
        )
        raw_content = response.choices[0].message.content
        parsed_json = extract_json_from_response(raw_content)
        
        if parsed_json:
            return parsed_json
        else:
            logging.warning(f"无法提取JSON，原始内容: {raw_content[:100]}...")
            return None
    except Exception as e:
        logging.warning(f"API调用异常: {e}")
        return None

# --- 4. 任务筛选逻辑 ---

def find_relabel_tasks(input_result_dir: Path) -> List[Dict]:
    """
    遍历之前的标注结果，寻找需要重标的任务。
    条件：'comprehensive_assessment' -> '异常类型' == '无异常'
    """
    tasks = []
    # 递归查找所有 .json 文件
    json_files = sorted(list(input_result_dir.rglob('*.json')))
    
    logging.info(f"正在扫描目录 '{input_result_dir}' 寻找需重标数据 (共 {len(json_files)} 个文件)...")
    
    for json_path in tqdm(json_files, desc="扫描筛选中"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_name = data.get('file_name')
            analyses = data.get('analyses', [])
            
            if not file_name or not analyses:
                continue

            # 检查是否有“无异常”的判断
            target_analysis = None
            needs_relabel = False

            for analysis in analyses:
                # 按照示例结构寻找
                event_analyses = analysis.get('event_analyses', [])
                if event_analyses:
                    # 获取第一个分析结果的评估
                    assessment = event_analyses[0].get('comprehensive_assessment', {})
                    if assessment.get('异常类型') == "无异常":
                        needs_relabel = True
                        target_analysis = analysis
                        break 
            
            if needs_relabel and target_analysis:
                # 提取必要信息，因为重标需要知道用哪个Prompt (source_event_type) 和哪些类别 (source_categories)
                source_event_type = target_analysis.get('source_event_type')
                source_categories = target_analysis.get('source_categories')
                
                if source_event_type and source_categories:
                    tasks.append({
                        "file_name": file_name,
                        "event_type": source_event_type,
                        "categories": source_categories,
                        "original_path": str(json_path)
                    })
        except Exception as e:
            # 文件损坏或格式不对跳过
            continue
            
    return tasks

# --- 5. 处理单个任务 ---

def process_single_task(task: Dict, image_dir: Path, output_dir: Path, prompt_templates: Dict[str, str], client: OpenAI, max_retries: int = 3):
    """
    执行单个重标任务
    """
    file_name = task['file_name']
    event_type = task['event_type']
    categories = task['categories']
    
    # 1. 检查图片
    image_path = image_dir / file_name
    if not image_path.exists():
        logging.warning(f"图片缺失: {file_name}")
        return

    # 2. 准备 Prompt
    if event_type not in prompt_templates:
        logging.error(f"缺失Prompt模板: {event_type}")
        return
        
    prompt_template = prompt_templates[event_type]
    category_list_str = json.dumps(categories, ensure_ascii=False)
    final_prompt = prompt_template.replace("__待检测类别占位符__", category_list_str)

    # 3. 编码图片
    base64_img = encode_image_to_base64(image_path)
    if not base64_img:
        return

    # 4. API 调用 (含重试)
    analysis_result = None
    for attempt in range(max_retries):
        analysis_result = call_vision_api(client, final_prompt, base64_img)
        if analysis_result:
            break
        time.sleep(1) # 失败等待

    if not analysis_result:
        logging.error(f"处理失败: {file_name} ({event_type})")
        return

    # 5. 组装结果 (保持与原始结构一致)
    analysis_result["source_event_type"] = event_type
    analysis_result["source_categories"] = categories
    
    final_output = {
        "file_name": file_name,
        "analyses": [analysis_result]
    }

    # 6. 保存结果
    # 按照事件类型创建子文件夹
    save_subdir = output_dir / event_type
    save_subdir.mkdir(parents=True, exist_ok=True)
    
    output_json_path = save_subdir / f"{Path(file_name).stem}.json"
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    logging.info(f"已重标保存: {file_name} -> {event_type}")

# --- 6. 主程序入口 ---

def main():
    parser = argparse.ArgumentParser(description="筛选'无异常'数据并利用ModelScope API重新标注。")
    
    # 路径参数
    parser.add_argument("--image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/正样本/合并_deduplicated/large_images", help="原始大图文件夹路径") 
    parser.add_argument("--input_result_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/正样本/分类and描述_正样本_删除_tmp", help="包含之前标注结果的文件夹(从中筛选无异常)")            
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/正样本/分类and描述_正样本_删除_tmp_重标无异常", help="重标结果保存路径")   
    
    # API 参数 (默认使用代码中提供的)
    parser.add_argument("--api_key", type=str, default=MS_API_KEY, help="ModelScope API Key")
    parser.add_argument("--workers", type=int, default=1, help="并发线程数")
    parser.add_argument("--retries", type=int, default=3, help="API重试次数")

    args = parser.parse_args()
    
    # 初始化
    if not args.api_key or args.api_key == "ms-5acdd5f7-":
        logging.warning("警告: 使用的 API Key 可能不完整，请检查代码或参数。")

    image_dir = Path(args.image_dir)
    input_result_dir = Path(args.input_result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 Prompt 模板
    try:
        prompt_templates = load_all_prompts(EVENT_CONFIG)
    except Exception as e:
        logging.error(f"Prompt加载失败: {e}")
        return

    # 2. 初始化 OpenAI Client (连接 ModelScope)
    client = OpenAI(
        api_key=args.api_key,
        base_url=MS_BASE_URL,
    )

    # 3. 扫描并筛选任务
    if not input_result_dir.exists():
        logging.error(f"输入目录不存在: {input_result_dir}")
        return

    all_tasks = find_relabel_tasks(input_result_dir)
    logging.info(f"共发现 {len(all_tasks)} 个需重标的任务。")

    if not all_tasks:
        return

    # 4. 过滤已完成任务 (断点续传)
    tasks_to_run = []
    for task in all_tasks:
        image_stem = Path(task['file_name']).stem
        target_path = output_dir / task['event_type'] / f"{image_stem}.json"
        if not target_path.exists():
            tasks_to_run.append(task)
    
    logging.info(f"扣除已存在结果，实际需执行 {len(tasks_to_run)} 个任务。")

    # 5. 并发执行
    if tasks_to_run:
        logging.info(f"开始使用 {args.workers} 个线程处理...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # 提交任务
            futures = [
                executor.submit(
                    process_single_task, 
                    task, 
                    image_dir, 
                    output_dir, 
                    prompt_templates, 
                    client, 
                    args.retries
                ) 
                for task in tasks_to_run
            ]
            
            # 进度条
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="重标进度"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"线程执行异常: {e}")

    logging.info("所有处理完成！")

if __name__ == "__main__":
    main()