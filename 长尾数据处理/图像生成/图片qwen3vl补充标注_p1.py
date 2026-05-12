import os
import json
import re
import base64
import argparse
import logging
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple, Set
import concurrent.futures
from tqdm import tqdm
import time
import shutil
from collections import defaultdict

# --- 配置区域 ---
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 1. P1 事件核心配置
EVENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "暴力冲突事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注_p1/cap_暴力冲突事件.md"),
        "categories": {"肢体冲突", "持械威胁"}
    },
    "火灾烟雾与违规焚烧事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注_p1/cap_火灾烟雾与违规焚烧事件.md"),
        "categories": {"城市烟雾/火焰", "垃圾焚烧", "不文明祭祀"}
    },
    "横幅事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注_p1/cap_横幅事件.md"),
        "categories": {"违规横幅", "广告横幅"}
    },
    "垃圾倾倒事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注_p1/cap_垃圾倾倒事件.md"),
        "categories": {"垃圾倾倒"}
    },
    "渣土车遗洒事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注_p1/cap_渣土车遗洒事件.md"),
        "categories": {"渣土车遗洒"}
    },
    "雨水井垃圾监测事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注_p1/cap_雨水井垃圾监测.md"),
        "categories": {"雨水井垃圾监测"}
    },
    "公共场所躺卧事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注_p1/cap_公共场所躺卧.md"),
        "categories": {"绿地躺卧/公共场所躺卧"}
    },
    "非机动车乱停放事件": {
        "prompt_path": Path("/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_标注/cap_非机动车乱停放事件.md"),
        "categories": {"非机动车乱停放"}
    },
}

# 2. P1 目录名别名映射 (从 p1 代码移植)
# Key: 目录名 -> Value: 标准类别名 (对应 EVENT_CONFIG 中的 categories)
TAG_ALIAS: Dict[str, str] = {
    # 常用类别名称
    "肢体冲突": "肢体冲突",
    "持械威胁": "持械威胁",
    "城市烟雾": "城市烟雾/火焰",
    "城市火焰": "城市烟雾/火焰",
    "城市烟雾火焰": "城市烟雾/火焰",
    "垃圾焚烧": "垃圾焚烧",
    "不文明祭祀": "不文明祭祀",
    "烧纸": "不文明祭祀",
    "违规横幅": "违规横幅",
    "广告横幅": "广告横幅",
    "垃圾倾倒": "垃圾倾倒",
    "渣土车遗洒": "渣土车遗洒",
    "雨水井垃圾": "雨水井垃圾监测",
    "雨水井堵塞": "雨水井垃圾监测",
    "绿地躺卧": "绿地躺卧/公共场所躺卧",
    "公共场所躺卧": "绿地躺卧/公共场所躺卧",
    "绿地公共场所躺卧": "绿地躺卧/公共场所躺卧",
    "非机动车乱停放": "非机动车乱停放",
    
    # 复合或特定目录
    "肢体冲突持械威胁": "肢体冲突", 
    "广告横幅正样本": "广告横幅",
    "违规横幅正样本": "违规横幅",
    "拉横幅": "违规横幅", 

    # 英文/其他别名
    "Physical_Conflict": "肢体冲突",
    "Armed_Threat": "持械威胁",
    "Weapon_Threat": "持械威胁",
    "City_Fire_Smoke": "城市烟雾/火焰",
    "Fire_Smoke": "城市烟雾/火焰",
    "Garbage_Burning": "垃圾焚烧",
    "Uncivil_Worship": "不文明祭祀",
    "Illegal_Banner": "违规横幅",
    "Ad_Banner": "广告横幅",
    "Garbage_Dumping": "垃圾倾倒",
    "Illegal_Dumping": "垃圾倾倒",
    "Muck_Truck_Spill": "渣土车遗洒",
    "Spilled_Soil": "渣土车遗洒",
    "Storm_Drain_Debris": "雨水井垃圾监测",
    "Drain_Blockage": "雨水井垃圾监测",
    "Public_Lying_Down": "绿地躺卧/公共场所躺卧",
    "Lawn_Lying": "绿地躺卧/公共场所躺卧",
    "Bike_Parking": "非机动车乱停放",
}

# 忽略的目录名

# 3. 自动构建辅助映射 (程序启动时生成)
# Category -> Event Key
CATEGORY_TO_EVENT_KEY: Dict[str, str] = {}
for evt, details in EVENT_CONFIG.items():
    for cat in details["categories"]:
        CATEGORY_TO_EVENT_KEY[cat] = evt

# Event Key -> Source Label Dir (p1 原有标注代码直接使用 Event Key 作为目录名)
CONFIG_KEY_TO_SOURCE_LABEL_DIR: Dict[str, str] = {k: k for k in EVENT_CONFIG.keys()}


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
    """
    根据图片路径，结合 TAG_ALIAS 解析出 (Event_Key, Standard_Category)。
    采用向上遍历父目录的方式，直到找到匹配 TAG_ALIAS 的目录名。
    """
    try:
        current_dir = image_path.parent
        # 向上遍历直到根目录
        while current_dir != image_root_dir and current_dir.is_relative_to(image_root_dir):
            dir_name = current_dir.name
            
            # 尝试匹配
            standard_category = TAG_ALIAS.get(dir_name)
            if standard_category:
                event_key = CATEGORY_TO_EVENT_KEY.get(standard_category)
                if event_key:
                    return event_key, standard_category
            
            current_dir = current_dir.parent
            
        # 如果未找到，尝试检查 image_root_dir 本身的名称（以防根目录就是类别）
        dir_name = image_root_dir.name
        standard_category = TAG_ALIAS.get(dir_name)
        if standard_category:
            event_key = CATEGORY_TO_EVENT_KEY.get(standard_category)
            if event_key:
                return event_key, standard_category

        logging.warning(f"无法从路径 {image_path} 解析出有效的 p1 事件类别。")
        return None, None
    except Exception as e:
        logging.warning(f"解析路径 {image_path} 时出错: {e}")
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
        logging.info(f"处理 {image_path.name} -> '{category}' (事件: {config_key}): 调用API (第 {attempt + 1}/{max_retries} 次)...")

        # 适配 p1 prompt 的格式: `[{"category_name": "..."}]`
        sub_event_list = [{"category_name": category}]
        sub_event_list_str = json.dumps(sub_event_list, ensure_ascii=False)
        
        # 这里的替换逻辑需根据 p1 的 prompt 实际占位符调整
        # 假设 p1 prompt 中的占位符结构类似于 p0 或能兼容列表替换
        # 如果 p1 prompt 使用的是 `__待检测类别占位符__`，则保持下行；
        # 如果 p1 prompt 使用的是 `[{"category_name": "..."}]` 示例，则替换对应部分。
        # 此处为了兼容性，假设 prompt 中有特定的占位符，或者替换整个列表结构
        # 常见占位符处理：
        final_prompt = prompt_template
        if "__待检测类别占位符__" in final_prompt:
             # p0 风格占位符
             final_prompt = final_prompt.replace("__待检测类别占位符__", json.dumps([category], ensure_ascii=False))
        elif "`[{\"category_name\": \"...\"}, ...]`" in final_prompt:
             # p1 风格占位符
             final_prompt = final_prompt.replace("`[{\"category_name\": \"...\"}, ...]`", sub_event_list_str)
        else:
             # 如果找不到显式占位符，可能需要根据具体prompt文件调整，这里默认尝试替换p1风格
             final_prompt = final_prompt.replace("`[{\"category_name\": \"...\"}, ...]`", sub_event_list_str)


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
        # 添加溯源信息
        analysis_result["source_event_type"] = config_key
        analysis_result["source_category"] = category
        analysis_result["source_image_path"] = str(image_path)

        # 保持 p0 的输出结构： { file_name: ..., analyses: [...] }
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
    parser = argparse.ArgumentParser(description="p1补充标注：增量式标注图片，强制重新生成负样本和重复样本。")
    # 默认路径修改为示例中的 p1 相关路径，请根据实际情况调整
    parser.add_argument("--image_root_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第二批_260209/images", help="图片根目录。")
    parser.add_argument("--source_label_dir", default="/host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/生成_results_0126_p1_qwen3vl235B", help="存放旧JSON标签的目录（p1第一批结果）。")
    parser.add_argument("--output_label_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第二批_260209/labels", help="保存最终JSON标签的根目录。")
    
    # --- API及模型参数 ---
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-A22B-Instruct", help="模型名称。")
    parser.add_argument("--api_key", type=str, default="McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ", help="API密钥。")                                                                                                                                                                             
    parser.add_argument("--api_base_url", type=str, default="https://arms-referring-matthew-appendix.trycloudflare.com/v1/", help="API的基础URL。")
    parser.add_argument("--proxy_secret", type=str, default="my_secret_password_sensoro_123", help="x-proxy-secret 头的值。")
    parser.add_argument("--workers", type=int, default=4, help="并发处理的线程数。")
    parser.add_argument("--retries", type=int, default=3, help="API调用失败后的最大重试次数。")
    args = parser.parse_args()

    # --- 初始化 ---
    prompt_templates = load_all_prompts(EVENT_CONFIG)
    if not prompt_templates:
        logging.critical("未加载到任何 Prompt 模板，请检查路径。")
        return

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
    unrecognized_count = 0

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
            unrecognized_count += 1
            continue
            
        # 规则1: 是否为负样本? (根据路径中是否包含特定关键词)
        is_negative = "负样本" in image_path.parts or "Negative" in str(image_path)
        # 规则2: 是否为重复图片?
        is_duplicate = image_path.name in duplicate_image_names

        # 如果满足任一强制生成规则
        if is_negative or is_duplicate:
            # reason = "负样本" if is_negative else "重复图片"
            tasks_to_generate.append((image_path, output_json_path, config_key, category))
            forced_regeneration_count += 1
            continue

        # 规则3: 对于唯一的正样本，尝试复用旧标签
        # P1 的旧标签结构通常是 source_root / EventType / image.json
        source_dir_name = CONFIG_KEY_TO_SOURCE_LABEL_DIR.get(config_key)
        
        source_json_path = None
        if source_dir_name:
            source_json_path = source_label_dir / source_dir_name / f"{image_path.stem}.json"

        if source_json_path and is_valid_json_file(source_json_path):
            try:
                # 注意：如果 p1 旧标签格式与 p0 不一致（旧的可能是直接 dict，新的要求 dict inside list），
                # 严格来说这里应该读取旧json，转换格式后再保存。
                # 但根据需求描述“p1补充标注生成的labels和p0补充标注的保存结构一致”，
                # 假设旧标签内容是可用的，或者我们仅复用文件。
                # 为了安全，这里做一个简单的格式转换逻辑：
                with open(source_json_path, 'r', encoding='utf-8') as src_f:
                    old_data = json.load(src_f)
                
                # 检查是否已经是 p0 格式 (含 analyses 列表)
                if "analyses" in old_data:
                    final_data = old_data
                else:
                    # 如果是 p1 旧格式 (analysis_result 在根节点或 analysis_result 键下)
                    inner_result = old_data.get("analysis_result", old_data)
                    final_data = {
                        "file_name": image_path.name,
                        "analyses": [inner_result]
                    }

                output_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_json_path, 'w', encoding='utf-8') as dst_f:
                    json.dump(final_data, dst_f, ensure_ascii=False, indent=2)
                
                copied_count += 1
            except Exception as e:
                logging.error(f"处理旧标签 {source_json_path} 失败: {e}，将尝试重新生成。")
                tasks_to_generate.append((image_path, output_json_path, config_key, category))
        else:
            # 没有有效的旧标签，加入生成队列
            tasks_to_generate.append((image_path, output_json_path, config_key, category))

    # --- 3. 打印任务统计信息 ---
    logging.info("="*50)
    logging.info("预处理完成！")
    logging.info(f"总图片数: {len(all_images)}")
    logging.info(f"无法识别类别 (跳过): {unrecognized_count}")
    logging.info(f"已跳过 (目标JSON已存在): {skipped_count}")
    logging.info(f"成功复用旧标签: {copied_count}")
    logging.info(f"需要生成新标签: {len(tasks_to_generate)}")
    logging.info(f"  - 其中强制生成 (负样本/重复): {forced_regeneration_count}")
    logging.info(f"  - 其中常规生成 (无有效旧标签): {len(tasks_to_generate) - forced_regeneration_count}")
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