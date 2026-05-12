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

# --- 配置区域 ---
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# === v2 事件配置 (已更新) ===
# 请确保下面的 prompt_path 指向您为v2事件新创建的、正确的prompt文件路径。
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
        "categories": {"违规横幅", "广告横幅"} # 拉条幅本身可以作为父目录，但具体的类别还是违规/广告横幅
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

# === 目录名到Config类别的映射别名 ===
# key: 目录名中可能出现的字符串 (需要与EVENT_CONFIG中的categories对应)
# value: EVENT_CONFIG["categories"] 里的标准名称
TAG_ALIAS: Dict[str, str] = {
    # 常用类别名称（可能直接作为目录名）
    "肢体冲突": "肢体冲突",
    "持械威胁": "持械威胁",
    "城市烟雾": "城市烟雾/火焰",
    "城市火焰": "城市烟雾/火焰",
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
    "非机动车乱停放": "非机动车乱停放",

    # 复合目录名（根据您提供的示例）
    "城市烟雾火焰": "城市烟雾/火焰", # 示例中的目录名
    "肢体冲突持械威胁": "肢体冲突", # 示例中的目录名，这里取其中一个作为主要类别，实际需根据prompt适配
    "绿地公共场所躺卧": "绿地躺卧/公共场所躺卧", # 示例中的目录名
    
    # "拉条幅"目录下的具体子类别
    "广告横幅正样本": "广告横幅",
    "违规横幅正样本": "违规横幅",
    # 如果"拉条幅"目录本身也可能包含图片，且需要一个默认类别：
    "拉条幅": "违规横幅", # 默认映射，如果更具体的横幅类型没被识别到

    # 英文标签 (文件名包含内容 - 尽管现在是目录名，但保持兼容性)
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

# 常见的非类别目录名，用于在路径遍历时跳过
IGNORED_DIR_NAMES: Set[str] = {
    "Yes", "No", "正样本", "负样本", "无效样本", "困难样本", 
    "无", "有", # 如果有类似这样的目录
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

def get_category_and_event_type_from_path(
    image_path: Path, 
    base_image_dir: Path, 
    category_map: Dict[str, str], 
    tag_alias_map: Dict[str, str],
    ignored_dir_names: Set[str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    根据图片路径及其父目录结构，从最接近的有效类别目录中提取类别和事件类型。
    返回 (原始目录名, 标准化类别名, 事件大类名)。
    例如: /base/不文明祭祀/Yes/img.jpg -> ("不文明祭祀", "不文明祭祀", "火灾烟雾与违规焚烧事件")
    例如: /base/拉条幅/386/违规横幅正样本/img.jpg -> ("违规横幅正样本", "违规横幅", "横幅事件")
    """
    
    # 从图片所在的父目录开始向上遍历
    current_dir = image_path.parent
    
    # 遍历直到达到base_image_dir或其上级
    # Path.parents会返回祖先目录，直到文件系统的根目录
    # 我们需要确保current_dir始终在base_image_dir之下或等于base_image_dir
    while current_dir != base_image_dir and current_dir.is_relative_to(base_image_dir):
        dir_name = current_dir.name
        
        # 忽略数字ID目录或常见的组织性目录
        if dir_name.isdigit() or dir_name in ignored_dir_names:
            current_dir = current_dir.parent
            continue

        # 尝试匹配目录名到标准化类别
        standard_category = tag_alias_map.get(dir_name)
        if standard_category:
            event_type = category_map.get(standard_category)
            if event_type:
                # 找到匹配的目录，返回其信息
                return dir_name, standard_category, event_type
        
        current_dir = current_dir.parent
    
    # 如果图片直接在base_image_dir下，或者base_image_dir本身就是一个类别目录
    # （例如，args.image_dir = "/ssddata/.../不文明祭祀"，且直接有图片在此目录下）
    # 检查base_image_dir的名称
    dir_name = base_image_dir.name
    standard_category = tag_alias_map.get(dir_name)
    if standard_category:
        event_type = category_map.get(standard_category)
        if event_type:
            return dir_name, standard_category, event_type

    return None, None, None # 未找到匹配的类别


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
def process_sample(
    image_path: Path, 
    output_dir: Path, 
    prompt_templates: Dict[str, str], 
    client: OpenAI, 
    model: str, 
    raw_dir_tag: str, # 实际匹配到的目录名
    category: str,     # 标准化后的类别名
    event_type: str,   # 事件大类名
    max_retries: int = 3
):
    """
    处理单个图片：根据传入的类别信息匹配Prompt -> 调用API -> 保存结果
    """
    file_name = image_path.name
    
    # 1. 检查是否有对应的 Prompt (类别信息已在外部解析并传入)
    if event_type not in prompt_templates:
        logging.warning(f"跳过 {file_name} (类别: '{category}', 事件: '{event_type}'): 没有加载到 Prompt 模板。")
        return

    # 2. 准备图像
    try:
        base64_large_image = encode_image_to_base64(image_path)
    except Exception as e:
        logging.error(f"编码大图 {image_path} 失败: {e}")
        return

    # 3. 调用 API
    analysis_result = None
    for attempt in range(max_retries):
        logging.info(f"处理 {file_name} -> '{category}' (事件: '{event_type}'): 尝试调用API (第 {attempt + 1}/{max_retries} 次)...")
        
        prompt_template = prompt_templates[event_type]
        # 假设新prompt仍然使用子事件列表的形式
        sub_event_list = [{"category_name": category}]
        sub_event_list_str = json.dumps(sub_event_list, ensure_ascii=False)
        final_prompt = prompt_template.replace("`[{\"category_name\": \"...\"}, ...]`", sub_event_list_str)
        
        analysis_result = call_vision_api(client, model, final_prompt, base64_large_image)
        
        if analysis_result:
            break
        else:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logging.error(f"处理 {file_name} 失败，已达最大重试次数。")

    # 4. 保存结果
    if analysis_result:
        # 在最终的JSON中添加一些溯源信息
        analysis_result["source_info"] = {
            "source_event_type": event_type,
            "source_category": category,
            "source_dir_tag": raw_dir_tag # 记录是哪个目录名匹配到的
        }
        
        final_output = { 
            "file_name": file_name, 
            "analysis_result": analysis_result 
        }
        
        output_subdir = output_dir / event_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        image_stem = image_path.stem
        output_json_path = output_subdir / f"{image_stem}.json"
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        logging.info(f"成功处理并保存结果: {output_json_path}")


# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="V2版本 - 遍历图片文件夹，根据子目录名称并发调用VLM生成分析结果。")
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-A22B-Instruct", help="模型名称。")
    
    # === 输入输出目录 (已更新为v2占位符) ===
    parser.add_argument("--image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第一批_260202/第一批260202/images", help="包含所有图片的根文件夹路径，类别目录应在此之下。") 
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第一批_260202/第一批260202/images_results_0126_p1_qwen3vl235B_dirbased", help="保存JSON分析结果的根目录。")     
    # === API 配置 (保持不变) ===
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
    if not prompt_templates:
        logging.critical("错误：没有任何prompt模板被成功加载，程序退出。请检查EVENT_CONFIG中的路径。")
        return
        
    category_map = build_category_map(EVENT_CONFIG)
    
    # === 初始化客户端 (保持不变) ===
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
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    
    if not all_images:
        logging.warning(f"在目录 '{image_dir}' 中未找到任何图片文件。")
        return

    # 2. 过滤已完成的任务与检查标签有效性
    tasks_to_run = []
    unrecognized_files = []
    
    logging.info("正在检查已完成任务和目录类别...")
    for img_path in tqdm(all_images, desc="预处理"):
        file_name = img_path.name
        
        # 使用新的函数从目录结构中提取类别信息
        raw_dir_tag, category, event_type = get_category_and_event_type_from_path(
            img_path, image_dir, category_map, TAG_ALIAS, IGNORED_DIR_NAMES
        )
        
        if not event_type:
            unrecognized_files.append(str(img_path)) # 记录完整路径，方便排查
            continue

        image_stem = img_path.stem
        output_json_path = output_dir / event_type / f"{image_stem}.json"
        
        if not output_json_path.exists():
            tasks_to_run.append((img_path, raw_dir_tag, category, event_type))
        # else: # 已经处理过的文件，跳过，不做额外打印，避免过多日志
        #     logging.debug(f"跳过已处理文件: {img_path}")
            
    # 3. 打印警告信息
    if unrecognized_files:
        logging.warning("="*80)
        logging.warning(f"发现 {len(unrecognized_files)} 个文件的父目录无法匹配到已知类别，将被跳过。部分示例：")
        for f in unrecognized_files[:10]: # 只打印前10个
            logging.warning(f"  - {f}")
        logging.warning("="*80)

    total_images = len(all_images)
    ready_tasks = len(tasks_to_run)
    skipped_count = total_images - ready_tasks
    logging.info(f"扫描到 {total_images} 张图片。")
    logging.info(f"需处理 {ready_tasks} 张 (已跳过 {skipped_count} 张已完成或无法识别目录类别的文件)。")

    if not tasks_to_run:
        logging.info("所有任务已完成！")
        return

    # 4. 并发执行
    logging.info(f"开始使用 {args.workers} 个线程进行并发处理...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for img_path, raw_dir_tag, category, event_type in tasks_to_run:
            futures.append(
                executor.submit(
                    process_sample, 
                    img_path, output_dir, 
                    prompt_templates, client, args.model,
                    raw_dir_tag, category, event_type, # 传入解析好的类别信息
                    max_retries=args.retries
                ) 
            )
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="并发处理中"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"任务执行异常: {e}", exc_info=True)

    logging.info("所有任务处理完成！")

if __name__ == "__main__":
    main()