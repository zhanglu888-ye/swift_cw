import json
import os
import re
import logging
from pathlib import Path
from tqdm import tqdm

# --- 配置区域 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 输入1：Model 1 的 JSONL 文件路径 (微调模型结果)
MODEL1_JSONL_PATH = "/host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/生成_results_v3_0123_p0_测试集_finetone_md2560.jsonl"

# 输入2：Model 2 的结果根目录 (Qwen3-235B结果)
MODEL2_ROOT_DIR = "/host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/生成_results_v3_0123_p0_qwen3vl235B"

# 输出：合并后的新目录
MERGED_OUTPUT_DIR = "/host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/生成_results_v3_0123_p0_qwen3vl235B_finetone_md2560"

# --- 映射配置 ---
EVENT_CONFIG = {
    "垃圾事件": {"categories": {"垃圾桶满溢", "垃圾堆积"}},
    "倒地事件": {"categories": {"人员异常倒地"}},
    "占道经营事件": {"categories": {"流动摊贩", "占道经营"}},
    "人群聚集事件": {"categories": {"人员异常聚集"}},
    "雨水井倾倒污水事件": {"categories": {"污水井倾倒"}},
    "非机动车乱停放事件": {"categories": {"非机动车乱停放"}},
}

TAG_ALIAS = {
    # 中文
    "出店经营": "占道经营", "流动摊贩": "流动摊贩", "人员倒地": "人员异常倒地",
    "人员聚集": "人员异常聚集", "倾倒污水": "污水井倾倒", "垃圾桶满溢": "垃圾桶满溢",
    "垃圾堆积": "垃圾堆积", "垃圾堆积 - 无桶": "垃圾堆积", "非机动车乱停放": "非机动车乱停放",
    # 英文
    "Trash_Overflow": "垃圾桶满溢", "Trash_Accumulation": "垃圾堆积",
    "Mobile_Vendor": "流动摊贩", "Store_Outside": "占道经营",
    "Person_Falling": "人员异常倒地", "Abnormal_Gathering": "人员异常聚集",
    "Pouring_Sewage": "污水井倾倒"
}

# --- 辅助函数 ---
def build_category_map(config):
    cat_map = {}
    for event_type, details in config.items():
        for category in details["categories"]:
            cat_map[category] = event_type
    return cat_map

def extract_info_from_filename(file_name, category_map):
    found_tag = None
    # 尝试从括号中提取，如 (Trash_Overflow)
    match = re.search(r'\((.*?)\)', file_name)
    if match:
        found_tag = match.group(1).strip()
        if found_tag in TAG_ALIAS:
            category = TAG_ALIAS[found_tag]
            event_type = category_map.get(category)
            if event_type:
                return found_tag, category, event_type
    
    # 尝试直接匹配关键字
    for key in TAG_ALIAS.keys():
        if key in file_name:
            found_tag = key
            category = TAG_ALIAS[key]
            event_type = category_map.get(category)
            if event_type:
                return found_tag, category, event_type
    return None, None, None

def clean_json_string(text):
    if not isinstance(text, str):
        return str(text)
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

def main():
    cat_map = build_category_map(EVENT_CONFIG)
    output_root = Path(MERGED_OUTPUT_DIR)
    model2_root = Path(MODEL2_ROOT_DIR)
    
    # 用于存储所有数据的字典，Key 为 image_stem (不含后缀的文件名)
    # Value 结构: { "file_name": ..., "event_type": ..., "category": ..., "raw_tag": ..., "m1_data": [], "m2_data": [] }
    merged_registry = {}

    # ==========================================
    # 阶段 1: 处理 Model 1 (JSONL)
    # ==========================================
    jsonl_path = Path(MODEL1_JSONL_PATH)
    if jsonl_path.exists():
        logging.info(f"正在读取 Model 1 JSONL: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Loading Model 1"):
            try:
                record = json.loads(line)
                if not record.get("images"):
                    continue
                
                img_path_str = record["images"][0]["path"]
                img_path = Path(img_path_str)
                file_name = img_path.name
                image_stem = img_path.stem
                
                # 提取分类信息
                raw_tag, category, event_type = extract_info_from_filename(file_name, cat_map)
                if not event_type:
                    continue

                # 提取 Model 1 数据
                model1_data = []
                raw_response = record.get("response", "")
                json_str = clean_json_string(raw_response)
                try:
                    response_dict = json.loads(json_str)
                    if "event_analyses" in response_dict:
                        model1_data = response_dict["event_analyses"]
                except json.JSONDecodeError:
                    pass # 解析失败则保持为空列表

                # 存入注册表
                merged_registry[image_stem] = {
                    "file_name": file_name,
                    "event_type": event_type,
                    "category": category,
                    "raw_tag": raw_tag,
                    "m1_data": model1_data,
                    "m2_data": [] # 暂时为空，稍后填充
                }

            except Exception as e:
                logging.error(f"Error parsing JSONL line: {e}")
    else:
        logging.error("未找到 Model 1 JSONL 文件，将仅处理 Model 2 数据。")

    # ==========================================
    # 阶段 2: 处理 Model 2 (目录扫描)
    # ==========================================
    logging.info(f"正在扫描 Model 2 目录: {model2_root}")
    # 递归遍历 Model 2 文件夹
    model2_files = list(model2_root.rglob("*.json"))
    
    for m2_file in tqdm(model2_files, desc="Loading Model 2"):
        try:
            image_stem = m2_file.stem
            
            # 读取 Model 2 数据
            model2_data = []
            try:
                with open(m2_file, 'r', encoding='utf-8') as f2:
                    m2_content = json.load(f2)
                    model2_data = m2_content.get("analyses", [])
            except Exception as e:
                logging.warning(f"读取 Model 2 文件失败 {m2_file}: {e}")
                continue

            # 逻辑分支：是更新现有记录，还是新增记录？
            if image_stem in merged_registry:
                # Case A: Model 1 已经添加过该图片，只更新 m2 数据
                merged_registry[image_stem]["m2_data"] = model2_data
            else:
                # Case B: Model 1 没有该图片 (新增)
                # 尝试从文件名/父目录推断信息
                # 此时 file_name 可能不带 .jpg/.png 后缀，我们暂时用 stem 或 json 名
                # 为了能在 extract_info 中匹配到 tag，我们需要传入包含 tag 的字符串
                
                # 尝试直接从 stem 解析 tag
                raw_tag, category, event_type = extract_info_from_filename(image_stem, cat_map)
                
                # 如果从文件名解析失败，且父目录是已知的 EventType，可以尝试反推 (可选优化)
                if not event_type:
                    parent_dir_name = m2_file.parent.name
                    if parent_dir_name in EVENT_CONFIG:
                        event_type = parent_dir_name
                        category = "Unknown_Category_From_Dir" # 无法精确得知子类
                        raw_tag = "Unknown"
                
                if event_type:
                    merged_registry[image_stem] = {
                        "file_name": f"{image_stem}.json", # 原始图片名未知，暂用json名代替
                        "event_type": event_type,
                        "category": category,
                        "raw_tag": raw_tag,
                        "m1_data": [], # Model 1 无数据
                        "m2_data": model2_data
                    }
        except Exception as e:
            logging.error(f"Error processing Model 2 file {m2_file}: {e}")

    # ==========================================
    # 阶段 3: 写入最终合并结果
    # ==========================================
    logging.info(f"开始写入合并结果，共 {len(merged_registry)} 条唯一数据...")
    processed_count = 0
    
    for image_stem, data in tqdm(merged_registry.items(), desc="Writing"):
        try:
            event_type = data["event_type"]
            
            final_output = {
                "file_name": data["file_name"],
                "source_event_type": event_type,
                "source_category": data["category"],
                "source_tag_in_filename": data["raw_tag"],
                
                # 合并数据
                "model2_qwen235b_analyses": data["m2_data"],
                "model1_finetuned_event_analyses": data["m1_data"]
            }
            
            # 写入
            target_dir = output_root / event_type
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"{image_stem}.json"
            
            with open(target_file, 'w', encoding='utf-8') as f_out:
                json.dump(final_output, f_out, ensure_ascii=False, indent=2)
            
            processed_count += 1
            
        except Exception as e:
            logging.error(f"写入失败 {image_stem}: {e}")

    logging.info("="*30)
    logging.info(f"处理完成！")
    logging.info(f"成功合并并生成文件: {processed_count} 个")
    logging.info(f"结果保存在: {MERGED_OUTPUT_DIR}")

if __name__ == "__main__":
    main()