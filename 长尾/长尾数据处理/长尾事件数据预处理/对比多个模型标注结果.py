import os
import json
import shutil
from pathlib import Path

# ================= 配置路径 (请修改这里) =================

# 1. 原图目录 (用于遍历所有文件名)
IMAGE_ROOT_DIR = '/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/待标注'

# 2. Model 1 结果目录 (每个图片一个JSON)
MODEL1_DIR = '/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/待标注_qwen3vl235B'

# 3. Model 2 结果文件 (大JSON/JSONL文件路径)
# 请替换为实际的 model2 json文件路径
MODEL2_FILE_PATH = '/host/ssddata/disk2/zhanglu/swift/4cls/result/train_260210p0v2p1v3_epochs20/result_0227_biaozhu_ck9900.json' 

# 4. Model 3 结果文件 (大JSON/JSONL文件路径)
# 请替换为实际的 model3 json文件路径
MODEL3_FILE_PATH = '/host/ssddata/disk2/zhanglu/swift/4cls/result/train_260210p0v2p1v3_epochs20/result_0227_biaozhu_nofinetune.json'

# 5. 输出目录
OUTPUT_DIR = '/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/comparison_results'


# =======================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_relative_info(full_path, root_path):
    """
    获取相对路径信息。
    返回: (rel_path, subdir, filename_stem)
    """
    abs_full = os.path.abspath(full_path)
    abs_root = os.path.abspath(root_path)
    
    if abs_full.startswith(abs_root):
        rel_path = abs_full[len(abs_root):].lstrip(os.sep)
    else:
        parts = full_path.split(os.sep)
        if len(parts) >= 2:
            rel_path = os.path.join(parts[-2], parts[-1])
        else:
            rel_path = os.path.basename(full_path)

    subdir = os.path.dirname(rel_path)
    filename = os.path.basename(rel_path)
    filename_stem = os.path.splitext(filename)[0]
    
    return rel_path, subdir, filename_stem

def find_key_recursive(data, target_key):
    """
    递归查找 JSON 中的某个 Key，返回其 Value。
    解决模型输出结构嵌套错乱的问题。
    """
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for key, value in data.items():
            res = find_key_recursive(value, target_key)
            if res is not None:
                return res
    elif isinstance(data, list):
        for item in data:
            res = find_key_recursive(item, target_key)
            if res is not None:
                return res
    return None

def extract_anomaly_type(json_data):
    """
    鲁棒性更强的提取 '异常类型'
    """
    if not isinstance(json_data, dict):
        return None

    try:
        # 1. 定位到 event_analyses 列表
        event_list = []
        if "analyses" in json_data and isinstance(json_data["analyses"], list) and json_data["analyses"]:
            # Model 1 结构
            event_list = json_data["analyses"][0].get("event_analyses", [])
        elif "event_analyses" in json_data:
            # Model 2/3 结构
            event_list = json_data.get("event_analyses", [])
        
        if not isinstance(event_list, list) or len(event_list) == 0:
            return None

        # 2. 取第一个事件对象
        first_event = event_list[0]

        # 3. 递归查找 'comprehensive_assessment' 对象
        # 无论它是在顶层，还是在 detailed_cot 里，都能找到
        comp_assessment = find_key_recursive(first_event, "comprehensive_assessment")

        # 4. 提取 '异常类型'
        if comp_assessment and isinstance(comp_assessment, dict):
            return comp_assessment.get("异常类型", None)
            
    except Exception:
        return None
    
    return None

def load_large_json_results(file_path):
    results_map = {}
    if not os.path.exists(file_path):
        print(f"[Warning] 文件不存在: {file_path}")
        return results_map

    print(f"正在加载大文件: {file_path} ...")
    
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                lines = json.loads(content)
            else:
                lines = content.split('\n')
    except Exception as e:
        print(f"读取文件出错: {e}")
        return results_map

    for item in lines:
        raw_obj = {}
        if isinstance(item, str):
            if not item.strip(): continue
            try:
                raw_obj = json.loads(item)
            except: continue
        else:
            raw_obj = item

        img_path = raw_obj.get("images", [{}])[0].get("path", "")
        if not img_path: continue
        
        rel_key, _, _ = get_relative_info(img_path, IMAGE_ROOT_DIR)
        
        response_str = raw_obj.get("response", "{}")
        try:
            parsed_response = json.loads(response_str)
        except:
            parsed_response = {"error": "Failed to parse response string"}

        anomaly_type = extract_anomaly_type(parsed_response)
        
        results_map[rel_key] = {
            "content": parsed_response,
            "type": anomaly_type
        }

    print(f"加载完成，共 {len(results_map)} 条数据。")
    return results_map

def save_json(output_root, category, sub_dir, file_stem, model_suffix, content):
    """
    保存格式: output_root/category/sub_dir/file_stem_model_suffix.json
    """
    if content is None:
        return

    target_dir = os.path.join(output_root, category, sub_dir)
    ensure_dir(target_dir)
    
    filename = f"{file_stem}_{model_suffix}.json"
    save_path = os.path.join(target_dir, filename)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

def main():
    model2_data = load_large_json_results(MODEL2_FILE_PATH)
    model3_data = load_large_json_results(MODEL3_FILE_PATH)

    print(f"开始遍历原图目录: {IMAGE_ROOT_DIR}")
    
    count_total = 0
    count_consistent = 0
    count_inconsistent = 0

    for root, dirs, files in os.walk(IMAGE_ROOT_DIR):
        for file in files:
            if not file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue
            
            count_total += 1
            abs_path = os.path.join(root, file)
            
            # rel_path: '垃圾堆积/xxx.jpg', sub_dir: '垃圾堆积', file_stem: 'xxx'
            rel_path, sub_dir, file_stem = get_relative_info(abs_path, IMAGE_ROOT_DIR)
            
            # --- Model 1 ---
            m1_path = os.path.join(MODEL1_DIR, sub_dir, file_stem + ".json")
            m1_content = None
            m1_type = None
            if os.path.exists(m1_path):
                try:
                    with open(m1_path, 'r', encoding='utf-8') as f:
                        m1_content = json.load(f)
                    m1_type = extract_anomaly_type(m1_content)
                except: pass
            
            # --- Model 2 ---
            m2_entry = model2_data.get(rel_path)
            m2_content = m2_entry['content'] if m2_entry else None
            m2_type = m2_entry['type'] if m2_entry else None

            # --- Model 3 ---
            m3_entry = model3_data.get(rel_path)
            m3_content = m3_entry['content'] if m3_entry else None
            m3_type = m3_entry['type'] if m3_entry else None

            # --- 对比逻辑 ---
            is_consistent = False
            # 必须都不为 None，且完全相等
            if (m1_type is not None) and (m2_type is not None) and (m3_type is not None):
                if m1_type.strip() == m2_type.strip() == m3_type.strip():
                    is_consistent = True
            
            # --- 保存 ---
            category = "一致" if is_consistent else "不一致"
            
            if is_consistent:
                count_consistent += 1
            else:
                count_inconsistent += 1
                # 调试打印 (可选)
                # print(f"不一致: {file_stem} -> M1:{m1_type}, M2:{m2_type}, M3:{m3_type}")

            if m1_content: save_json(OUTPUT_DIR, category, sub_dir, file_stem, "model1", m1_content)
            if m2_content: save_json(OUTPUT_DIR, category, sub_dir, file_stem, "model2", m2_content)
            if m3_content: save_json(OUTPUT_DIR, category, sub_dir, file_stem, "model3", m3_content)

            if count_total % 200 == 0:
                print(f"进度: {count_total} | 一致: {count_consistent} | 不一致: {count_inconsistent}")

    print("="*40)
    print(f"处理完成 | 总数: {count_total}")
    print(f"一致: {count_consistent}")
    print(f"不一致: {count_inconsistent}")
    print(f"结果保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()