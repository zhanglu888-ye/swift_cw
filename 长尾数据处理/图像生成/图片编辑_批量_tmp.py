import os
import base64
import requests
import random
import time
import json
import re
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
from openai import OpenAI # 使用 OpenAI SDK
from openai import APIStatusError, APIConnectionError # 导入异常类型

# 屏蔽 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 配置区域 =================

# 注意：OpenAI SDK 需要 base_url 以 /v1/ 结尾
API_BASE_URL = "https://arms-referring-matthew-appendix.trycloudflare.com/v1/"
API_KEY = "McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ"
PROXY_SECRET = "my_secret_password_sensoro_123"

# 模型名称
VLM_MODEL = "Qwen3-VL-235B-A22B-Instruct"
EDIT_MODEL = "Qwen-Image-Edit-2511"


# 2. 路径配置
INPUT_ROOT = "/host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/原始图片_大马路"
OUTPUT_ROOT = "/host/ssddata/disk3/zhanglu/dataset/changwei/20260113_生成/生成_results_0128_p1_渣土车遗洒" 


HISTORY_DIRS = [
    OUTPUT_ROOT, # 包含本次输出目录
]

# 提示词文件路径 (修改这里来切换 V1 或 V2)
PROMPTS_FILE = "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/图像生成/prompts_渣土车遗洒.md"

# 3. 时间配置
NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 3

# ================= 核心工具函数 =================

def parse_prompts_md(md_file_path):
    prompts_db = {}
    difficulty_modifiers = []
    weather_modifiers = [] 
    global_suffix = ""
    current_category = None

    if not os.path.exists(md_file_path):
        raise FileNotFoundError(f"提示词文件未找到: {md_file_path}")

    with open(md_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("# "): continue 
        
        if line.startswith("##"):
            raw_key = line.replace("##", "").strip()
            current_category = raw_key.split('(')[0].strip()
            if current_category not in ["Global_Suffix", "Difficulty_Modifiers", "Weather_Modifiers"] and current_category not in prompts_db:
                prompts_db[current_category] = []
        elif line.startswith("-") and current_category:
            content = line[1:].strip()
            if current_category == "Global_Suffix":
                if not global_suffix: global_suffix = content
            elif current_category == "Difficulty_Modifiers":
                difficulty_modifiers.append(content)
            elif current_category == "Weather_Modifiers":
                weather_modifiers.append(content)
            else:
                prompts_db[current_category].append(content)

    return prompts_db, difficulty_modifiers, weather_modifiers, global_suffix

def extract_hour_from_filename(filename):
    match = re.search(r'_(\d{8})_(\d{6})_', filename)
    if match:
        time_str = match.group(2)
        try:
            return int(time_str[:2])
        except ValueError:
            return None
    return None

def get_time_context_prompt(hour):
    if hour is None: return ""
    if 5 <= hour < 9: return ", 在清晨柔和的光线下"
    elif 9 <= hour < 16: return ", 在正午或白天明亮的光线下"
    elif 16 <= hour < 19: return ", 在黄昏或傍晚时分"
    elif 19 <= hour <= 23 or 0 <= hour < 5: return ", 在夜晚，依靠月光或微弱的人造光照明" # 针对森林调整夜晚描述
    return ""

def get_weather_context_prompt(weather_modifiers):
    if weather_modifiers and random.random() < 0.6:
        return ", " + random.choice(weather_modifiers)
    return ""

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_result(image_bytes, json_data, output_dir, file_prefix):
    image_path = os.path.join(output_dir, f"{file_prefix}.png")
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    json_path = os.path.join(output_dir, f"{file_prefix}.json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

# ================= VLM 识别逻辑 (OpenAI SDK版) - 已移除 =================
# def analyze_image_with_vlm(client, base64_image):
#     # 此函数已移除，不再需要VLM判断
#     pass

# ================= 图片生成逻辑 (OpenAI SDK版) =================

def call_edit_api(client, base64_image, final_prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    
    for attempt in range(3):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=EDIT_MODEL,
                messages=messages,
                timeout=180 
            )
            
            msg = response.choices[0].message.content
            
            try:
                parsed_msg = json.loads(msg)
                if isinstance(parsed_msg, list) and parsed_msg[0].get('type') == 'image_url':
                    img_url = parsed_msg[0]['image_url']['url']
                    _, encoded = img_url.split(",", 1)
                    return base64.b64decode(encoded), round(time.time() - start_time, 2)
            except:
                pass # Try regex fallback if JSON parsing fails
            
            if "data:image" in str(msg):
                match = re.search(r'data:image\/.*?;base64,(.*)', str(msg))
                if match:
                    return base64.b64decode(match.group(1)), round(time.time() - start_time, 2)

            return None, 0
            
        except (APIStatusError, APIConnectionError) as e:
            print(f"🔄 Edit API Error (Attempt {attempt+1}): {e}")
            time.sleep(15)
        except Exception as e:
            print(f"❌ Edit Unknown Error: {e}")
            time.sleep(5)
            
    return None, 0

def process_single_image(image_path, original_stem, prompts_db, difficulty_modifiers, weather_modifiers, global_suffix):
    
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        default_headers={"x-proxy-secret": PROXY_SECRET},
        timeout=120.0,
        max_retries=1 
    )

    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        print(f"❌ [{original_stem}] 图像编码失败: {e}")
        return

    # 1. VLM 识别逻辑已移除。直接假定场景为森林。
    # scene_label = "🌳 背景：森林"
    print(f"👁️ [{original_stem}] 背景：森林")

    available_event_types = list(prompts_db.keys())

    if not available_event_types:
        print(f"⚠️ [{original_stem}] 提示词库中无可用事件类型")
        return

    hour = extract_hour_from_filename(original_stem)
    time_prompt = get_time_context_prompt(hour)

    is_night_mode = False
    if hour is not None:
        is_night_mode = (hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR)
    
    # 森林场景，通常白天和夜晚事件数量可以保持一致，或者根据需要调整
    # 默认为1次生成，夜间模式可增加生成次数
    num_generations = 3 if is_night_mode else 1 
        
    # 从所有可用事件类型中随机选取，确保不重复
    selected_events = random.sample(available_event_types, min(num_generations, len(available_event_types)))

    for i, event_type in enumerate(selected_events):
        raw_prompt = random.choice(prompts_db[event_type])
        
        difficulty_text = ""
        if difficulty_modifiers and random.random() < 0.7: 
            difficulty_text = ", " + random.choice(difficulty_modifiers)
            
        weather_text = get_weather_context_prompt(weather_modifiers) # 始终适用，因为是室外场景

        final_prompt = f"{raw_prompt}{time_prompt}{weather_text}{difficulty_text}{global_suffix}"
        
        print(f"   👉 ({i+1}/{num_generations}) 生成: {event_type}")

        # 调用 Edit API
        image_bytes, cost_time = call_edit_api(client, base64_image, final_prompt)
        
        if image_bytes:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_prefix = f"{original_stem}_{event_type}_{i+1}_{timestamp_str}"
            
            trace_data = {
                "original_filename": original_stem,
                # "vlm_info": info, # VLM 信息已移除
                "generation_info": {
                    "event_type": event_type,
                    "prompt_used": final_prompt,
                    "cost_time": cost_time,
                    "is_night_mode": is_night_mode,
                    "background_assumed": "forest" # 明确标记背景假设
                }
            }
            save_result(image_bytes, trace_data, OUTPUT_ROOT, file_prefix)
            print(f"   ✅ 完成: {file_prefix}.png")
        else:
            print(f"   ❌ 生成失败")

# ================= 主程序 =================

def main():
    print("🚀 初始化脚本 (森林背景特化版)...")
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)

    try:
        prompts_db, diff_mods, weather_mods, global_suffix = parse_prompts_md(PROMPTS_FILE)
        print(f"📖 提示词库加载: {len(prompts_db)} 类事件")
    except Exception as e:
        print(f"❌ 提示词错误: {e}")
        return

    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 输入路径不存在: {INPUT_ROOT}")
        return

    processed_stems = set()
    print(f"🔍 扫描历史记录...")
    for h_dir in HISTORY_DIRS:
        if not os.path.exists(h_dir): continue
        for root, _, files in os.walk(h_dir):
            for fname in files:
                if fname.endswith('.json'):
                    try:
                        with open(os.path.join(root, fname), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if "original_filename" in data: processed_stems.add(data["original_filename"])
                    except: continue
    print(f"⏭️  已完成: {len(processed_stems)}")

    files_to_process = []
    print(f"🔍 扫描图片...")
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                stem = Path(file).stem
                if stem in processed_stems: continue
                files_to_process.append({"path": os.path.join(root, file), "stem": stem})

    total = len(files_to_process)
    if total == 0:
        print("⚠️ 无待处理图片。")
        return

    print(f"📂 本次待处理: {total} 张")
    
    # 建议先用 1 线程测试
    MAX_WORKERS = 4
    print(f"🔥 启动线程池: {MAX_WORKERS} 并发")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(
                process_single_image, 
                item['path'], 
                item['stem'], 
                prompts_db, 
                diff_mods, 
                weather_mods, 
                global_suffix
            ): item['stem'] 
            for item in files_to_process
        }
        for future in as_completed(future_to_file):
            try: future.result()
            except Exception as e:
                print(f"❌ 处理文件 {future_to_file[future]} 发生异常: {e}")


    print("✅ 全部完成！")

if __name__ == "__main__":
    main()