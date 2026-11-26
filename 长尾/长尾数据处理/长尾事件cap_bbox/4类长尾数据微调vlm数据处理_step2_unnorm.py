import json
import sys
import os
import re # 导入正则表达式模块

def to_samples(item):
    img = item["images"][0]

    # 1. 获取原始的用户和助手内容
    try:
        user_prompt = next(m["content"] for m in item["messages"] if m["role"] == "user")
        assistant_content_str = next(m["content"] for m in item["messages"] if m["role"] == "assistant")
    except StopIteration:
        print(f"警告：跳过一个缺少 'user' 或 'assistant' role 的条目。")
        return []

    # 2. 解析 assistant.content 以提取 bbox
    j = json.loads(assistant_content_str)
    boxes = []
    for ea in j.get("event_analyses", []):
        b = ea.get("bbox")
        # 确保bbox是有效的列表
        if isinstance(b, list) and len(b) == 4:
            # 将可能存在的浮点数坐标转换为整数
            boxes.append([int(coord) for coord in b])

    # 3. 从用户prompt中动态提取事件类型
    event_type = "目标"  # 如果找不到，使用一个通用的默认值
    match = re.search(r'与“(.+?)”相关', user_prompt)
    if match:
        event_type = match.group(1)
    else:
        print(f"警告：无法从prompt中提取事件类型，将使用默认值'目标'。 Prompt: {user_prompt[:50]}...")

    # 4. 构建 ① 检测样本 (det)
    # 使用动态生成的、标准化的检测prompt
    det_user_prompt = f"<image>请在图中定位与“{event_type}”相关的目标。"
    det = {
      "messages":[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content": det_user_prompt},
        {"role":"assistant","content":"".join(["<bbox>" for _ in boxes])}
      ],
      "images":[img],
      "objects":{"ref":[], "bbox":boxes, "bbox_type":"real", "image_id":0},
      "channel":"det"
    }

    # 5. 清理 assistant 内容中的 bbox，为描述样本做准备
    for ea in j.get("event_analyses", []): 
        ea.pop("bbox", None)

    # 6. 构建 ② 描述样本 (desc)
    # 直接复用原始的、完整的用户prompt
    desc = {
      "messages":[
        {"role":"user", "content": user_prompt},
        {"role":"assistant", "content":json.dumps(j, ensure_ascii=False)}
      ],
      "images":[img],
      "channel":"desc"
    }
    
    return [det, desc]

def main(in_path, out_json):
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {in_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：输入文件 {in_path} 不是有效的JSON格式。")
        return

    all_samples = []
    for item in data:
        all_samples.extend(to_samples(item))
        
    # 输出为“单个 JSON 数组”
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共生成 {len(all_samples)} 条样本，已保存至 {out_json}")

if __name__=="__main__":
    # 用法：python convert_to_array.py input.json output.json
    ##.  python 4类长尾数据微调vlm数据处理_step2_unnorm.py /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified.json /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified_unnormalized.json 
    main(sys.argv[1], sys.argv[2])