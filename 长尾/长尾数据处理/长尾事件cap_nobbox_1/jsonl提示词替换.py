import json
import os

def update_prompts_from_md(input_jsonl, output_jsonl, md_dir="./"):
    """
    input_jsonl: 原始jsonl文件路径
    output_jsonl: 修改后的jsonl保存路径
    md_dir: 存放 .md 文件的目录，默认当前目录
    """
    
    # 1. 定义事件名称与对应文件名的映射
    event_mapping = {
        "人员异常倒地": "cap_倒地事件.md",
        "非机动车乱停放": "cap_非机动车乱停放.md",
        "垃圾事件": "cap_垃圾事件.md",
        "人群异常聚集": "cap_人群异常聚集.md",
        "雨水井倾倒污水": "cap_雨水井倾倒污水.md",
        "森林火灾": "cap_森林火灾.md",
        "占道经营事件": "cap_占道经营事件.md"
    }

    # 2. 预加载所有 Markdown 文件的内容
    md_content_cache = {}
    for event_name, file_name in event_mapping.items():
        file_path = os.path.join(md_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content_cache[event_name] = f.read().strip()
            print(f"已加载事件库: {event_name} <- {file_name}")
        else:
            print(f"⚠️ 警告: 找不到文件 {file_path}，该事件将跳过替换。")

    # 3. 处理 JSONL 文件
    count = 0
    with open(input_jsonl, 'r', encoding='utf-8') as f_in, \
         open(output_jsonl, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            # 假设 messages[0] 是 user，messages[1] 是 assistant
            user_message = data["messages"][0]["content"]
            
            # 检查 user 消息中包含哪个关键词
            replaced = False
            for event_name, new_prompt in md_content_cache.items():
                if event_name in user_message:
                    data["messages"][0]["content"] = new_prompt
                    replaced = True
                    count += 1
                    break # 匹配到一个就跳出循环
            
            # 写入新文件
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"✨ 处理完成！共替换了 {count} 条数据。")
    print(f"📂 结果已保存至: {output_jsonl}")

# --- 执行设置 ---
input_path = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述/train_zhengyb_converted.jsonl"    # 你原始的那个报错的jsonl
output_path = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述/train_zhengyb_converted_longprompt.jsonl" # 修改后的文件名
markdown_folder = "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox/prompts_训练"     # 如果你的md文件在prompts文件夹里，请修改这里

update_prompts_from_md(input_path, output_path, md_dir=markdown_folder)
