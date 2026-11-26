import json
from pathlib import Path
import re
import argparse

def diagnose_prompts(file_path: Path):
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return

    print(f"\n--- 正在诊断文件: {file_path.name} ---")
    
    unmatched_prompts = []
    pattern = re.compile(r'与“([^”]+)”相关的情况')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                user_prompt = data['messages'][0]['content']
                if not pattern.search(user_prompt):
                    unmatched_prompts.append((i, user_prompt))
            except (json.JSONDecodeError, KeyError, IndexError):
                print(f"  - 第 {i} 行: 格式错误或无法解析")

    if unmatched_prompts:
        print(f"发现 {len(unmatched_prompts)} 条不匹配的prompt:")
        # 只打印前5个作为示例
        for line_num, prompt in unmatched_prompts[:5]:
            print(f"  - 第 {line_num} 行: {prompt[:100]}...") # 打印前100个字符
    else:
        print("所有prompt格式均符合预期！")

def main():
    parser = argparse.ArgumentParser(description="诊断V1数据集中不符合格式的prompt。")
    parser.add_argument("--data_v1_dir", type=Path, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述", help="第一批jsonl数据所在的目录。")
     
    args = parser.parse_args()

    files_to_check = [
        "train_zhengyb_converted_longprompt.jsonl",
        "train_fuyb_converted_longprompt.jsonl",
        "val_zhengyb_converted_longprompt.jsonl",
        "val_fuyb_converted_longprompt.jsonl",
    ]

    for filename in files_to_check:
        diagnose_prompts(args.data_v1_dir / filename)

if __name__ == "__main__":
    main()