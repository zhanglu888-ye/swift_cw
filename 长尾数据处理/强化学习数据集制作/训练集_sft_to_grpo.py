import json
import os

# ================= 配置区域 =================
# 你的原始 SFT 数据集路径
INPUT_FILE = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/sft_val.jsonl" 
# 转换后的 GRPO 数据集输出路径
OUTPUT_FILE = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json/grpo_val.jsonl"
# ===========================================

def process_sft_to_grpo(input_path, output_path):
    processed_count = 0
    error_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_idx, line in enumerate(fin):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                
                images = data.get("images",[])
                messages = data.get("messages",[])
                
                user_content = ""
                assistant_content = ""
                
                # 1. 提取 User 和 Assistant 的内容
                for msg in messages:
                    if msg["role"] == "user":
                        user_content = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_content = msg["content"]
                
                if not user_content or not assistant_content:
                    print(f"第 {line_idx+1} 行缺少 user 或 assistant 内容，跳过。")
                    error_count += 1
                    continue
                
                # 2. 从 Assistant 的 JSON 中提取核心的 Ground Truth (异常类型 & 风险等级)
                # 这样可以极大简化 Reward Function 的编写
                try:
                    ast_dict = json.loads(assistant_content)
                    # 按照你的数据结构，提取 comprehensive_assessment
                    assessment = ast_dict.get("event_analyses", [{}])[0].get("comprehensive_assessment", {})
                    # 转成 JSON 字符串，存入 solution
                    solution_str = json.dumps(assessment, ensure_ascii=False)
                except json.JSONDecodeError:
                    print(f"第 {line_idx+1} 行 Assistant 内容不是合法的 JSON，直接存储原文本。")
                    solution_str = assistant_content
                except Exception as e:
                    print(f"第 {line_idx+1} 行提取核心字段失败: {e}，直接存储原文本。")
                    solution_str = assistant_content

                # 3. 构建 GRPO 所需的数据格式
                # 注意：messages 里只保留 user，让模型自己生成 assistant 的内容
                grpo_item = {
                    "images": images,
                    "messages":[
                        {"role": "user", "content": user_content}
                    ],
                    "solution": solution_str  # 这个字段会透传给 Reward Function
                }
                
                # 写入文件
                fout.write(json.dumps(grpo_item, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except Exception as e:
                print(f"处理第 {line_idx+1} 行时发生未知错误: {e}")
                error_count += 1

    print("="*40)
    print(f"转换完成！")
    print(f"成功转换: {processed_count} 条")
    print(f"失败跳过: {error_count} 条")
    print(f"GRPO 数据已保存至: {output_path}")
    print("="*40)

if __name__ == "__main__":
    process_sft_to_grpo(INPUT_FILE, OUTPUT_FILE)