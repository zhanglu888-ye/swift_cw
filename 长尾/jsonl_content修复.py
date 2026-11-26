import json

def convert_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            
            # 解析原始 JSON
            data = json.loads(line)
            
            # 找到 assistant 的消息内容
            for message in data.get("messages", []):
                if message.get("role") == "assistant":
                    content = message.get("content")
                    
                    # 如果 content 是字典类型，将其转换为格式化的 JSON 字符串
                    if isinstance(content, dict):
                        # ensure_ascii=False 保证中文不被编码，indent=2 保持可读性（可选）
                        # 如果你希望压缩格式，可以去掉 indent
                        message["content"] = json.dumps(content, ensure_ascii=False, indent=2)
            
            # 将处理后的结果写入新文件
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

# 使用方法
input_path = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述/val_zhengyb.jsonl"  # 你的第二个文件名
output_path = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述/val_zhengyb_converted.jsonl"           # 转换后的文件名
convert_jsonl(input_path, output_path)
print(f"转换完成！已保存至: {output_path}")
