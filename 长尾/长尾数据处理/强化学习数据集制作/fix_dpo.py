import json

def convert_to_strict_format(input_file, output_file):
    count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip(): continue
            data = json.loads(line)
            
            # 1. 提取图片
            images = data.get("images",[])
            
            # 2. 提取原始 messages (通常只包含 user 的 prompt)
            messages = data.get("messages",[])
            
            # 3. 提取 chosen (正确的回复)
            chosen = data.get("chosen")
            if not chosen: continue
            
            chosen_content = ""
            if isinstance(chosen, dict):
                chosen_content = chosen.get("content", "")
            elif isinstance(chosen, list):
                chosen_content = chosen[0].get("content", "")
            else:
                chosen_content = str(chosen)
                
            # 4. 提取 rejected (错误的回复)
            # 注意：为了绝对稳定，这里我们丢弃没有 rejected 的 SFT 数据，只用纯 DPO 对子
            rejected = data.get("rejected")
            if not rejected: continue
            
            rejected_content = ""
            if isinstance(rejected, dict):
                rejected_content = rejected.get("content", "")
            elif isinstance(rejected, list):
                rejected_content = rejected[0].get("content", "")
            else:
                rejected_content = str(rejected)
                
            if not chosen_content or not rejected_content:
                continue

            # ==========================================
            # 5. 严格构建官方格式
            # ==========================================
            
            # A. messages 列表的最后一项必须是 assistant 的正确回复
            strict_messages =[]
            for m in messages:
                strict_messages.append(m)
                
            strict_messages.append({
                "role": "assistant",
                "content": chosen_content
            })
            
            # B. 组装最终字典
            strict_data = {
                "images": images,
                "messages": strict_messages,
                "rejected_response": rejected_content # 必须是纯字符串
            }
            
            fout.write(json.dumps(strict_data, ensure_ascii=False) + '\n')
            count += 1
            
    print(f"转换完成！共保留 {count} 条严格格式数据 -> {output_file}")

# ================= 执行转换 =================

base_dir = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/train_json"

# 转换训练集 (读取原始的 mixed 数据)
convert_to_strict_format(
    f"{base_dir}/dpo_train_mixed_strict_reject2.jsonl", 
    f"{base_dir}/dpo_train_mixed_strict_reject2_new.jsonl"
)

# 转换验证集
convert_to_strict_format(
    f"{base_dir}/dpo_val_strict_reject2.jsonl", 
    f"{base_dir}/dpo_val_strict_reject2_new.jsonl"
)