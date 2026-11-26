import json
import re

# 修复核心：转义content字段内的双引号，保留JSON结构
input_path = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述/val_zhengyb.jsonl"
output_path = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述/val_zhengyb_xf.jsonl"

# 匹配content字段内的双引号（非JSON结构的双引号）
def fix_content_quotes(line):
    # 步骤1：先匹配content字段的值（捕获引号内的内容）
    # 正则匹配 "content": "..." 中的内容部分
    pattern = r'"content":\s*"([^"]+)"'
    def replace_quotes(match):
        content = match.group(1)
        # 转义内容内的双引号为 \"
        content_fixed = content.replace('"', '\\"')
        return f'"content": "{content_fixed}"'
    # 替换所有content字段内的未转义双引号
    line_fixed = re.sub(pattern, replace_quotes, line)
    return line_fixed

with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
    valid_lines = 0
    for idx, line in enumerate(f_in):
        line = line.strip()
        if not line:
            continue
        try:
            # 第一步：修复content内的双引号
            line_fixed = fix_content_quotes(line)
            # 第二步：解析验证（确保修复后合法）
            data = json.loads(line_fixed)
            # 第三步：重新序列化（规范格式）
            json.dump(data, f_out, ensure_ascii=False)
            f_out.write("\n")
            valid_lines += 1
        except json.JSONDecodeError as e:
            print(f"第{idx+1}行仍有错误（跳过）：{e}，原始内容：{line[:100]}")
            continue
print(f"修复完成！有效样本数：{valid_lines}，输出文件：{output_path}")
