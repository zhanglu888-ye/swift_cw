import json
with open('/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述/train_zhengyb.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        try:
            json.loads(line.strip())
        except json.JSONDecodeError as e:
            print(f'第{idx+1}行语法错误：{e}')
