import json
import random

# 简单版本
def simple_split_json(input_file, train_ratio=0.9):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机打乱
    random.shuffle(data)
    
    # 分割
    split_idx = int(len(data) * train_ratio)
    train = data[:split_idx]
    test = data[split_idx:]
    
    # 保存
    with open('/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述/train_fuyb.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    
    with open('/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述/val_fuyb.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=2)
    
    print(f"完成! 训练集: {len(train)}, 测试集: {len(test)}")

# 使用简单版本
simple_split_json('/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述/all_fuyb.json')
