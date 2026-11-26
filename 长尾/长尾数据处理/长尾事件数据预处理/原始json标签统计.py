import json
from collections import Counter

def simple_count(json_file):
    """简化版统计函数"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    counts = Counter()
    
    for item in data:
        sample_type = item['result']['info'][0]['value']
        counts[sample_type] += 1
    
    print("样本分布:")
    for sample_type, count in counts.most_common():
        print(f"  {sample_type}: {count}个")
    
    print(f"\n总计: {sum(counts.values())}个样本")
    return counts

# 使用简化版
if __name__ == "__main__":
    result = simple_count("/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/原始标注结果/占道经营/下载结果_json_351_more_20251127100127.json")
