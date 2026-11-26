import json
import os
from pathlib import Path

def clean_data(json_dir):
    """
    遍历目录，删除指定的冗余字段并统一格式
    """
    for file_path in Path(json_dir).glob("*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 递归清理字典或列表中的冗余字段
        def recursive_clean(obj):
            if isinstance(obj, dict):
                keys_to_remove = ["source_event_type", "source_category", "source_image_path"]
                for key in keys_to_remove:
                    obj.pop(key, None)
                for value in obj.values():
                    recursive_clean(value)
            elif isinstance(obj, list):
                for item in obj:
                    recursive_clean(item)
        
        recursive_clean(data)
        
        # 保存回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print("清洗完成！")

if __name__ == "__main__":
    clean_data("/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/训练/人员倒地都对_labels")