import json
import os

def convert_test_set(input_jsonl, output_jsonl, md_dir="./prompts"):
    # 1. 定义旧 Prompt 关键词与内部类别的映射
    # 根据你提供的描述段落提取核心特征词
    prompt_to_category = {
        "商品售卖的情况": "流动摊贩",
        "超出了店铺门口台阶": "出店经营",
        "未按规定停在黄色停车线框内": "非机动车乱停放",
        "往雨水井中倾倒": "雨水井倾倒污水",
        "垃圾桶外地面有较大量的垃圾": "垃圾桶满溢",
        "建筑材料/垃圾、生活垃圾、废弃家具": "垃圾堆积",
        "人员因意外、疾病、外力等原因": "人员异常倒地",
        "人群密度是否异常增高": "人群异常聚集"
    }

    # 2. 定义类别对应的 MD 文件名（根据你的要求）
    category_to_md = {
        "流动摊贩": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox/prompts_训练/cap_占道经营事件.md",
        "出店经营": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox/prompts_训练/cap_占道经营事件.md", # 对应你之前的“占道经营事件.md”
        "非机动车乱停放": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox/prompts_训练/cap_非机动车乱停放.md",
        "雨水井倾倒污水": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox/prompts_训练/cap_雨水井倾倒污水.md",
        "垃圾桶满溢": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox/prompts_训练/cap_垃圾事件.md", # 假设你也有对应的md
        "垃圾堆积": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox/prompts_训练/cap_垃圾事件.md",
        "人员异常倒地": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox/prompts_训练/cap_倒地事件.md",
        "人群异常聚集": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox/prompts_训练/cap_人群异常聚集.md"
    }

    # 3. 预加载所有 MD 内容
    md_content_cache = {}
    for cat, filename in category_to_md.items():
        path = os.path.join(md_dir, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                md_content_cache[cat] = f.read().strip()
        else:
            print(f"⚠️ 警告: 找不到 {path}")

    def generate_dummy_json(category, is_positive):
        """生成符合格式的 Assistant JSON 字符串"""
        if not is_positive:
            res = {
                "event_analyses": [{
                    "simplified_cot": {"画面分析": "环境描述：场景正常。\n事件描述：未发现异常行为。", "推理过程": "定性分析：该场景符合合规标准。\n风险评估：无安全隐患。"},
                    "detailed_cot": {"画面分析": "环境描述：整体环境整洁有序。\n事件描述：视野内未发现任何违规或异常情况。", "推理过程": "定性分析：经过细致排查，所有要素均在正常范围内。\n风险评估：判定为无风险。"},
                    "comprehensive_assessment": {"异常类型": "无异常", "风险等级": "0级 - 无风险"}
                }]
            }
        else:
            res = {
                "event_analyses": [{
                    "simplified_cot": {"画面分析": f"环境描述：检测到相关场景。\n事件描述：发现{category}现象。", "推理过程": f"定性分析：符合{category}判定标准。\n风险评估：存在一定风险。"},
                    "detailed_cot": {"画面分析": "环境描述：[待模型推理详情]\n事件描述：[待模型推理详情]", "推理过程": f"定性分析：依据特征识别为{category}。\n风险评估：需要注意并进行相应处理。"},
                    "comprehensive_assessment": {"异常类型": category, "风险等级": "2级 - 中风险"}
                }]
            }
        return json.dumps(res, ensure_ascii=False)

    # 4. 开始转换
    converted_count = 0
    with open(input_jsonl, 'r', encoding='utf-8') as f_in, \
         open(output_jsonl, 'w', encoding='utf-8') as f_out:
        
        # 兼容处理：如果输入是标准的JSON数组或JSONL
        try:
            raw_data = json.load(f_in) # 尝试读取整个文件
            if not isinstance(raw_data, list): raw_data = [raw_data]
        except:
            f_in.seek(0)
            raw_data = [json.loads(line) for line in f_in if line.strip()]

        for item in raw_data:
            user_content = item["messages"][0]["content"]
            assistant_label = item["messages"][1]["content"] # "是" 或 "否"
            
            # 匹配类别
            target_cat = "未知"
            for kw, cat in prompt_to_category.items():
                if kw in user_content:
                    target_cat = cat
                    break
            
            if target_cat == "未知" or target_cat not in md_content_cache:
                continue
            
            # 构造新的消息结构
            new_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": md_content_cache[target_cat]
                    },
                    {
                        "role": "assistant",
                        "content": generate_dummy_json(target_cat, assistant_label == "是")
                    }
                ],
                "images": item["images"]
            }
            
            f_out.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            converted_count += 1

    print(f"✨ 转换完成！共处理 {converted_count} 条测试数据。")

# 执行转换
input_file = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao/test_single.json"  # 你的旧测试集文件
output_file = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/test_jintao/test_single_longprompt.jsonl"
convert_test_set(input_file, output_file)
