import json
import os
import re

def convert_image_folder(input_dir, output_jsonl, md_dir="./prompts"):
    """
    遍历文件夹内的所有图片，根据图片的上一级文件夹名称生成测试数据。
    """
    
    # 支持的图片扩展名
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # 1. 定义【文件夹名称】到【内部类别】的映射
    # key 是文件夹的名字，value 是对应的内部类别
    folder_tag_to_category = {
        "流动摊贩": "流动摊贩",
        "占道经营": "出店经营",
        "人员异常倒地": "人员异常倒地",         
        "人群异常聚集": "人群异常聚集",         
        "污水倾倒": "雨水井倾倒污水",       
        "垃圾桶满溢": "垃圾桶满溢",
        "垃圾堆积": "垃圾堆积",
        "非机动车乱停放": "非机动车乱停放"
    }    

    # 2. 定义类别对应的 MD 文件名
    category_to_md = {
        "流动摊贩": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_占道经营事件.md",
        "出店经营": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_占道经营事件.md",
        "非机动车乱停放": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_非机动车乱停放.md",
        "雨水井倾倒污水": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_雨水井倾倒污水.md",
        "垃圾桶满溢": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_垃圾事件.md",
        "垃圾堆积": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_垃圾事件.md",
        "人员异常倒地": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_倒地事件.md",
        "人群异常聚集": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_人群异常聚集.md"
    }

    # 3. 预加载所有 MD 内容
    md_content_cache = {}
    for cat, filename in category_to_md.items():
        path = os.path.join(md_dir, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                md_content_cache[cat] = f.read().strip()
        else:
            print(f"⚠️ 警告: 找不到Prompt文件 {path}")

    def generate_dummy_json(category, is_positive=True):
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

    # 4. 遍历文件夹处理图片
    converted_count = 0
    skipped_count = 0

    print(f"📂 开始遍历文件夹: {input_dir}")

    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for root, dirs, files in os.walk(input_dir):
            # 获取当前遍历到的文件夹名称（即图片的上一级目录）
            current_folder_name = os.path.basename(root)
            
            # 确定当前文件夹对应的类别
            target_cat = "未知"
            if current_folder_name in folder_tag_to_category:
                target_cat = folder_tag_to_category[current_folder_name]
            
            # 如果当前文件夹不在定义的映射中，或者对应类别没有 prompt，则该文件夹下的图片都将跳过
            if target_cat == "未知" or target_cat not in md_content_cache:
                # 仅统计跳过的图片数量，避免每个文件都打印 log
                # 这里为了统计准确，需要遍历 files 计算数量
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in VALID_EXTENSIONS:
                        skipped_count += 1
                continue

            # 遍历当前文件夹下的所有文件
            for file in files:
                # 检查后缀名
                ext = os.path.splitext(file)[1].lower()
                if ext not in VALID_EXTENSIONS:
                    continue

                # 获取绝对路径
                img_path = os.path.abspath(os.path.join(root, file))
                
                # 构造数据
                new_item = {
                    "messages": [
                        {
                            "role": "user",
                            "content": md_content_cache[target_cat]
                        },
                        {
                            "role": "assistant",
                            "content": generate_dummy_json(target_cat, is_positive=True)
                        }
                    ],
                    "images": [img_path]
                }
                
                f_out.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 100 == 0:
                    print(f"   已处理 {converted_count} 张图片...")

    print(f"✨ 转换完成！")
    print(f"   - 成功生成: {converted_count} 条")
    print(f"   - 跳过/文件夹未定义: {skipped_count} 条")
    print(f"   - 输出文件: {output_jsonl}")

# ================= 配置路径 =================
# 输入的图片文件夹路径
input_images_dir = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/测试数据集_3cls/算法测试_3cls_清洗后/正样本" 

# 输出的 jsonl 文件路径
output_file = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/测试数据集_3cls/算法_3cls_清洗后_正.jsonl"


# 执行
if os.path.exists(input_images_dir):
    convert_image_folder(input_images_dir, output_file)
else:
    print(f"❌ 错误: 找不到输入文件夹 {input_images_dir}")