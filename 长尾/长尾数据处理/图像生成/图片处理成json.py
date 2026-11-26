import json
import os
import re

def convert_image_folder(input_dir, output_jsonl, md_dir="./prompts"):
    """
    遍历文件夹内的所有图片，支持两种文件名格式：
    1. 带括号标签: ..._Trash_Accumulation (垃圾堆积 - 无桶)_1...
    2. 纯英文标签: ..._Trash_Overflow_1...
    """
    
    # 支持的图片扩展名
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # 1. 定义文件名中的【标签】到【内部类别】的映射
    # key: 图片文件名 () 中的中文内容，或者文件名中包含的英文关键词
    # value: category_to_md 中的 key
    filename_tag_to_category = {
        # --- 中文标签 (括号内内容) ---
        "流动摊贩": "流动摊贩",
        "出店经营": "出店经营",
        "人员倒地": "人员异常倒地",         
        "人员聚集": "人群异常聚集",         
        "倾倒污水": "雨水井倾倒污水",       
        "垃圾桶满溢": "垃圾桶满溢",
        "垃圾堆积 - 无桶": "垃圾堆积",       
        
        # --- 英文标签 (文件名包含内容) ---
        "Trash_Overflow": "垃圾桶满溢",
        "Trash_Accumulation": "垃圾堆积", 
        "Mobile_Vendor": "流动摊贩",
        "Store_Outside": "出店经营",
        "Person_Falling": "人员异常倒地",  
        "Abnormal_Gathering": "人群异常聚集", 
        "Pouring_Sewage": "雨水井倾倒污水" 
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
            for file in files:
                # 检查后缀名
                ext = os.path.splitext(file)[1].lower()
                if ext not in VALID_EXTENSIONS:
                    continue

                # 获取绝对路径
                img_path = os.path.abspath(os.path.join(root, file))
                
                target_cat = "未知"
                matched_tag = None

                # --- 策略 1: 优先提取文件名中括号 () 内的内容 (针对格式1) ---
                # 示例: ...Accumulation (垃圾堆积 - 无桶)_1... -> 提取 "垃圾堆积 - 无桶"
                match = re.search(r'\((.*?)\)', file)
                if match:
                    tag_in_brackets = match.group(1).strip()
                    if tag_in_brackets in filename_tag_to_category:
                        target_cat = filename_tag_to_category[tag_in_brackets]
                        matched_tag = tag_in_brackets

                # --- 策略 2: 如果策略1未匹配，遍历字典 key 检查文件名是否包含该关键词 (针对格式2) ---
                # 示例: ..._Trash_Overflow_1... -> 包含 "Trash_Overflow"
                if target_cat == "未知":
                    for key in filename_tag_to_category.keys():
                        # 确保 key 在文件名中，且这里可以加一些边界判断防止部分匹配（如 'Trash' 匹配 'Trash_Overflow'）
                        # 但由于字典key比较特异（如下划线连接），直接包含判断通常足够
                        if key in file:
                            target_cat = filename_tag_to_category[key]
                            matched_tag = key
                            break # 找到一个匹配的就停止
                
                # 如果找不到对应的类别，或该类别没有加载到MD内容，则跳过
                if target_cat == "未知" or target_cat not in md_content_cache:
                    skipped_count += 1
                    # 可以取消注释查看哪些没匹配上
                    # print(f"⚠️ 跳过 - 未匹配到已知类别: {file}")
                    continue
                
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
    print(f"   - 跳过/无匹配: {skipped_count} 条")
    print(f"   - 输出文件: {output_jsonl}")

# ================= 配置路径 =================
# 输入的图片文件夹路径
input_images_dir = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/测试数据集_3cls/jintao_3cls_清洗后" 

# 输出的 jsonl 文件路径
output_file = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/测试数据集_3cls/jintao_3cls_清洗后.jsonl"

# 执行
if os.path.exists(input_images_dir):
    convert_image_folder(input_images_dir, output_file)
else:
    print(f"❌ 错误: 找不到输入文件夹 {input_images_dir}")