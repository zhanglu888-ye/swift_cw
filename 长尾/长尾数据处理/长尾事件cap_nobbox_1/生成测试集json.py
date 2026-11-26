import json
import os

def convert_image_folders(input_root_dir, output_jsonl, md_dir="./prompts"):
    """
    input_root_dir: 包含分类子文件夹的根目录
                    结构示例:
                    /root/data/
                        ├── 流动摊贩/
                        │   ├── 1.jpg
                        │   └── 2.jpg
                        ├── 人员异常倒地/
                        │   └── 3.jpg
    output_jsonl: 输出的jsonl文件路径
    """
    
    # 1. 定义类别对应的 MD 文件名
    # 键(Key)必须与你的文件夹名称完全一致
    category_to_md = {
        "流动摊贩": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_占道经营事件.md",
        "出店经营": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_占道经营事件.md",
        "非机动车乱停放": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_非机动车乱停放.md",
        "雨水井倾倒污水": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_雨水井倾倒污水.md",
        "垃圾桶满溢": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_垃圾事件.md",
        "垃圾堆积": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_垃圾事件.md",
        "人员倒地": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_倒地事件.md",
        "人群异常聚集": "/host/ssddata/disk2/zhanglu/tools/长尾数据处理/长尾事件cap_nobbox_1/prompts_训练/cap_人群异常聚集.md"
    }

    # 支持的图片扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # 2. 预加载所有 MD 内容
    md_content_cache = {}
    print("正在加载 Prompt 模板...")
    # 这里为了防止某些文件夹对应的MD路径重复，我们先去重加载，或者直接按需加载
    # 为了保证逻辑简单，我们遍历配置表
    for cat, filepath in category_to_md.items():
        # 如果是相对路径，结合 md_dir；如果是绝对路径，直接使用
        if not os.path.isabs(filepath):
            filepath = os.path.join(md_dir, filepath)
            
        if os.path.exists(filepath):
            if filepath not in md_content_cache: # 避免重复读取同一个文件
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    md_content_cache[filepath] = content
        else:
            print(f"⚠️ 警告: 找不到 Prompt 文件: {filepath} (对应类别: {cat})")

    def generate_dummy_json(category, is_positive=True):
        """生成符合格式的 Assistant JSON 字符串"""
        # 注意：因为是从分类文件夹读取，通常这里 is_positive 默认为 True
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

    # 3. 开始遍历文件夹
    converted_count = 0
    skipped_count = 0
    
    print(f"开始扫描输入目录: {input_root_dir}")
    
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        # 遍历根目录下的子文件夹
        if not os.path.exists(input_root_dir):
            print(f"❌ 错误: 输入目录不存在: {input_root_dir}")
            return

        for folder_name in os.listdir(input_root_dir):
            folder_path = os.path.join(input_root_dir, folder_name)
            
            # 确保是文件夹
            if not os.path.isdir(folder_path):
                continue
                
            # 这里的 folder_name 就是事件类别 (例如 "流动摊贩")
            current_category = folder_name.strip()
            
            # 检查是否在我们的配置列表中
            if current_category not in category_to_md:
                print(f"ℹ️ 跳过未配置的类别文件夹: {current_category}")
                continue
            
            # 获取对应的 MD 内容
            md_path = category_to_md[current_category]
            # 处理路径可能是相对路径的情况
            if not os.path.isabs(md_path):
                md_path = os.path.join(md_dir, md_path)
                
            if md_path not in md_content_cache:
                # 如果对应的MD文件之前没加载成功，跳过
                continue
                
            prompt_content = md_content_cache[md_path]
            
            # 遍历该类别文件夹下的所有图片
            image_files = os.listdir(folder_path)
            for img_file in image_files:
                name, ext = os.path.splitext(img_file)
                if ext.lower() in valid_extensions:
                    # 获取图片的绝对路径
                    image_abs_path = os.path.abspath(os.path.join(folder_path, img_file))
                    
                    # 构造数据
                    # 既然图片在分类文件夹下，默认它是该类别的正样本 (is_positive=True)
                    assistant_response = generate_dummy_json(current_category, is_positive=True)
                    
                    new_item = {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt_content
                            },
                            {
                                "role": "assistant",
                                "content": assistant_response
                            }
                        ],
                        "images": [image_abs_path]
                    }
                    
                    f_out.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                    converted_count += 1
                else:
                    # 非图片文件跳过
                    pass

    print(f"✨ 转换完成！共生成 {converted_count} 条数据。")
    print(f"📂 结果已保存至: {output_jsonl}")

# 配置参数
# 输入文件夹路径（包含 "流动摊贩", "人员异常倒地" 等子文件夹的根目录）
input_dir_path = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/待标注" 

# 输出文件路径
output_file_path = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202603/待标注_longprompt.jsonl"

# 执行转换
convert_image_folders(input_dir_path, output_file_path)