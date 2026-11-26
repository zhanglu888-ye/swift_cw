import os
import json
import base64
import argparse
import logging
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any

# --- 配置区域 ---

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 事件配置: 您可以根据需要修改、增加或删除这里的配置
# 确保 prompt_path 指向您存放 prompt 文件的正确位置
EVENT_CONFIG: Dict[str, Dict[str, Any]] = {
    # 第一批4类
    "垃圾事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_垃圾事件.md"),
        "categories": {"垃圾桶满溢", "垃圾堆积"}
    },
    "倒地事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_倒地事件.md"),
        "categories": {"人员异常倒地"}
    },
    "占道经营事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_占道经营事件.md"),
        "categories": {"流动摊贩", "占道经营"}
    },
    #"人群聚集事件": {
    #    "prompt_path": Path("prompts/cap_人群聚集事件.md"),
    #    "categories": {"异质文化异常聚集", "开放空间异常聚集"}
    #},
    # 第二批新增4类 (示例)
    "森林火灾事件": {
        "prompt_path": Path("/nfs/zhanglu/tools/长尾事件cap_nobbox/prompts/cap_森林火灾.md"),
        "categories": {"森林火灾"}
    },
    #"设施损坏事件": {
    #    "prompt_path": Path("prompts/cap_设施损坏事件.md"),
    #    "categories": {"公共设施损坏"}
    #},
    #"违规撑伞事件": {
    #    "prompt_path": Path("prompts/cap_违规撑伞事件.md"),
    #    "categories": {"违规撑伞"}
    #},
    #"人员打斗事件": {
    #    "prompt_path": Path("prompts/cap_人员打斗事件.md"),
    #    "categories": {"人员打斗"}
    #}
}

# --- 辅助函数 ---

def encode_image_to_base64(image_path: Path) -> str:
    """将图片文件编码为Base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_all_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    """根据配置加载所有Prompt模板"""
    templates = {}
    for event_type, details in config.items():
        try:
            templates[event_type] = details["prompt_path"].read_text(encoding='utf-8')
            logging.info(f"成功加载Prompt for '{event_type}' from '{details['prompt_path']}'")
        except FileNotFoundError:
            logging.error(f"致命错误: Prompt文件 '{details['prompt_path']}' 未找到！请检查路径。")
            raise
        except Exception as e:
            logging.error(f"读取Prompt '{details['prompt_path']}' 失败: {e}")
            raise
    return templates

def build_category_map(config: Dict[str, Any]) -> Dict[str, str]:
    """构建从具体类别到高级事件类型的反向映射"""
    category_map = {}
    for event_type, details in config.items():
        for category in details["categories"]:
            category_map[category] = event_type
    return category_map

def call_vision_api(client: OpenAI, model: str, prompt: str, base64_large_image: str) -> Any:
    """调用视觉API，参数已根据新要求更新"""
    messages_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_large_image}"}}
    ]
   
    messages_content = [
              {"type": "text", "text": "请介绍一下你的模型能力"}
            ]
    #try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": messages_content}],
        temperature=0.2,
        max_tokens=512,
        # 注意：仍然保留 response_format，因为它对于确保输出是可解析的JSON至关重要。
        # 如果新模型不支持此参数，可以将其移除，但需要对API的原始文本输出进行手动解析和错误处理。
        response_format={"type": "json_object"}
    )
    print(response.choices[0])
    return json.loads(response.choices[0].message.content)
    #except Exception as e:
    #    logging.error(f"调用API时发生错误: {e}")
    #    return None

# --- 核心处理逻辑 ---

def process_sample(
    label_path: Path, 
    image_dir: Path, 
    output_dir: Path, 
    prompt_templates: Dict[str, str], 
    category_map: Dict[str, str],
    client: OpenAI, 
    model: str
):
    """处理单个样本（一张图及其对应的标签json）"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        file_name = label_data.get("file_name")
        labels = label_data.get("labels", [])
        
        if not file_name or not labels:
            logging.warning(f"跳过 {label_path.name}: 'file_name' 或 'labels' 字段缺失/为空。")
            return

        # **修改点**: 从独立的image_dir中寻找图片
        image_path = image_dir / file_name
        if not image_path.exists():
            logging.warning(f"跳过 {file_name}: 在图片目录 '{image_dir}' 中找不到对应的图片文件。")
            return
            
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error(f"读取或解析标签文件 {label_path} 失败: {e}")
        return

    logging.info(f"--- 正在处理样本: {file_name} ---")
    logging.info(f"    原始标签: {labels}")

    events_to_process: Dict[str, List[str]] = {}
    for label in labels:
        event_type = category_map.get(label)
        if event_type:
            if event_type not in events_to_process:
                events_to_process[event_type] = []
            events_to_process[event_type].append(label)
        else:
            logging.info(f"    注意：标签 '{label}' 在当前EVENT_CONFIG中未配置，将被忽略。")

    if not events_to_process:
        logging.info(f"    样本 {file_name} 中没有需要处理的已配置事件，跳过。")
        return

    logging.info(f"    识别到的事件组: {events_to_process}")

    try:
        base64_large_image = encode_image_to_base64(image_path)
    except Exception as e:
        logging.error(f"    编码大图 {image_path} 失败: {e}")
        return

    all_analyses = []
    processed_event_types = []
    for event_type, category_list in events_to_process.items():
        logging.info(f"      -> 正在为 '{event_type}' (包含: {category_list}) 调用API...")
        
        prompt_template = prompt_templates[event_type]
        category_list_str = json.dumps(category_list, ensure_ascii=False)
        
        final_prompt = prompt_template.split("#### # 输入信息")[0] + (
            f"#### # 输入信息\n\n"
            f"*   **主图片**: [Image_Main]\n"
            f"*   **待检测类别**: `{category_list_str}`\n\n"
            f"请根据主图片和待检测类别，对图中发生的事件进行分析。"
        )

        analysis_result = call_vision_api(client, model, final_prompt, base64_large_image)
        
        if analysis_result:
            analysis_result["source_event_type"] = event_type
            analysis_result["source_categories"] = category_list
            all_analyses.append(analysis_result)
            processed_event_types.append(event_type)
            logging.info(f"      [+] '{event_type}' 分析成功。")
        else:
            logging.error(f"      [!] '{event_type}' API调用失败或返回空结果。")
    
    if not all_analyses:
        logging.error(f"    样本 {file_name} 的所有事件组都处理失败，不保存任何结果。")
        return

    final_output = { "file_name": file_name, "analyses": all_analyses }
    processed_event_types.sort()
    save_event_type_dir = processed_event_types[0]
    
    output_subdir = output_dir / save_event_type_dir
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    output_json_path = output_subdir / f"{image_path.stem}.json"
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    logging.info(f"    [SUCCESS] 组合分析结果已保存至: {output_json_path}")

# --- 主函数 ---

def main():
    parser = argparse.ArgumentParser(description="根据大图和标签JSON，动态调用VLM进行多事件分析。")
    # **修改点**: 输入参数变为两个独立目录
    parser.add_argument("--image_dir", default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/合并_deduplicated/large_images", help="包含所有图片的文件夹路径。")
    parser.add_argument("--label_dir", default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/合并_deduplicated/label_json", help="包含所有标签json文件的文件夹路径。")
    parser.add_argument("--output_dir", default="/nfs/zhanglu/changwei/dataset/202511/vlm_prepare/分类and描述", help="保存JSON分析结果的根目录。")
    
    # **修改点**: 更新API相关参数的默认值
    parser.add_argument(
        "--api_key", type=str, default="McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ",
        help="API密钥。"
    )
    parser.add_argument(
        "--api_base_url", type=str, default="http://10.21.65.20:8080/v1/",
        help="API的基础URL。"
    )
    parser.add_argument(
        "--model", type=str, 
        default="Qwen3-VL-235B-A22B-Instruct",
        help="要使用的模型名称。"
    )
    args = parser.parse_args()

    if not args.api_key:
        logging.critical("致命错误: 未提供API密钥。")
        return

    try:
        prompt_templates = load_all_prompts(EVENT_CONFIG)
        category_map = build_category_map(EVENT_CONFIG)
        client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)
        
        image_path = Path(args.image_dir)
        label_path = Path(args.label_dir)
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        label_files = list(label_path.glob('*.json'))
        if not label_files:
            logging.warning(f"在标签目录 '{label_path}' 中未找到任何 .json 文件。")
            return
            
        logging.info(f"发现 {len(label_files)} 个标签文件，开始处理...")

        for label_file_path in label_files:
            process_sample(label_file_path, image_path, output_path, prompt_templates, category_map, client, args.model)
            
        logging.info("所有任务处理完成！")

    except Exception as e:
        logging.critical(f"主流程发生严重错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()
