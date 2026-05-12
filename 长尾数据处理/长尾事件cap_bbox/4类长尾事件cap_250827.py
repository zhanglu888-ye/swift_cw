import os
import json
import base64
import argparse
import logging
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any, Optional

# --- 基本配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 全局常量 ---
# TARGET_CATEGORIES = {"垃圾桶满溢", "垃圾堆积"}
# TARGET_CATEGORIES = {"人员异常倒地"}
# TARGET_CATEGORIES = {"流动摊贩", "出店经营"}
TARGET_CATEGORIES = {"异质文化异常聚集", "开放空间异常聚集"}

def encode_image_to_base64(image_path: Path) -> str:
    """读取图片文件并将其编码为base64字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        logging.error(f"无法读取或编码图片 {image_path}: {e}")
        raise

def get_prompt_template(prompt_path: Path) -> str:
    """从文件加载Prompt模板。"""
    try:
        return prompt_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        logging.error(f"Prompt模板文件未找到: {prompt_path}")
        raise
    except Exception as e:
        logging.error(f"读取Prompt模板时发生错误 {prompt_path}: {e}")
        raise

def call_vision_api(
    client: OpenAI,
    model: str,
    prompt: str,
    base64_large_image: str,
    sub_events: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    调用视觉API，发送一张大图和多张（或零张）小图。
    """
    logging.info(f"正在为包含 {len(sub_events)} 个子事件的案例调用视觉API...")

    # 构建API请求的消息体
    messages_content = [{"type": "text", "text": prompt}]

    # 添加主图片
    messages_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_large_image}"}
    })

    # 添加所有子事件的特写图片 (如果sub_events为空，此循环不执行)
    for event in sub_events:
        messages_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{event['base64_image']}"}
        })
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": messages_content}],
            max_tokens=6144,
            temperature=0.0,
            response_format={"type": "json_object"}, # 确保返回JSON对象
        )
        response_text = response.choices[0].message.content
        
        return json.loads(response_text)

    except json.JSONDecodeError as e:
        logging.error(f"API返回的不是有效的JSON格式: {e}")
        logging.error(f"原始响应: {response_text}")
        return None
    except Exception as e:
        logging.error(f"调用API时发生错误: {e}")
        return None


def process_directories(
    large_image_dir: Path,
    cropped_data_dir: Path,
    output_dir: Path,
    prompt_template: str,
    client: OpenAI,
    model: str
):
    """
    【修改后】遍历抠图数据目录，聚合数据（包括空集），调用API并保存结果。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"开始处理目录: {cropped_data_dir}")

    # 遍历以大图命名的案例文件夹
    for case_dir in cropped_data_dir.iterdir():
        if not case_dir.is_dir():
            continue

        case_name = case_dir.name
        logging.info(f"--- 正在检查案例: {case_name} ---")

        # 1. 查找对应的大图
        large_image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            potential_path = large_image_dir / f"{case_name}{ext}"
            if potential_path.exists():
                large_image_path = potential_path
                break
        
        if not large_image_path:
            logging.warning(f"找不到与案例 '{case_name}' 对应的大图，跳过。")
            continue
        
        # 2. 【修改】筛选目标子文件夹，但不在此处跳过
        sub_event_dirs = [d for d in case_dir.iterdir() if d.is_dir() and d.name in TARGET_CATEGORIES]
        
        if not sub_event_dirs:
            logging.info(f"  -> 未找到目标类别，案例 '{case_name}' 将作为负样本处理。")
        else:
            logging.info(f"  -> 找到目标类别 {[d.name for d in sub_event_dirs]}，案例 '{case_name}' 将作为正样本处理。")

        # 3. 聚合数据：如果sub_event_dirs为空，则sub_events_for_...列表也为空
        sub_events_for_prompt = []
        sub_events_for_api = []
        crop_index = 1
        
        for category_dir in sub_event_dirs:
            category_name = category_dir.name
            for small_image_path in category_dir.glob('*.*'):
                try:
                    base64_small_image = encode_image_to_base64(small_image_path)
                    
                    sub_events_for_prompt.append({
                        "crop_image": f"[Image_Crop_{crop_index}]",
                        "category_name": category_name
                    })
                    
                    sub_events_for_api.append({
                        "base64_image": base64_small_image
                    })
                    
                    crop_index += 1
                except Exception as e:
                    logging.warning(f"处理小图 {small_image_path} 时失败: {e}")
                    continue
        
        # 4. 【修改】调整安全检查逻辑
        # 仅当它是正样本（有目标文件夹）但所有小图都处理失败时才跳过
        if sub_event_dirs and not sub_events_for_api:
            logging.warning(f"案例 '{case_name}' 中没有成功处理的小图，跳过。")
            continue

        # 5. 构建最终Prompt (对于负样本，sub_event_list_str将是 '[]')
        sub_event_list_str = json.dumps(sub_events_for_prompt, ensure_ascii=False, indent=2)
        
        final_prompt = prompt_template.split("#### # 输入信息")[0] 
        final_prompt += (
            "#### # 输入信息\n\n"
            f"*   **主图片**: [Image_Main]\n"
            f"*   **子事件列表**: `{sub_event_list_str}`"
        )

        # 6. 调用API
        try:
            base64_large_image = encode_image_to_base64(large_image_path)
            analysis_result = call_vision_api(
                client=client,
                model=model,
                prompt=final_prompt,
                base64_large_image=base64_large_image,
                sub_events=sub_events_for_api
            )

            if analysis_result:
                # 7. 保存结果
                # 为负样本结果添加后缀以区分
                suffix = "_negative" if not sub_event_dirs else ""
                output_json_path = output_dir / f"{case_name}{suffix}.json"
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, ensure_ascii=False, indent=2)
                logging.info(f"    [+] 分析成功！结果已保存至: {output_json_path}")
            else:
                logging.error(f"案例 '{case_name}' 的API调用失败或返回无效结果。")

        except Exception as e:
            logging.error(f"处理案例 '{case_name}' 时发生严重错误: {e}")


def main():
    # 尝试从.env文件加载环境变量
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass # 如果没有安装dotenv，就忽略

    parser = argparse.ArgumentParser(
        description="使用统一的视觉API Prompt分析城市管理事件图片（支持正/负样本）。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 参数与您提供的代码保持一致
    parser.add_argument(
        "--large_image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/tmp_0827/images_负样本",
        help="包含原始大图的文件夹路径。"
    )
    parser.add_argument(
        "--cropped_data_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/tmp_0827/imgandlabel_show",
        help="包含按大图名称组织的抠图结果的根目录。"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/tmp_0827/analysis_output_聚集",
        help="保存JSON分析结果的文件夹路径 (默认: 'analysis_output')。"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="4类长尾事件cap_聚集.md",   ##4类长尾事件cap_垃圾堆积.md
        help="包含Prompt模板的 .md 文件路径 (默认: 'prompt.md')。"
    )
    parser.add_argument(
        "--api_key", type=str, default='1h1K6lRCB4M2s7n94y14Yr7C9i9D2C8ilr4J3wSq2H4q8T2C',
        help="OpenAI API密钥。如果未提供，将尝试从环境变量 'OPENAI_API_KEY' 读取。"
    )
    parser.add_argument(
        "--api_base_url", type=str, default="https://api.yourouter.ai/v1/",
        help="OpenAI API的基础URL。可用于指向兼容的第三方服务。"
    )
    parser.add_argument(
        "--model", type=str, 
        # default="doubao-pro-vision-2",
        default="gemini-2.5-pro",
        help="要使用的模型名称。"
    )

    args = parser.parse_args()

    if not args.api_key:
        logging.critical("致命错误: 未提供API密钥。请使用 --api_key 参数或在 .env 文件中设置 OPENAI_API_KEY。")
        return
        
    for path_str in [args.large_image_dir, args.cropped_data_dir, args.prompt_path]:
        if not Path(path_str).exists():
            logging.critical(f"致命错误: 路径 '{path_str}' 不存在。请检查路径。")
            return

    try:
        client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)
        prompt_template = get_prompt_template(Path(args.prompt_path))

        process_directories(
            large_image_dir=Path(args.large_image_dir),
            cropped_data_dir=Path(args.cropped_data_dir),
            output_dir=Path(args.output_dir),
            prompt_template=prompt_template,
            client=client,
            model=args.model
        )
        logging.info("所有任务处理完成！")

    except Exception as e:
        logging.critical(f"在主流程中发生无法恢复的错误: {e}")

if __name__ == "__main__":
    main()