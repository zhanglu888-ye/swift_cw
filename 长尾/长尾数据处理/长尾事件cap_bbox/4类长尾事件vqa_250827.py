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
# 根据您的需要设置目标类别
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

def call_vqa_api(
    client: OpenAI,
    model: str,
    prompt: str,
    base64_large_image: str,
    base64_crop_image: str
) -> Optional[Dict[str, Any]]:
    """
    【VQA模式】调用视觉API，每次发送一张大图和一张小图。
    """
    # 构建API请求的消息体
    messages_content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_large_image}"}
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_crop_image}"}
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": messages_content}],
            max_tokens=2048, # VQA通常不需要很长的token
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
    【VQA模式】遍历抠图数据目录，为每个小图生成VQA对，并保存结果。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"开始处理目录: {cropped_data_dir}")

    # 遍历以大图命名的案例文件夹
    for case_dir in cropped_data_dir.iterdir():
        if not case_dir.is_dir():
            continue

        case_name = case_dir.name
        logging.info(f"--- 正在处理案例: {case_name} ---")

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
        
        try:
            # 2. 编码大图（每个案例只需编码一次）
            base64_large_image = encode_image_to_base64(large_image_path)
        except Exception as e:
            logging.error(f"编码大图 {large_image_path} 失败，跳过案例 '{case_name}': {e}")
            continue
            
        logging.info(f"找到并编码大图: {large_image_path}")

        # 3. 遍历目标类别的子文件夹
        sub_event_dirs = [d for d in case_dir.iterdir() if d.is_dir() and d.name in TARGET_CATEGORIES]
        
        if not sub_event_dirs:
            logging.info(f"案例 '{case_name}' 不包含目标类别，跳过。")
            continue

        # 4. 【核心改动】遍历每一张小图，并单独调用API
        for category_dir in sub_event_dirs:
            category_name = category_dir.name
            
            for small_image_path in category_dir.glob('*.*'):
                logging.info(f"  -> 正在处理小图: {small_image_path.name} (类别: {category_name})")
                
                try:
                    # 4.1 准备数据：编码小图，构建Prompt
                    base64_crop_image = encode_image_to_base64(small_image_path)
                    
                    # 替换Prompt中的类别占位符
                    final_prompt = prompt_template.replace("{{category_name}}", category_name)

                    # 4.2 调用VQA API
                    vqa_result = call_vqa_api(
                        client=client,
                        model=model,
                        prompt=final_prompt,
                        base64_large_image=base64_large_image,
                        base64_crop_image=base64_crop_image
                    )

                    # 4.3 保存结果
                    if vqa_result:
                        # 创建一个唯一的文件名，避免覆盖
                        output_filename = f"{case_name}_{small_image_path.stem}.json"
                        output_json_path = output_dir / output_filename
                        with open(output_json_path, 'w', encoding='utf-8') as f:
                            json.dump(vqa_result, f, ensure_ascii=False, indent=2)
                        logging.info(f"    [+] VQA生成成功！结果已保存至: {output_json_path}")
                    else:
                        logging.error(f"    [-] 小图 '{small_image_path.name}' 的API调用失败或返回无效结果。")

                except Exception as e:
                    logging.error(f"    [-] 处理小图 '{small_image_path.name}' 时发生严重错误: {e}")
                    continue


def main():
    # 尝试从.env文件加载环境变量
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass # 如果没有安装dotenv，就忽略

    parser = argparse.ArgumentParser(
        description="【VQA模式】使用视觉API为城市管理事件图片生成问答对。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--large_image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/tmp_0827/images",
        help="包含原始大图的文件夹路径。"
    )
    parser.add_argument(
        "--cropped_data_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/tmp_0827/imgandlabel_show",
        help="包含按大图名称组织的抠图结果的根目录。"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/tmp_0827/vqa_output_聚集",
        help="保存JSON格式VQA结果的文件夹路径。"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="4类长尾事件vqa_聚集.md",
        help="包含VQA Prompt模板的 .md 文件路径。请确保此文件内容是您提供的VQA提示词。"
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

    # 优先从环境变量获取API Key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.critical("致命错误: 未提供API密钥。请使用 --api_key 参数或在 .env 文件中设置 OPENAI_API_KEY。")
        return
        
    for path_str in [args.large_image_dir, args.cropped_data_dir, args.prompt_path]:
        if not Path(path_str).exists():
            logging.critical(f"致命错误: 路径 '{path_str}' 不存在。请检查路径。")
            return

    try:
        client = OpenAI(api_key=api_key, base_url=args.api_base_url)
        prompt_template = get_prompt_template(Path(args.prompt_path))

        # 确认Prompt模板包含必要的占位符
        if "{{category_name}}" not in prompt_template:
            logging.warning("警告：您的Prompt模板中似乎缺少 '{{category_name}}' 占位符。")

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