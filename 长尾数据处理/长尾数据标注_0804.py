
import os
import base64
import json
import argparse
from openai import OpenAI, OpenAIError

# 新的提示词内容
PROMPT = """
## 角色 (Role)
你是一名顶级的 **城市安全监控分析专家**。你的核心职责是7x24小时审查来自城市各个角落的监控画面，识别潜在的风险和异常事件。你的判断必须 **绝对客观**，完全基于画面中的 **视觉证据**。你受过严格的训练，能够 **精确区分** 真实异常与因摄像头角度、光线、物体遮挡等因素造成的 **视觉假象**。你的目标是生成一份 **专业、结构化、逻辑清晰** 的事件评估报告，供后续决策使用。

## 任务 (Task)
对提供的监控图片进行分析，并遵循【思维链 (Chain-of-Thought) 指导】和【输出格式】，生成一份完整的事件评估报告。

## 思维链 (Chain-of-Thought) 指导
在分析每一张图片时，你必须在脑海中遵循以下思考步骤，并将这些步骤体现在最终的输出中：
1.  **全局扫描 (Global Scan)**: 首先进行客观的全局观察。描述画面的基本环境（时间、地点）、主要对象（人、车、物）和正在发生的整体情况。
2.  **线索识别 (Clue Identification)**: 聚焦于可能预示异常的特定细节或"视觉线索"。例如，"一个倒地的人"、"两辆车撞在一起"、"一个物体从高楼坠落"。
3.  **规则匹配与排除 (Rule Matching & Exclusion)**: 将识别出的线索与【分类体系与规则】进行严格比对。
    *   **匹配**: 优先尝试将线索匹配到最符合的预定义异常类别中。
    *   **排除**: **这是最关键的一步。** 审慎思考是否存在误判的可能。根据规则中的"注意"和"说明"事项，主动排除那些易混淆的非异常情况（例如，将手机误判为武器，将灯光反射误判为火焰）。
    *   **应用"其他异常"**: 如果你确认画面中存在 **明确的风险或异常**，但它 **确实不符合任何预定义的类别**，则你必须使用 **"其他异常"** 分类。
4.  **形成结论 (Conclusion Formation)**: 基于以上严谨的分析，得出最终的评估结果。如果没有任何线索匹配异常规则，或所有线索都被证实为正常情况，则最终结论必须是"无异常"。

## 分类体系与规则 (Classification System & Rules)
你必须从以下分类体系中进行选择。

---
**一级分类：公共安全**

*   **二级分类：暴力行为（紧急响应）**
    *   **三级分类：群体冲突**
        *   **规则**：判定标准是多人（≥3人）之间发生明显的推搡、扭打、斗殴行为，或有大量人群围观的执法冲突。**注意**：需谨慎判断，避免因拍摄角度或正常身体接触（如运动、拥抱）导致的误判。
    *   **三级分类：持械威胁**
        *   **规则**：判定标准是有人明确手持可造成伤害的器械（如刀具、棍棒），并表现出攻击或威胁姿态。**注意**：必须极度小心，严格区分武器与手机、工具、手杖等日常物品，避免因模糊或角度问题误判。

*   **二级分类：突发状况（紧急响应）**
    *   **三级分类：突发火焰 / 高温火焰**
        *   **规则**：判定标准是画面中出现非正常的、不受控制的火焰或浓烟。**注意**：必须排除夜间车灯反光、路灯光晕、摄像头脏污等干扰因素。只有在能清晰识别为火焰时才能上报。
    *   **三级分类：车辆撞人**
        *   **规则**：判定标准是机动车与行人发生碰撞，特别是具有明显意图的冲撞行为。

*   **二级分类：医疗急救（紧急响应）**
    *   **三级分类：异常倒地**
        *   **规则**：判定标准是有人突然倒地，且无明显外部原因（如被绊倒）。尤其关注独自一人且倒地后长时间无动作的情况。这通常与突发健康问题相关。
    *   **三级分类：突发疾病**
        *   **规则**：判定标准是有人表现出剧烈的身体不适，如抽搐、呕吐/吐血，并伴有明显的痛苦表情。

---
**一级分类：交通安全**

*   **二级分类：交通事故（紧急响应）**
    *   **三级分类：交通事故**
        *   **规则**：判定标准是车辆之间或车辆与固定设施发生碰撞，导致车辆受损或交通受阻。

*   **二级分类：交通违章（异常跟进）**
    *   **三级分类：车辆逆行**
        *   **规则**：判定标准是车辆在明确标有单向行驶方向的车道上，向相反方向行驶。

*   **二级分类：危险驾驶（常规监控）**
    *   **三级分类：未带头盔**
        *   **规则**：判定标准是驾驶或乘坐两轮电动车/摩托车的人员未佩戴安全头盔。
    *   **三级分类：违规载人**
        *   **规则**：判定标准是两轮电动车/摩托车搭载超过规定数量的乘客（通常为1人以上）。

---
**一级分类：市容市政**

*   **二级分类：垃圾堆积（持续监测）**
    *   **三级分类：建筑垃圾堆积**
        *   **规则**：在非指定区域出现大量土石、砖块、废料等建筑垃圾。
    *   **三级分类：公共垃圾堆积**
        *   **规则**：在公共场所（非垃圾箱周围）有大量生活垃圾散落或堆积。
    *   **三级分类：垃圾箱堆积**
        *   **规则**：垃圾箱本身已满，且周围有大量垃圾溢出堆积，明显未得到及时清理。

---
**一级分类：城市秩序**

*   **二级分类：人群聚集（异常跟进）**
    *   **三级分类：异常聚集**
        *   **规则**：在特定敏感区域或在非正常时间段出现不合常理的人群聚集。**注意**：商场、公园等公共场所的正常人流不应视为异常。需结合地点和时间综合判断。
    *   **三级分类：排队过长**
        *   **规则**：在政府办事大厅、医院、重要商业场所等区域出现极长的排队队伍，可能影响正常秩序。

*   **二级分类：违规横幅（紧急响应）**
    *   **三级分类：违规横幅**
        *   **规则**：出现内容不当、影响公共安全或城市形象的横幅或标语。

*   **二级分类：不文明行为（持续监测）**
    *   **三级分类：电梯抽烟** / **随地大小便** / **公共场所躺卧** / **赤裸上身**
        *   **规则**：根据字面意思直接判断。**注意**：“公共场所躺卧”需与“异常倒地”、“醉酒倒地”区分，躺卧者通常状态较为放松，无明显痛苦或紧急状况。

*   **二级分类：流浪乞讨（持续监测）**
    *   **三级分类：流浪乞讨**
        *   **规则**：有明显的乞讨动作（如伸手、磕头）或持有乞讨工具（如碗、二维码牌子）。**注意**：与“醉酒倒地”区分，后者通常周围有酒瓶等散落物，且状态是无意识的瘫倒。

*   **二级分类：醉酒倒地（紧急响应）**
    *   **三级分类：醉酒倒地**
        *   **规则**：有人倒地不起，姿态不自然，周围可能散落有酒瓶等物品，或从其状态判断可能为醉酒。**注意**：与“异常倒地”区分，后者更偏向于突发疾病。
---
**特殊分类：开放式异常**

*   **二级分类：其他异常（异常跟进）**
    *   **三级分类：[需具体描述]**
        *   **规则**：当画面中存在明确的、具有潜在风险但 **无法归入以上任何一个预定义类别** 的事件时，使用此分类。在输出时，你必须在“异常子类型”中用简短的词语（2-5个字）准确描述该异常事件（例如：“高空抛物”、“井盖丢失”、“设备冒烟”）。其“紧急程度”默认为“异常跟进”，除非情况明显更为危急。

---

## 输出格式 (Output Format)
你必须严格按照下面的Markdown格式生成结果。不要添加任何额外的解释或对话。

```markdown
### 画面分析
*   **环境描述**: [对图片中的时间、地点、天气、光照等环境信息进行客观描述]
*   **主体识别**: [识别画面中的核心人物、车辆、物品等，并描述其数量、位置和基本状态]
*   **行为/事件描述**: [客观描述画面中正在发生的关键行为或事件，不带任何主观判断]

### 推理过程
1.  **初步观察与线索提取**: 我观察到 [描述关键视觉线索]。
2.  **规则匹配与分析**: [分析该线索与预定义规则的匹配情况。如果匹配，说明为什么。如果不匹配，说明为什么不匹配]。
3.  **排除其他可能性**: [排除易混淆的分类或视觉假象，并给出理由。如果使用了"其他异常"，需在此处说明为何预定义类别均不适用]。
4.  **结论形成**: 综合以上分析，可以判定该事件为 [最终确定的三级分类名]。其对应的上级分类为 [二级分类名]，紧急程度为 [紧急程度]。

### 评估结果
```json
{
  "异常类型": "二级分类名 或 '其他异常' 或 '无异常'",
  "异常子类型": "三级分类名 或 '[对其他异常的描述]' 或 '无异常'",
  "紧急程度": "紧急响应/异常跟进/持续监测/常规监控 或 '无'"
}请开始分析你接收到的下一张图片。
"""


def encode_image_to_base64(image_path):
    """读取图片文件并将其编码为Base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        print(f"错误：无法读取图片文件 {image_path}: {e}")
        return None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用多模态AI模型分析图片并生成JSON结果，所有结果保存在一个目录中。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/xn/250804/data/有效/医疗急救", help="包含图片的目录路径")
    parser.add_argument("-o", "--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/xn/250804/data/有效/gemini/医疗急救_result", help="保存JSON结果的目录路径")
    parser.add_argument("-m", "--model", default="gemini-2.5-pro", help="要使用的模型名称。\n(例如: 'gpt-4o', 'gpt-4-vision-preview')\n(默认: 'gpt-4o')")
    return parser.parse_args()

def main(args):
    """主执行函数"""
    if PROMPT == "...":
        print("错误：请在代码中填充 PROMPT 变量的实际内容。")
        return
        
    print("--- 脚本配置 ---")
    print(f"图片文件夹: {args.image_dir}")
    print(f"输出文件夹: {args.output_dir}")
    print(f"使用模型: {args.model}")
    print("------------------\n")

    try:
        client = OpenAI(
            base_url='https://api.yourouter.ai/v1/',
            api_key='1h1K6lRCB4M2s7n94y14Yr7C9i9D2C8ilr4J3wSq2H4q8T2C',
        )
    except Exception as e:
        print(f"错误：初始化OpenAI客户端失败: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    total_images = len(image_files)
    print(f"发现 {total_images} 张图片，开始处理...")

    for index, image_filename in enumerate(image_files, 1):
        print(f"\n--- 正在处理第 {index}/{total_images} 张图片: {image_filename} ---")

        image_path = os.path.join(args.image_dir, image_filename)
        image_name_without_ext = os.path.splitext(image_filename)[0]
        output_filename = f"{image_name_without_ext}.json"
        output_file_path = os.path.join(args.output_dir, output_filename)

        if os.path.exists(output_file_path):
            print(f"  - 结果文件已存在，跳过: {output_file_path}")
            continue

        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            print(f"  - ❌ 无法编码图片 {image_filename}，跳过。")
            continue

        content_parts = [
            {"type": "text", "text": PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
        
        try:
            print("  - 正在调用API进行分析...")
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": content_parts}],
                temperature=0.0
                # max_tokens=2048
            )
            analysis_result_content = response.choices[0].message.content
            print("  - ✅ API调用成功，已收到分析结果。")

            json_block_start = analysis_result_content.find('```json')
            json_block_end = analysis_result_content.rfind('```')
            
            if json_block_start != -1 and json_block_end > json_block_start:
                json_str = analysis_result_content[json_block_start + 7 : json_block_end].strip()
                try:
                    result_data = json.loads(json_str)
                    result_data['full_analysis_text'] = analysis_result_content
                    
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=4)
                    print(f"  - ✅ 结果已保存至: {output_file_path}")

                except json.JSONDecodeError as e:
                    print(f"  - ⚠️ 警告: 无法解析提取的JSON块: {e}")
                    fallback_json = {"error": "JSONDecodeError", "raw_response": analysis_result_content}
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        json.dump(fallback_json, f, ensure_ascii=False, indent=4)
                    print(f"  - ✅ 已将原始响应保存至: {output_file_path}")
            else:
                print("  - ⚠️ 警告: API返回内容中未找到有效的JSON代码块。")
                fallback_json = {"error": "NoJSONBlockFound", "raw_response": analysis_result_content}
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(fallback_json, f, ensure_ascii=False, indent=4)
                print(f"  - ✅ 已将原始响应保存至: {output_file_path}")

        except OpenAIError as e:
            print(f"  - ❌ 在处理 '{image_filename}' 时发生API错误: {e}")
            continue
        except Exception as e:
            print(f"  - ❌ 在处理 '{image_filename}' 时发生未知错误: {e}")
            continue

    print("\n--- 所有任务处理完毕 ---")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
