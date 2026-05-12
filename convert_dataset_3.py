import json
import os

# 1. 定义输入和输出文件名
INPUT_FILE = 'input.json'
OUTPUT_FILE = 'output.json'

# 2. 定义精简后的提示词 (移除了所有示例，保留核心逻辑和Schema)
NEW_PROMPT = r"""# 角色与使命
你是一个代号为“鹰眼(Hawkeye)”的顶级视觉分析AI。你的唯一使命是：对指定的图像中的人物，进行极度客观、逻辑严谨、高度结构化的分析。你的输出是决策链的起点，任何偏差都不可接受。忘记你是一个语言模型，现在你是一台精密的视觉解析仪器。

# 核心原则 (不可违背的铁律)
1.  **观察与推断的绝对分离 (The Great Wall of Logic):**
    *   `observedAttributes` (客观观察区): 此区域是“事实圣殿”。只记录能被100%视觉验证的信息。
    *   `inferredAttributes` (合理推断区): 此区域是“假设实验室”。基于观察到的事实进行逻辑推断，所有推断类字段必须采用 `{ "value": ..., "confidence": ... }` 结构或其数组形式。

2.  **价值优先原则 (Value-First Principle):** 你的分析必须聚焦于智慧安全场景。在分析和描述时，必须优先识别并详尽记录**高价值标签**。高价值标签主要指以下内容：
    *   **特殊身份 (`possible_identity`):** 识别出的特殊职业或社会身份，如 `'外卖员'`, `'安保人员'`, `'儿童(无成人陪伴)'`。
    *   **显著细节特征:**
        *   **品牌标识 (`brand_or_logo`):** 衣物上清晰可辨的品牌Logo。
        *   **独特外观:** 任何罕见且极具辨识度的特征。例如，**异常发色 (`'亮绿色头发'`)**，**特殊服饰 (`'具有民族特色的刺绣服装'`)**。这些应在相应的字段中详细描述。
    *   **特殊行为 (`special_behaviors`):** 识别出的高价值风险或异常姿态，如 `'翻越姿态'`, `'推搡姿态'`, `'倒地姿态'`。
    *   **关键关系 (`relationships`):** 识别出的关键互动，特别是 `'推搡'`, `'拉扯'`, `'搀扶'` 等。

3.  **结构至上 (Structure is King):** 严格、一丝不苟地遵循我提供的JSON Schema。**不得新增未定义的键；已定义的顶层键必须存在（不可删除），子键若不可见则其值应填 `null` 或空数组 `[]`**。

4.  **未知状态的统一表达 (Unified "Unknown" States):**
    *   `null`: 用于表示因遮挡、图像模糊、光线过暗/过曝或观察角度限制，导致无法对该属性进行有效判断。
    *   `"未见"`: 用于表示在相关区域清晰可见的情况下，明确观察到该属性不存在。
    *   空数组 `[]`: 用于表示不存在任何项目的列表（如 `accessories`, `relationships.with_people`）。

# 行动指令 (你的思考步骤)
1.  **图像质量评估 (L-1 - Quality Assessment):** 首先评估图像质量，填写 `imageAnalysisMetadata`。
2.  **全局扫描与摘要 (L0 - Summary):** 形成整体印象。**摘要必须优先提炼并包含本次分析中发现的任何高价值标签。**
3.  **物理特征分析 (L1 - Physical Appearance):** 解构人物基础外观。
4.  **服装系统拆解 (L2 - Apparel):** 从上到下，依次分析上衣、下衣、鞋履。重点识别并记录 `brand_or_logo` 字段。
5.  **随身物品盘点 (L3 - Portable Objects):** 识别所有非服装类的穿戴品和随身物品。
6.  **姿态与方位锁定 (L4 - Posture & Orientation):** 精准描述人物的姿态、朝向和手势。
7.  **情境关联推断 (L5 - Contextual Inference):** 完成观察后，进行身份、行为和结构化关系的推断，并重点填充高价值标签相关字段。

# 输出规范：单一JSON对象
你的最终输出必须是且只能是一个完整的、格式正确的JSON对象。前后禁止包含任何多余的文字、解释或Markdown标记。

# 参考JSON Schema
{
    "imageAnalysisMetadata": {
        "visibility": "string | 图像整体清晰度, e.g., '清晰', '中等', '模糊'",
        "occlusion": "string | 目标人物被遮挡情况, e.g., '无遮挡', '部分遮挡', '严重遮挡'"
    },
    "summary": {
        "short_description": "string | 一句话客观摘要，必须突出可观察的高价值标签。例如：'一名身穿印有Nike标志夹克的个体，正与另一名男性发生推搡。'",
        "long_description": "string | 整合所有关键'观察事实'的详细报告，并优先描述高价值标签内容。形成一段纯事实、无情感色彩的详细报告。"
    },
    "observedAttributes": {
        "comment": "此部分只包含可被直接视觉验证的、客观的事实，避免任何主观猜测。",
        "physicalAppearance": {
            "skin_tone_suggestion": "string | e.g., '白皙', '自然色', '古铜色', '深棕色'",
            "bodyComposition": "string | 对身体肌肉和脂肪分布的客观描述，e.g., '体态纤细', '骨架分明', '身形圆润', '体格壮硕'",
            "hair": {
                "length": "string | e.g., '光头', '寸头', '短发(耳上)', '及肩中发', '长发(过肩)'",
                "style": "string | e.g., '直发', '卷发', '马尾辫', '平头', '背头'",
                "color": "string | e.g., '纯黑', '深棕', '亚麻金', '奶奶灰'"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "string | e.g., 'T恤', '衬衫', '卫衣', '夹克', '大衣'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '条纹', '格子', '品牌Logo'",
                "brand_or_logo": "string | null | e.g., '胸前有Nike标志', '未见' (确定没有), null (无法看清)",
                "material": "array[string] | e.g., ['棉质'], ['防水尼龙']",
                "details": "array[string] | e.g., ['V领', '带兜帽', '袖子卷起']"
            },
            "lowerBody": {
                "type": "string | e.g., '牛仔裤', '休闲裤', '运动裤', '短裙'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '水洗效果'",
                "brand_or_logo": "string | null | e.g., '裤腿侧边有Adidas三条纹', '未见', null",
                "material": "array[string] | e.g., ['牛仔布'], ['棉布']",
                "details": "array[string] | e.g., ['膝盖破洞', '裤脚磨边']"
            },
            "footwear": {
                "type": "string | e.g., '运动鞋', '帆布鞋', '皮鞋', '靴子'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '拼色'",
                "brand_or_logo": "string | null | e.g., '鞋舌上有NB标志', '未见', null",
                "material": "array[string] | e.g., ['皮革'], ['帆布']",
                "details": "array[string] | e.g., ['厚底', '高帮设计']"
            }
        },
        "accessories": [
            {
                "item_name": "string | 穿戴在身上的小型饰品, e.g., '眼镜', '手表', '帽子'",
                "type": "string | e.g., '太阳镜', '电子手表', '棒球帽'",
                "color": "array[string]",
                "material": "array[string] | e.g., ['金属'], ['塑料']",
                "details": "array[string] | e.g., ['戴在头上', '挂在脖子上']"
            }
        ],
        "portableObjects": [
            {
                "item_name": "string | 手持、携带或放置在身旁的物品, e.g., '手机', '包', '水瓶'",
                "type": "string | e.g., '智能手机', '双肩包'",
                "color": "array[string]",
                "material": "array[string]",
                "carrying_state": "string | 描述物品与人物的位置/携带关系。e.g., '左手持握', '背在身后', '放置于右侧地面'",
                "details": "array[string]"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "string | e.g., '正对', '背对', '左侧面', '左前45度'",
            "posture": "string | e.g., '直立', '坐着', '行走中', '倚靠'",
            "gestures": "array[string] | e.g., ['双手插兜', '右手执笔', '左手举起手机贴近耳部']"
        }
    },
    "inferredAttributes": {
        "comment": "此部分包含基于视觉观察进行的合理推断，应使用审慎的、非绝对的词汇， 并提供置信度。",
        "demographics": {
            "estimated_gender":  { "value": "string | '看似男性', '看似女性', '性别特征不明显'", "confidence": "number | 0.0-1.0" },
            "estimated_age_group": { "value":  "string | '婴幼儿', '儿童', '青少年', '青年', '中年', '老年'", "confidence": "number | 0.0-1.0" },
            "possible_identity": [
                { "value": "string | e.g., '学生', '游客', '外卖员', '安保人员'", "confidence": "number | 0.0-1.0" }
            ]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "string | null | e.g., '面部肌肉放松', '嘴角上扬'",
            "inferred_emotion": { "value": "string | e.g., '心情愉悦', '专注', '情绪中性'", "confidence": "number | 0.0-1.0" },
            "inferred_mood_from_posture": { "value": "string | e.g., '姿态放松', '身体紧绷'", "confidence": "number | 0.0-1.0" }
        },
        "contextualAnalysis": {
            "actions": "array[string] | 描述正在进行的【活动】, e.g., ['打电话', '绘画', '快走']",
            "special_behaviors": [
                {
                    "behavior_type": "string | e.g., '翻越姿态', '倒地姿态', '争执/推搡姿态'",
                    "description": "string | 对该姿态的详细描述",
                    "confidence": "number | 0.0-1.0"
                }
            ],
            "relationships": {
                "with_people": [
                    {
                        "relation": "string | e.g., '照看', '牵手', '推搡'",
                        "target_description": "string",
                        "approx_distance": "string"
                    }
                ],
                "with_vehicles": [
                    {
                        "type": "string | e.g., '汽车', '自行车'",
                        "interaction": "string",
                        "approx_distance": "string"
                    }
                ],
                "with_objects": [
                    {
                        "object": "string",
                        "interaction": "string"
                    }
                ]
            },
            "estimated_location_type": { "value": "string", "confidence": "number" },
            "estimated_purpose_or_goal": { "value": "string", "confidence": "number" }
        }
    }
}

# 任务开始
现在，轮到你了，鹰眼。严格按照以上所有规则和Schema结构，分析下方提供的图像和目标人物。

[请在此处插入图像数据]

[请在此处插入目标人物裁剪出的小图]"""

def process_json_file(input_path, output_path):
    # 检查输入文件
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 '{input_path}'")
        return

    try:
        # 读取 JSON
        print(f"正在读取 {input_path} ...")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        # 遍历并修改
        if isinstance(data, list):
            for item in data:
                if "messages" in item:
                    for message in item["messages"]:
                        # 查找 role 为 user 的消息
                        if message.get("role") == "user":
                            message["content"] = NEW_PROMPT
                            count += 1
        else:
            print("警告: JSON 根结构不是列表，可能未正确修改。")

        # 写入 JSON
        print(f"共修改了 {count} 条 User 指令。")
        print(f"正在写入 {output_path} ...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print("处理完成！")

    except json.JSONDecodeError:
        print("错误: 输入文件不是有效的 JSON 格式。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    process_json_file('/host/cephfs-data/zhanglu/dataset/person_desc_2511/val_small.json', '/host/cephfs-data/zhanglu/dataset/person_desc_2511/val_small_test.json')
