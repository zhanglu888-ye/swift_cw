import gradio as gr
import json
import os
import shutil
from collections import defaultdict

# --- 核心功能函数 ---
def get_event_categories(annotation_path):
    """从标注路径获取所有事件类别（即子文件夹名称）"""
    # --- 新增的调试代码 ---
    print(f"[DEBUG] 正在尝试从以下路径获取事件类别: '{annotation_path}'")
    
    if not os.path.isdir(annotation_path):
        print(f"[DEBUG] 错误：该路径 '{annotation_path}' 不是一个有效的目录或不存在。")
        return []
    
    try:
        print(f"[DEBUG] 目录 '{annotation_path}' 存在，正在列出其中的内容...")
        all_items = os.listdir(annotation_path)
        print(f"[DEBUG] 目录下的所有项目: {all_items}")
        
        categories = [d for d in all_items if os.path.isdir(os.path.join(annotation_path, d)) and not d.startswith('.')]
        print(f"[DEBUG] 成功找到的事件类别 (子目录): {categories}")
        # --- 调试代码结束 ---
        
        return sorted(categories)
    except PermissionError:
        print(f"[DEBUG] 错误：权限不足，无法读取目录 '{annotation_path}'。")
        return []
    except Exception as e:
        print(f"[DEBUG] 发生了未知错误: {e}")
        return []
def get_risk_levels(annotation_path):
    """扫描一部分JSON文件，获取所有出现过的风险等级"""
    if not os.path.isdir(annotation_path):
        return ["(请先提供有效的标注路径)"]
    
    levels = set()
    categories = get_event_categories(annotation_path)
    # 只扫描每个类别下的少量文件以提高效率
    for category in categories:
        category_path = os.path.join(annotation_path, category)
        try:
            json_files = [f for f in os.listdir(category_path) if f.endswith('.json')]
            for i, filename in enumerate(json_files):
                if i >= 20:  # 每个类别最多检查20个文件
                    break
                try:
                    with open(os.path.join(category_path, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        level = data.get("analyses", [{}])[0].get("event_analyses", [{}])[0].get("comprehensive_assessment", {}).get("风险等级")
                        if level:
                            levels.add(level)
                except Exception:
                    continue
        except Exception:
            continue
            
    if not levels:
        return ["(未找到风险等级)"]
        
    return sorted(list(levels))

def update_dropdowns(annotation_path):
    """当标注路径改变时，更新事件类别和风险等级的下拉菜单"""
    categories = get_event_categories(annotation_path)
    levels = get_risk_levels(annotation_path)
    return gr.Dropdown(choices=categories, value=categories[0] if categories else None, interactive=True), \
           gr.Dropdown(choices=levels, value=levels[0] if levels else None, interactive=True)


'''
def filter_data(image_path, annotation_path, deleted_data_path, event_category, risk_level):
    """根据选择的类别和风险等级筛选数据"""
    if not all([image_path, annotation_path, deleted_data_path, event_category, risk_level]):
        return [], 0, None, "请填写所有路径并选择筛选条件。", "---", []

    if not os.path.isdir(deleted_data_path):
        try:
            os.makedirs(deleted_data_path)
        except OSError as e:
            return [], 0, None, f"创建删除目录失败: {e}", "---", []

    category_path = os.path.join(annotation_path, event_category)
    if not os.path.isdir(category_path):
        return [], 0, None, f"事件类别目录不存在: {category_path}", "---", []

    filtered_files = []
    json_files = [f for f in os.listdir(category_path) if f.endswith('.json')]

    for filename in json_files:
        json_file_path = os.path.join(category_path, filename)
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            current_risk_level = data.get("analyses", [{}])[0].get("event_analyses", [{}])[0].get("comprehensive_assessment", {}).get("风险等级")
            
            if current_risk_level == risk_level:
                image_filename = data.get("file_name")
                if image_filename:
                    # 图片可能在根目录，也可能在以事件类别命名的子目录
                    # 优先检查根目录
                    current_image_path = os.path.join(image_path, image_filename)
                    if not os.path.exists(current_image_path):
                        # 如果根目录没有，则尝试在事件子目录中查找
                        current_image_path = os.path.join(image_path, event_category, image_filename)

                    if os.path.exists(current_image_path):
                        filtered_files.append({
                            "image_path": current_image_path,
                            "json_path": json_file_path,
                            "data": data
                        })
        except Exception as e:
            print(f"处理文件 {json_file_path} 时出错: {e}")
            continue

    if not filtered_files:
        return [], 0, None, "没有找到符合条件的数据。", "---", []

    # 初始化状态
    initial_index = 0
    total = len(filtered_files)
    
    # 显示第一条数据
    image_to_show, text_to_show, progress_text = get_display_data(filtered_files, initial_index)

    # 返回给UI组件和状态
    return filtered_files, initial_index, image_to_show, text_to_show, progress_text, []
'''


# --- 请用这个新函数替换掉你代码里原来的 filter_data 函数 ---

def filter_data(image_path, annotation_path, deleted_data_path, event_category, risk_level):
    """根据选择的类别和风险等级筛选数据"""
    if not all([image_path, annotation_path, deleted_data_path, event_category, risk_level]):
        return [], 0, None, "请填写所有路径并选择筛选条件。", "---", []

    # ... (前面的代码保持不变) ...
    if not os.path.isdir(deleted_data_path):
        try:
            os.makedirs(deleted_data_path)
        except OSError as e:
            return [], 0, None, f"创建删除目录失败: {e}", "---", []

    category_path = os.path.join(annotation_path, event_category)
    if not os.path.isdir(category_path):
        return [], 0, None, f"事件类别目录不存在: {category_path}", "---", []

    filtered_files = []
    json_files = [f for f in os.listdir(category_path) if f.endswith('.json')]

    print("\n" + "="*80)
    print(f"[FILTERING] 开始筛选... 事件类别: '{event_category}', 风险等级: '{risk_level}'")
    print(f"[FILTERING] 将在目录 '{category_path}' 中查找 {len(json_files)} 个JSON文件。")
    print("="*80)

    for i, filename in enumerate(json_files):
        json_file_path = os.path.join(category_path, filename)
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            current_risk_level = data.get("analyses", [{}])[0].get("event_analyses", [{}])[0].get("comprehensive_assessment", {}).get("风险等级")

            # --- 新增的详细调试日志 ---
            if i < 10: # 只打印前10个文件的详细日志，避免刷屏
                print("-" * 50)
                print(f"[DEBUG] 正在处理文件: {filename}")
                print(f"[DEBUG] 用户选择的风险等级: '{risk_level}'")
                print(f"[DEBUG] 从JSON中找到的风险等级: '{current_risk_level}'")

            if current_risk_level and current_risk_level.strip() == risk_level.strip():
                if i < 10: print("[DEBUG] >> 风险等级匹配成功! << 正在检查图片文件...")

                image_filename = data.get("file_name")
                if image_filename:
                    # 检查两个可能的图片路径
                    path_option1 = os.path.join(image_path, image_filename)
                    path_option2 = os.path.join(image_path, event_category, image_filename)

                    current_image_path = None
                    if os.path.exists(path_option1):
                        current_image_path = path_option1
                        if i < 10: print(f"[DEBUG] 在路径1找到图片: '{path_option1}'")
                    elif os.path.exists(path_option2):
                        current_image_path = path_option2
                        if i < 10: print(f"[DEBUG] 在路径2找到图片: '{path_option2}'")

                    if current_image_path:
                        if i < 10: print("[DEBUG] >> 图片存在，此条数据有效! <<")
                        filtered_files.append({
                            "image_path": current_image_path,
                            "json_path": json_file_path,
                            "data": data
                        })
                    else:
                        if i < 10: print(f"[DEBUG] !! 警告: 找不到图片文件 '{image_filename}'。已跳过。")
                else:
                    if i < 10: print("[DEBUG] !! 警告: JSON中没有'file_name'字段。已跳过。")
            else:
                if i < 10: print("[DEBUG] >> 风险等级不匹配。已跳过。")
            # --- 调试日志结束 ---

        except Exception as e:
            print(f"[ERROR] 处理文件 {json_file_path} 时出错: {e}")
            continue

    print("="*80)
    print(f"[FILTERING] 筛选完成。总共找到 {len(filtered_files)} 条符合条件的数据。")
    print("="*80 + "\n")

    # ... (后面的代码保持不变) ...
    if not filtered_files:
        return [], 0, None, "没有找到符合条件的数据。", "---", []

    initial_index = 0
    image_to_show, text_to_show, progress_text = get_display_data(filtered_files, initial_index)
    return filtered_files, initial_index, image_to_show, text_to_show, progress_text, []

def get_display_data(filtered_files_state, current_index_state):
    """根据当前索引获取要显示的数据"""
    if not filtered_files_state or current_index_state < 0 or current_index_state >= len(filtered_files_state):
        return None, "没有更多数据了。", f"0 / {len(filtered_files_state)}"
    
    item = filtered_files_state[current_index_state]
    image_path = item["image_path"]
    data = item["data"]

    # 格式化JSON内容以供显示
    try:
        event_analysis = data["analyses"][0]["event_analyses"][0]
        simplified = event_analysis["simplified_cot"]
        detailed = event_analysis["detailed_cot"]
        assessment = event_analysis["comprehensive_assessment"]

        md_output = f"""
        ### 综合评估
        - **异常类型**: {assessment.get('异常类型', 'N/A')}
        - **风险等级**: {assessment.get('风险等级', 'N/A')}
        
        ### 简要分析 (Simplified CoT)
        - **画面分析**: {simplified.get('画面分析', 'N/A')}
        - **推理过程**: {simplified.get('推理过程', 'N/A')}
        
        ### 详细分析 (Detailed CoT)
        - **画面分析**: {detailed.get('画面分析', 'N/A')}
        - **推理过程**: {detailed.get('推理过程', 'N/A')}
        
        ---
        **图片文件**: `{os.path.basename(image_path)}`
        """
    except (KeyError, IndexError):
        md_output = "JSON 结构不符合预期，无法解析。\n\n" + json.dumps(data, indent=2, ensure_ascii=False)

    progress_text = f"进度: {current_index_state + 1} / {len(filtered_files_state)}"
    
    return image_path, md_output, progress_text

def navigate(direction, filtered_files_state, current_index_state):
    """处理上一张/下一张的导航"""
    new_index = current_index_state + direction
    if 0 <= new_index < len(filtered_files_state):
        current_index_state = new_index
    
    image, text, progress = get_display_data(filtered_files_state, current_index_state)
    return filtered_files_state, current_index_state, image, text, progress

def delete_current_file(filtered_files_state, current_index_state, deleted_data_path, deleted_log_state):
    """删除当前文件并移动到指定目录 (只移动JSON)"""
    if not filtered_files_state or not (0 <= current_index_state < len(filtered_files_state)):
        return filtered_files_state, current_index_state, None, "没有可删除的文件。", "---", deleted_log_state, gr.Markdown(value="`无删除记录`")

    item_to_delete = filtered_files_state[current_index_state]
    json_path = item_to_delete["json_path"]
    
    # 获取事件类别用于创建子目录
    event_category = os.path.basename(os.path.dirname(json_path))
    
    # 确定目标路径
    dest_dir = os.path.join(deleted_data_path, event_category)
    os.makedirs(dest_dir, exist_ok=True)
    
    dest_json_path = os.path.join(dest_dir, os.path.basename(json_path))

    try:
        # 只移动JSON标注文件，保留原始图片
        shutil.move(json_path, dest_json_path)
        
        # 更新删除日志
        log_message = f"已删除标注: {os.path.basename(json_path)} (类别: {event_category})"
        deleted_log_state.append(log_message)
        
        # 从当前列表中移除已删除项
        del filtered_files_state[current_index_state]
        
        # 调整索引，确保不会越界
        if current_index_state >= len(filtered_files_state) and filtered_files_state:
            current_index_state = len(filtered_files_state) - 1
            
    except Exception as e:
        # 如果移动失败，则不更新状态，并显示错误
        error_message = f"删除失败: {e}"
        image, text, progress = get_display_data(filtered_files_state, current_index_state)
        # 在删除日志中显示错误
        deleted_log_state.append(f"失败: {error_message}")
        return filtered_files_state, current_index_state, image, text, progress, deleted_log_state, gr.Markdown(value="\n".join(f"- {log}" for log in deleted_log_state))

    # 更新显示
    image, text, progress = get_display_data(filtered_files_state, current_index_state)
    
    # 格式化删除日志
    deleted_log_md = "\n".join(f"- {log}" for log in reversed(deleted_log_state))
    
    return filtered_files_state, current_index_state, image, text, progress, deleted_log_state, gr.Markdown(value=deleted_log_md)


# --- Gradio 界面定义 ---

with gr.Blocks(theme=gr.themes.Soft(), title="数据清洗工具") as demo:
    # 状态管理
    filtered_files_state = gr.State([])
    current_index_state = gr.State(0)
    deleted_log_state = gr.State([])

    gr.Markdown("# 模型标注结果人工清洗工具")
    gr.Markdown("请提供数据路径，选择事件类别和风险等级，然后开始清洗数据。")

    with gr.Row():
        image_path_input = gr.Textbox(label="1. 输入图片根目录路径", placeholder="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并_deduplicated")
        annotation_path_input = gr.Textbox(label="2. 输入标注结果根目录路径", placeholder="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述_正样本")
        deleted_data_path_input = gr.Textbox(label="3. 输入删除数据存放路径", placeholder="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/train_data/分类and描述_正样本_删除")

    with gr.Row():
        event_category_dropdown = gr.Dropdown(label="4. 选择事件类别", interactive=False)
        risk_level_dropdown = gr.Dropdown(label="5. 选择风险等级", interactive=False)
        filter_button = gr.Button("开始筛选", variant="primary")
    
    # 当标注路径输入框内容改变时，自动更新下拉菜单
    annotation_path_input.change(
        fn=update_dropdowns,
        inputs=[annotation_path_input],
        outputs=[event_category_dropdown, risk_level_dropdown]
    )

    gr.HTML("<hr>") # 分割线

    with gr.Row():
        with gr.Column(scale=1):
            image_display = gr.Image(label="图片预览", type="filepath")
            progress_label = gr.Label(label="进度")
        with gr.Column(scale=1):
            json_display = gr.Markdown(label="标注结果")

    with gr.Row():
        prev_button = gr.Button("上一张")
        delete_button = gr.Button("删除标注", variant="stop") # 按钮文字稍作修改
        correct_button = gr.Button("结果正确 (保留并下一张)", variant="primary")
    
    with gr.Accordion("查看已删除标注记录", open=False):
        deleted_files_display = gr.Markdown("`无删除记录`")

    # --- 事件绑定 ---

    # 点击“开始筛选”按钮
    filter_button.click(
        fn=filter_data,
        inputs=[image_path_input, annotation_path_input, deleted_data_path_input, event_category_dropdown, risk_level_dropdown],
        outputs=[filtered_files_state, current_index_state, image_display, json_display, progress_label, deleted_log_state]
    )

    # 点击“上一张”
    prev_button.click(
        fn=lambda fs, ci: navigate(-1, fs, ci),
        inputs=[filtered_files_state, current_index_state],
        outputs=[filtered_files_state, current_index_state, image_display, json_display, progress_label]
    )
    
    # 点击“结果正确” (功能等同于下一张)
    correct_button.click(
        fn=lambda fs, ci: navigate(1, fs, ci),
        inputs=[filtered_files_state, current_index_state],
        outputs=[filtered_files_state, current_index_state, image_display, json_display, progress_label]
    )
    
    # 点击“删除”
    delete_button.click(
        fn=delete_current_file,
        inputs=[filtered_files_state, current_index_state, deleted_data_path_input, deleted_log_state],
        outputs=[filtered_files_state, current_index_state, image_display, json_display, progress_label, deleted_log_state, deleted_files_display]
    )


if __name__ == "__main__":
    image_base_path = "/host/ssddata/disk2/zhanglu/clip-data/datasets"
    demo.launch()
