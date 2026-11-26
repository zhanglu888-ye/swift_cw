import os
import json
import shutil
import uuid
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, abort

app = Flask(__name__)
app.secret_key = 'a_very_secret_random_string_for_session'

USER_SESSIONS = {}

# --- Helper Functions (No changes needed here) ---
def scan_for_categories_and_levels(annotation_path):
    categories, levels = [], set()
    if not os.path.isdir(annotation_path): return {"categories": [], "levels": []}
    try:
        categories = sorted([d for d in os.listdir(annotation_path) if os.path.isdir(os.path.join(annotation_path, d)) and not d.startswith('.')])
        for category in categories:
            category_path = os.path.join(annotation_path, category)
            for i, filename in enumerate(f for f in os.listdir(category_path) if f.endswith('.json')):
                if i >= 20: break
                try:
                    with open(os.path.join(category_path, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        level = data.get("analyses", [{}])[0].get("event_analyses", [{}])[0].get("comprehensive_assessment", {}).get("风险等级")
                        if level: levels.add(level)
                except Exception: continue
    except Exception: return {"categories": [], "levels": []}
    return {"categories": categories, "levels": sorted(list(levels))}

def filter_data_logic(image_path, annotation_path, event_category, risk_level):
    category_path = os.path.join(annotation_path, event_category)
    filtered_files = []
    if not os.path.isdir(category_path): return []
    for filename in os.listdir(category_path):
        if not filename.endswith('.json'): continue
        json_file_path = os.path.join(category_path, filename)
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            current_risk_level = data.get("analyses", [{}])[0].get("event_analyses", [{}])[0].get("comprehensive_assessment", {}).get("风险等级")
            if current_risk_level and current_risk_level.strip() == risk_level.strip():
                image_filename = data.get("file_name")
                if not image_filename: continue
                possible_image_path = os.path.join(image_path, image_filename)
                if not os.path.exists(possible_image_path): possible_image_path = os.path.join(image_path, event_category, image_filename)
                if os.path.exists(possible_image_path):
                    filtered_files.append({"image_path": possible_image_path, "json_path": json_file_path, "data": data})
        except Exception as e: print(f"Error processing {json_file_path}: {e}")
    return filtered_files

# --- Flask Routes (Main changes are here) ---

@app.route('/', methods=['GET'])
def index():
    user_id = session.get('user_session_id')
    user_data = USER_SESSIONS.get(user_id, {})
    
    # 获取所有需要恢复的状态
    paths = user_data.get('paths', {})
    selection = user_data.get('selection', {})
    review_data = user_data.get('review_data', [])
    current_index = user_data.get('current_index', 0)
    deleted_log = user_data.get('deleted_log', [])
    
    current_item, formatted_json_str = None, ""
    if review_data and 0 <= current_index < len(review_data):
        current_item = review_data[current_index]
        formatted_json_str = json.dumps(current_item['data'], indent=2, ensure_ascii=False)

    return render_template('index.html', 
                           current_item=current_item,
                           formatted_json=formatted_json_str,
                           current_index=current_index,
                           total_items=len(review_data),
                           deleted_log=list(reversed(deleted_log)),
                           paths=paths,
                           selection=selection) # 传递用户的选择

@app.route('/scan', methods=['POST'])
def scan_path():
    path = request.json.get('path')
    if not path: return jsonify({"error": "Path is required"}), 400
    return jsonify(scan_for_categories_and_levels(path))

@app.route('/start_review', methods=['POST'])
def start_review():
    user_id = session.get('user_session_id')
    if not user_id:
        user_id = str(uuid.uuid4())
        session['user_session_id'] = user_id
    
    image_path = request.form.get('image_path')
    annotation_path = request.form.get('annotation_path')
    deleted_path = request.form.get('deleted_path')
    event_category = request.form.get('event_category')
    risk_level = request.form.get('risk_level')
    
    filtered_data = filter_data_logic(image_path, annotation_path, event_category, risk_level)
    
    USER_SESSIONS[user_id] = {
        'paths': {'image': image_path, 'annotation': annotation_path, 'deleted': deleted_path},
        'selection': {'category': event_category, 'level': risk_level}, # 保存用户的选择
        'review_data': filtered_data,
        'current_index': 0,
        'deleted_log': USER_SESSIONS.get(user_id, {}).get('deleted_log', []) # 保留已有的删除记录
    }
    return redirect(url_for('index'))

# --- Navigation and Delete Routes (No major changes) ---
@app.route('/navigate', methods=['POST'])
def navigate():
    user_id = session.get('user_session_id')
    if user_id not in USER_SESSIONS: return redirect(url_for('index'))
    direction = int(request.form.get('direction', 1))
    current_index = USER_SESSIONS[user_id]['current_index']
    total_items = len(USER_SESSIONS[user_id]['review_data'])
    new_index = current_index + direction
    if 0 <= new_index < total_items: USER_SESSIONS[user_id]['current_index'] = new_index
    return redirect(url_for('index'))

@app.route('/delete', methods=['POST'])
def delete_item():
    user_id = session.get('user_session_id')
    if user_id not in USER_SESSIONS: return redirect(url_for('index'))
    user_data, current_index = USER_SESSIONS[user_id], USER_SESSIONS[user_id]['current_index']
    review_data, deleted_path = user_data['review_data'], user_data['paths']['deleted']
    if not all([review_data, 0 <= current_index < len(review_data), deleted_path]): return redirect(url_for('index'))
    item_to_delete = review_data[current_index]
    json_path = item_to_delete['json_path']
    event_category = os.path.basename(os.path.dirname(json_path))
    dest_dir = os.path.join(deleted_path, event_category)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        shutil.move(json_path, os.path.join(dest_dir, os.path.basename(json_path)))
        user_data['deleted_log'].append(f"已删除标注: {os.path.basename(json_path)} (类别: {event_category})")
        review_data.pop(current_index)
        if current_index >= len(review_data) and review_data: user_data['current_index'] = len(review_data) - 1
    except Exception as e: user_data['deleted_log'].append(f"删除失败: {e}")
    return redirect(url_for('index'))

@app.route('/get_image')
def get_image():
    user_id, image_path = session.get('user_session_id'), request.args.get('path')
    if not user_id or user_id not in USER_SESSIONS: abort(403)
    image_root = USER_SESSIONS[user_id]['paths']['image']
    if not image_path or not image_root or not os.path.abspath(image_path).startswith(os.path.abspath(image_root)): abort(404)
    try: return send_file(image_path)
    except FileNotFoundError: abort(404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
