import argparse
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- 配置 ---
BOX_COLOR = "red"
TEXT_COLOR = "white"
TEXT_BG_COLOR = "red"

def get_validity(info_list):
    """从 result.info 列表中判断数据是否有效"""
    for item in info_list:
        if item.get("header") == "数据是否有效":
            return item.get("value") == "Yes"
    return False

def construct_image_path(base_dir, image_url, category_name):
    """根据规则构建图片的完整源路径"""
    try:
        if ".zip/" in image_url:
            path_suffix = image_url.split(".zip/", 1)[1]
            full_path1 = base_dir / path_suffix
            if full_path1.exists():
                return full_path1
            parts = Path(path_suffix).parts
            if len(parts) > 1:
                full_path2 = base_dir / Path(*parts[1:])
                if full_path2.exists(): return full_path2
                full_path3 = base_dir / category_name / Path(*parts[1:])
                if full_path3.exists(): return full_path3
            return full_path1
        else:
            print(f"  [警告] image_url '{image_url}' 中不包含 '.zip/' 分隔符。")
            return None
    except Exception:
        print(f"  [错误] 从 image_url '{image_url}' 中解析路径时失败。")
        return None

def visualize_image(src_image_path, annotations, output_path, font_path):
    """在图片上绘制标注框和标签，并保存"""
    try:
        image = Image.open(src_image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(font_path, size=20)
        except IOError:
            print(f"  [警告] 无法加载字体 '{font_path}'，将使用默认字体。")
            font = ImageFont.load_default()

        for ann in annotations:
            if ann.get("type") != "rect" or "coordinate" not in ann:
                continue
            coords = ann.get("coordinate")
            if not coords: continue
            x_coords = [p['x'] for p in coords]
            y_coords = [p['y'] for p in coords]
            box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            label_text = ann.get("label", ["未知"])[0]
            draw.rectangle(box, outline=BOX_COLOR, width=3)
            text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
            draw.rectangle([box[0], box[1], box[0] + (text_bbox[2] - text_bbox[0]) + 4, box[1] + (text_bbox[3] - text_bbox[1]) + 4], fill=TEXT_BG_COLOR)
            draw.text((box[0] + 2, box[1] + 2), label_text, fill=TEXT_COLOR, font=font)
        image.save(output_path)
    except Exception as e:
        print(f"  [错误] 可视化图片 '{src_image_path}' 失败: {e}")

def process_files_and_update_stats(category_path, image_base_dir, output_base_dir, font_path, all_stats):
    """
    处理单个文件夹内的JSON文件，进行文件操作，并更新全局统计数据。
    统计逻辑已变更为基于标签，而非文件夹。
    """
    category_name = category_path.name
    print(f"\n--- 正在扫描文件夹: {category_name} ---")

    output_paths = {
        "无效": output_base_dir / category_name / "无效",
        "有效": output_base_dir / category_name / "有效",
        "有框": output_base_dir / category_name / "有效_可视化" / "有框",
        "无框": output_base_dir / category_name / "有效_可视化" / "无框"
    }
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(category_path.glob("*.json")))
    if not json_files:
        print(f"在 '{category_path}' 中未找到 JSON 文件。")
        return

    for json_file in tqdm(json_files, desc=f"扫描 {category_name}"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        except Exception as e:
            print(f"  [警告] 解析JSON文件 '{json_file.name}' 失败: {e}，跳过。")
            continue
        
        for item in data_list:
            src_image_path_str = item.get("data", {}).get("image_url")
            result = item.get("result", {})
            if not src_image_path_str or not result: continue

            src_image_path = construct_image_path(image_base_dir, src_image_path_str, category_name)
            if not src_image_path or not src_image_path.exists():
                print(f"  [警告] 源图片不存在: {src_image_path}，跳过。")
                continue

            output_img_name = src_image_path.name
            output_json_name = src_image_path.with_suffix(".json").name
            is_valid = get_validity(result.get("info", []))
            
            if not is_valid:
                dest_dir = output_paths["无效"]
                shutil.copy2(src_image_path, dest_dir / output_img_name)
                with open(dest_dir / output_json_name, 'w', encoding='utf-8') as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
            else:
                dest_dir_valid = output_paths["有效"]
                shutil.copy2(src_image_path, dest_dir_valid / output_img_name)
                with open(dest_dir_valid / output_json_name, 'w', encoding='utf-8') as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)

                annotations = result.get("data", [])
                
                # --- 新的统计逻辑 ---
                for ann in annotations:
                    label_list = ann.get("catetips")
                    if not label_list: continue
                    label_name = label_list[0]
                    
                    # 如果是新标签，则在统计字典中初始化
                    if label_name not in all_stats:
                        all_stats[label_name] = {"bbox_count": 0, "image_paths": set()}
                    
                    # 更新统计数据
                    all_stats[label_name]["bbox_count"] += 1
                    all_stats[label_name]["image_paths"].add(src_image_path)

                # 可视化逻辑保持不变
                if annotations:
                    dest_dir_viz = output_paths["有框"]
                    visualize_image(src_image_path, annotations, dest_dir_viz / output_img_name, font_path)
                else:
                    dest_dir_viz = output_paths["无框"]
                    shutil.copy2(src_image_path, dest_dir_viz / output_img_name)

def main():
    parser = argparse.ArgumentParser(description="处理图像数据集，根据标签分类并进行可视化。")
    parser.add_argument("--image_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类数据标注", help="存放所有图片的根目录路径。")
    parser.add_argument("--label_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果", help="存放标签的根目录，其下应为'垃圾堆积'等子文件夹。")
    parser.add_argument("--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_可视化", help="处理后数据的输出根目录。")
    parser.add_argument("--font_path", type=str, default="/host/ssddata/disk2/zhanglu/SimHei.ttf", help="用于可视化的中文字体文件路径 (例如 /path/to/simhei.ttf)。")
   
    args = parser.parse_args()

    image_base_dir = Path(args.image_dir)
    label_base_dir = Path(args.label_dir)
    output_base_dir = Path(args.output_dir)
    font_path = args.font_path

    if not image_base_dir.is_dir() or not label_base_dir.is_dir():
        print("错误：输入的图片目录或标签目录不存在！")
        return

    category_dirs = [d for d in label_base_dir.iterdir() if d.is_dir()]
    
    # 全局统计字典，键为标签名
    all_stats = {}

    for category_path in category_dirs:
        process_files_and_update_stats(
            category_path, image_base_dir, output_base_dir, font_path, all_stats
        )
    
    # --- 打印最终的、基于标签的汇总信息 ---
    print("\n" + "="*60)
    print("按[标签类别]统计汇总")
    print("="*60)

    if not all_stats:
        print("未找到任何有效的目标框标签。")
    else:
        grand_total_bboxes = 0
        
        print(f"{'标签名称':<20} | {'目标框总数':<12} | {'独立图片数'}")
        print("-" * 60)
        
        # 按标签名排序，使输出更整洁
        for label, stats in sorted(all_stats.items()):
            bbox_count = stats["bbox_count"]
            # 通过集合的大小获取独立图片数
            image_count = len(stats["image_paths"])
            
            print(f"- {label:<18} : {bbox_count:<12} : {image_count}")
            
            grand_total_bboxes += bbox_count
            
        print("-" * 60)
        print(f"{'所有标签目标框总计':<35}: {grand_total_bboxes}")
    
    print("="*60)
    print("\n所有处理完成！")


if __name__ == "__main__":
    main()
