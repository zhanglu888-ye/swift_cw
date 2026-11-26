# 文件名: prepare_vlm_input.py
import argparse
import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def sanitize_filename(name: str) -> str:
    """移除或替换文件名中的非法字符。"""
    return name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')

def get_validity(info_list: list) -> bool:
    """从 result.info 列表中判断数据是否有效。"""
    for item in info_list:
        if item.get("header") == "数据是否有效":
            return item.get("value") == "Yes"
    return False

def construct_image_path(base_dir: Path, image_url: str, category_name: str) -> Path:
    """根据规则构建图片的完整源路径。"""
    try:
        if ".zip/" in image_url:
            path_suffix = image_url.split(".zip/", 1)[1]
            # 尝试多种可能的路径拼接方式
            paths_to_try = [
                base_dir / path_suffix,
                base_dir / Path(*Path(path_suffix).parts[1:]),
                base_dir / category_name / Path(*Path(path_suffix).parts[1:])
            ]
            for p in paths_to_try:
                if p.exists():
                    return p
            return paths_to_try[0] # 返回最可能的一个以便于报错
        return None
    except Exception:
        return None


def process_raw_dataset(image_dir: Path, label_dir: Path, output_dir: Path):
    """
    从原始数据处理、筛选、抠图，并生成VLM脚本所需的输入。
    """
    large_images_output_dir = output_dir / "large_images"
    cropped_data_output_dir = output_dir / "cropped_data"
    large_images_output_dir.mkdir(parents=True, exist_ok=True)
    cropped_data_output_dir.mkdir(parents=True, exist_ok=True)

    copied_large_images = set()

    for category_dir in label_dir.iterdir():
        if not category_dir.is_dir(): continue
        
        logging.info(f"--- 正在扫描原始标签文件夹: {category_dir.name} ---")
        json_files = list(category_dir.glob("*.json"))

        for json_path in tqdm(json_files, desc=f"处理 {category_dir.name}"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
            except Exception as e:
                logging.warning(f"无法读取或解析 {json_path.name}: {e}")
                continue

            for item in data_list:
                result = item.get("result", {})
                if not get_validity(result.get("info", [])):
                    continue

                image_url = item.get("data", {}).get("image_url")
                if not image_url: continue

                src_image_path = construct_image_path(image_dir, image_url, category_dir.name)
                if not src_image_path or not src_image_path.exists():
                    logging.warning(f"找不到源图片 {src_image_path}，跳过。")
                    continue

                if src_image_path not in copied_large_images:
                    shutil.copy2(src_image_path, large_images_output_dir)
                    copied_large_images.add(src_image_path)

                case_name = src_image_path.stem
                case_output_dir = cropped_data_output_dir / case_name
                annotations = result.get("data", [])
                
                if not annotations: continue

                with Image.open(src_image_path) as img:
                    crop_counters = {}
                    for ann in annotations:
                        if ann.get("type") != "rect": continue
                        coords = ann.get("coordinate")
                        labels = ann.get("catetips")
                        if not coords or not labels: continue

                        label_name = sanitize_filename(labels[0])
                        x_coords = [p['x'] for p in coords]
                        y_coords = [p['y'] for p in coords]
                        box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                        cropped_image = img.crop(box)
                        
                        category_output_dir = case_output_dir / label_name
                        category_output_dir.mkdir(parents=True, exist_ok=True)

                        crop_index = crop_counters.get(label_name, 0) + 1
                        crop_counters[label_name] = crop_index
                        
                        # =============================================================
                        # <<< FIX: 在保存为JPEG前，将图像转换为RGB模式 >>>
                        # 这会移除RGBA模式中的Alpha（透明度）通道，解决错误。
                        if cropped_image.mode != 'RGB':
                            cropped_image = cropped_image.convert('RGB')
                        # =============================================================

                        # 保存抠图
                        crop_img_path = category_output_dir / f"crop_{crop_index}.jpg"
                        cropped_image.save(crop_img_path, "JPEG")

                        # 保存Bbox元数据
                        meta_path = category_output_dir / f"crop_{crop_index}.json"
                        meta_data = {
                            "original_bbox": [round(c, 2) for c in box],
                            "category_name": labels[0]
                        }
                        with open(meta_path, 'w', encoding='utf-8') as f:
                            json.dump(meta_data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="从原始数据集准备VLM描述脚本的输入。")
    parser.add_argument("--image_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类数据标注", help="原始图片的根目录。")
    parser.add_argument("--label_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果", help="原始标签JSON的根目录。")
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare", help="VLM输入数据的输出根目录。")
    args = parser.parse_args()

    process_raw_dataset(Path(args.image_dir), Path(args.label_dir), Path(args.output_dir))
    logging.info("="*30)
    logging.info("VLM输入数据准备完成！")
    logging.info(f"输出目录: {args.output_dir}")
    logging.info("="*30)

if __name__ == "__main__":
    main()