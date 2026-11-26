# 文件名: inject_bbox_postprocess_cn.py
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import demjson3 as demjson # 导入用于修复JSON的库

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

EVENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "垃圾事件": {"categories": {"垃圾桶满溢", "垃圾堆积"}},
    "倒地事件": {"categories": {"人员异常倒地"}},
    "占道经营事件": {"categories": {"流动摊贩", "出店经营"}},
    "人群聚集事件": {"categories": {"异质文化异常聚集", "开放空间异常聚集"}}
}

def build_bbox_map(cropped_data_dir: Path) -> Dict[str, Dict[str, List[List[float]]]]:
    logging.info("第1步: 正在从抠图元数据构建Bbox映射表...")
    bbox_map = {}
    category_to_event_map = {cat: event for event, details in EVENT_CONFIG.items() for cat in details["categories"]}

    for case_dir in tqdm([d for d in cropped_data_dir.iterdir() if d.is_dir()], desc="扫描案例以构建Bbox映射"):
        case_name = case_dir.name
        bbox_map[case_name] = {}
        for category_dir in case_dir.iterdir():
            if not category_dir.is_dir(): continue
            for meta_path in category_dir.glob('*.json'):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    category_name = meta_data.get("category_name")
                    bbox = meta_data.get("original_bbox")
                    event_type = category_to_event_map.get(category_name)
                    if all([category_name, bbox, event_type]):
                        if event_type not in bbox_map[case_name]:
                            bbox_map[case_name][event_type] = []
                        bbox_map[case_name][event_type].append(bbox)
                except Exception as e:
                    logging.warning(f"处理元数据文件 {meta_path} 时出错: {e}")
    logging.info(f"Bbox映射表构建完成，共包含 {len(bbox_map)} 个案例。")
    return bbox_map

def inject_bboxes_into_captions(caption_dir: Path, output_dir: Path, bbox_map: Dict):
    logging.info("第2步: 正在向Caption JSON文件注入Bbox（包含自动修复）...")
    caption_files = list(caption_dir.rglob('*.json'))
    success_count, fail_count, repaired_count = 0, 0, 0

    for caption_path in tqdm(caption_files, desc="注入Bbox并保存新文件"):
        caption_data = None
        try:
            # --- 自动修复逻辑 ---
            try:
                # 步骤 1: 尝试用标准解析器
                with open(caption_path, 'r', encoding='utf--8') as f:
                    caption_data = json.load(f)
            except json.JSONDecodeError as e:
                # 步骤 2: 如果失败，尝试用 demjson 修复
                logging.warning(f"文件 {caption_path.name} 格式不正确 ({e})，尝试自动修复...")
                with open(caption_path, 'r', encoding='utf-8') as f:
                    bad_json_text = f.read()
                
                try:
                    caption_data = demjson.decode(bad_json_text)
                    repaired_count += 1
                    logging.info(f"  -> 文件 {caption_path.name} 已成功修复！")
                except demjson.JSONDecodeError as dem_e:
                    logging.error(f"  -> 自动修复失败: {dem_e}。跳过文件 {caption_path.name}")
                    fail_count += 1
                    continue # 跳到下一个文件

            # --- Bbox注入逻辑 ---
            parts = caption_path.stem.rsplit('_', 1)
            if len(parts) != 2:
                logging.warning(f"文件名格式不符合预期，跳过: {caption_path.name}")
                fail_count += 1
                continue
            case_name, event_type = parts

            bboxes_to_inject = bbox_map.get(case_name, {}).get(event_type)
            if not bboxes_to_inject:
                caption_data["bbox_注入说明"] = "未能从抠图元数据中找到对应的Bbox信息。"
            else:
                analyses = caption_data.get("event_analyses")
                if isinstance(analyses, list):
                    if len(analyses) == len(bboxes_to_inject):
                        for i, analysis in enumerate(analyses):
                            comp_assessment = analysis.get("comprehensive_assessment") or analysis.get("detailed_cot", {}).get("comprehensive_assessment")
                            if isinstance(comp_assessment, dict):
                                comp_assessment["bbox"] = " ".join(map(str, bboxes_to_inject[i]))
                            else:
                                analysis["bbox_注入说明"] = "Bbox未能注入：在JSON中未找到'comprehensive_assessment'字典。"
                    else:
                        caption_data["bbox_注入说明"] = f"Bbox未能注入：需要的Bbox数量({len(bboxes_to_inject)})与JSON中的分析条目数({len(analyses)})不匹配。"
                else:
                     caption_data["bbox_注入说明"] = "Bbox未能注入：在JSON中未找到'event_analyses'键或其不是一个列表。"

            # 准备输出路径并保存新文件
            relative_path = caption_path.relative_to(caption_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(caption_data, f, ensure_ascii=False, indent=2)
            
            success_count += 1

        except Exception as e:
            fail_count += 1
            logging.error(f"处理文件 {caption_path.name} 时发生未知严重错误: {e}")

    logging.info(f"注入处理完成。成功处理: {success_count}个文件 (其中 {repaired_count} 个是自动修复的), 失败: {fail_count}个文件。")

def main():
    parser = argparse.ArgumentParser(
        description="后处理脚本：自动修复不规范的JSON，注入Bbox，并保存到新目录。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--cropped_data_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/cropped_data", help="The directory containing cropped images and their metadata JSONs (the source of bboxes).")
    parser.add_argument("--caption_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm", help="The root directory of the final caption results to be modified.")
    parser.add_argument("--output_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_caption_bbox", help="保存注入了Bbox的新JSON文件的根目录。")
    
    args = parser.parse_args()

    # 安装 demjson3 (如果尚未安装)
    try:
        import demjson3
    except ImportError:
        logging.critical("错误：需要 demjson3 库来进行JSON修复。请运行 'pip install demjson3'。")
        return

    cropped_data_path = Path(args.cropped_data_dir)
    caption_path = Path(args.caption_dir)
    output_path = Path(args.output_dir)

    if not cropped_data_path.is_dir() or not caption_path.is_dir():
        logging.critical("错误：指定的输入目录不存在，请检查路径。")
        return
    output_path.mkdir(parents=True, exist_ok=True)

    bbox_map = build_bbox_map(cropped_data_path)
    inject_bboxes_into_captions(caption_path, output_path, bbox_map)

    logging.info("所有操作已完成。")

if __name__ == "__main__":
    main()
    