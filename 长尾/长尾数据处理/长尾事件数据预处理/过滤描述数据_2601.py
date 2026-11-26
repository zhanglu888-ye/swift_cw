import os
import json
import shutil
from pathlib import Path
import argparse
import logging
from typing import Dict, Any, Tuple

# 配置日志
# 默认INFO级别，会显示重要的处理信息。要查看详细的调试信息（例如JSON内容解析），可以将级别设置为logging.DEBUG。
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='根据JSON内容清洗正负样本图片及标签。')
    parser.add_argument('--image_base_dir', type=str, 
                        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第二批_260209/images", 
                        help='图片文件的根目录。例如: /path/to/清洗标注结果/images')
    parser.add_argument('--label_base_dir', type=str, 
                        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第二批_260209/labels", 
                        help='JSON标签文件的根目录。例如: /path/to/清洗标注结果/labels')
    parser.add_argument('--moved_images_root', type=str, 
                        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第二批_260209_cleaned/images_moved", 
                        help='符合“移动”条件的图片目标根目录。原始文件将被移动到这里。')
    parser.add_argument('--moved_labels_root', type=str, 
                        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第二批_260209_cleaned/labels_moved", 
                        help='符合“移动”条件的JSON标签目标根目录。原始文件将被移动到这里。')
    parser.add_argument('--kept_images_root', type=str,
                        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第二批_260209_cleaned/images_kept",
                        help='不符合“移动”条件（即保留）的图片目标根目录。原始文件将被移动到这里。')
    parser.add_argument('--kept_labels_root', type=str,
                        default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第二批_260209_cleaned/labels_kept",
                        help='不符合“移动”条件（即保留）的JSON标签目标根目录。原始文件将被移动到这里。')
    parser.add_argument('--move_action', default=True,
                        help='如果设置此参数，则实际移动文件。否则，脚本只报告哪些文件会被移动，但不执行实际操作（干运行模式）。')
    return parser.parse_args()

def check_json_conditions(json_data: Dict, is_positive_sample: bool) -> bool:
    """
    判断JSON文件是否满足“移动”的清洗条件。

    Args:
        json_data (Dict): JSON文件的内容。
        is_positive_sample (bool): 指示当前处理的是否为正样本。

    Returns:
        bool: 如果文件应被移动，则返回 True；否则返回 False（即应被保留）。
    """
    try:
        analyses = json_data.get('analyses', [])
        if not analyses or not isinstance(analyses, list) or len(analyses) == 0:
            logging.debug("  JSON中未找到analyses字段或其为空列表")
            return False # 无法判断，默认不移动
        
        # 假设我们只关心第一个event_analyses
        event_analyses = analyses[0].get('event_analyses', [])
        if not event_analyses or not isinstance(event_analyses, list) or len(event_analyses) == 0:
            logging.debug("  JSON中未找到event_analyses字段或其为空列表")
            return False # 无法判断，默认不移动
        
        # 假设我们只关心第一个comprehensive_assessment
        comprehensive_assessment = event_analyses[0].get('comprehensive_assessment', {})
        
        abnormal_type = comprehensive_assessment.get('异常类型', '')
        risk_level = comprehensive_assessment.get('风险等级', '')
        
        # 判断是否符合“无异常且0级风险”的基准条件
        meets_no_anomaly_condition = (abnormal_type == "无异常" and risk_level == "0级 - 无风险")

        if is_positive_sample:
            # 正样本：如果它是“无异常且0级风险”，则视为异常情况，需要“移动”。
            # 目标是希望正样本是“有异常”的。
            return meets_no_anomaly_condition
        else:
            # 负样本：如果它不是“无异常且0级风险”（即“有异常”），则视为异常情况，需要“移动”。
            # 目标是希望负样本是“无异常”的。
            return not meets_no_anomaly_condition
            
    except Exception as e:
        logging.error(f"  解析JSON内容时出错: {e}")
        return False # 解析错误，默认不移动

def is_valid_json_file(path: Path) -> bool:
    """检查文件是否存在、非空，并且是有效的JSON"""
    if not path.exists() or path.stat().st_size == 0:
        logging.debug(f"文件 {path} 不存在或为空。")
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        logging.warning(f"文件 {path} 存在但JSON格式无效或编码错误。")
        return False

def _do_file_operation(src_path: Path, dst_path: Path, action_func: callable):
    """
    内部辅助函数：执行文件操作（移动或复制），处理同名文件冲突。
    action_func 可以是 shutil.move 或 shutil.copy2
    """
    counter = 0
    original_dst_path = dst_path
    # 如果目标文件已存在，添加后缀避免覆盖
    while dst_path.exists():
        counter += 1
        dst_path = original_dst_path.with_stem(f"{original_dst_path.stem}_{counter}")
    
    # 确保目标目录存在
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    action_func(src_path, dst_path)
    return dst_path

def process_file_pair(
    image_source_path: Path, 
    json_source_path: Path, 
    image_base_dir: Path,
    moved_images_root: Path, 
    moved_labels_root: Path,
    kept_images_root: Path,
    kept_labels_root: Path,
    move_action: bool
) -> str:
    """
    处理单个图片-标签对：检查条件，决定移动或保留，并执行操作。

    Returns:
        str: 'moved', 'kept', 'skipped_json_missing', 'skipped_json_invalid', 'error_operating'
    """
    # 1. 检查JSON文件是否存在且有效
    if not json_source_path.exists():
        logging.warning(f"  ✗ 跳过 {image_source_path.name}: 对应的JSON文件 {json_source_path.name} 不存在。")
        return 'skipped_json_missing'
    if not is_valid_json_file(json_source_path):
        logging.warning(f"  ✗ 跳过 {image_source_path.name}: 对应的JSON文件 {json_source_path.name} 无效。")
        return 'skipped_json_invalid'
    
    # 2. 获取JSON数据
    json_data = {}
    try:
        with open(json_source_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        logging.error(f"  ✗ 读取JSON文件 {json_source_path.name} 失败: {e}")
        return 'skipped_json_invalid'

    # 3. 判断是正样本还是负样本 (通过路径中的“正样本”或“负样本”目录名)
    is_positive_sample = "正样本" in image_source_path.parts
    
    # 4. 检查是否需要“移动”（即不符合训练集要求）
    should_be_moved = check_json_conditions(json_data, is_positive_sample)

    # 获取相对于image_base_dir的相对路径，例如 "人员倒地/正样本/image.jpg"
    relative_path_from_base = image_source_path.relative_to(image_base_dir)

    target_image_path: Path
    target_json_path: Path
    action_result_str: str # 用于返回统计结果

    if should_be_moved:
        target_image_path = moved_images_root / relative_path_from_base
        target_json_path = moved_labels_root / relative_path_from_base.with_suffix('.json')
        action_result_str = 'moved'
        logging.debug(f"  图片 {image_source_path.name} 和标签 {json_source_path.name} 符合移动条件。")
    else:
        target_image_path = kept_images_root / relative_path_from_base
        target_json_path = kept_labels_root / relative_path_from_base.with_suffix('.json')
        action_result_str = 'kept'
        logging.debug(f"  图片 {image_source_path.name} 和标签 {json_source_path.name} 不符合移动条件，将保留。")

    if move_action:
        try:
            # 执行文件移动操作
            moved_img_final_path = _do_file_operation(image_source_path, target_image_path, shutil.move)
            moved_json_final_path = _do_file_operation(json_source_path, target_json_path, shutil.move)
            logging.info(f"  ✓ 已将 {'图片' if should_be_moved else '图片'} {image_source_path.name} -> {action_result_str} 到 {moved_img_final_path.parent.name}/{moved_img_final_path.name}")
            logging.info(f"  ✓ 已将 {'标签' if should_be_moved else '标签'} {json_source_path.name} -> {action_result_str} 到 {moved_json_final_path.parent.name}/{moved_json_final_path.name}")
            return action_result_str
        except Exception as e:
            logging.error(f"  ✗ 执行操作失败 ({action_result_str}): {image_source_path.name} 和 {json_source_path.name}: {e}")
            return 'error_operating'
    else:
        # 干运行模式，只报告不操作
        logging.info(f"  [干运行] {'将移动' if should_be_moved else '将保留'} 图片: {image_source_path} -> {target_image_path}")
        logging.info(f"  [干运行] {'将移动' if should_be_moved else '将保留'} 标签: {json_source_path} -> {target_json_path}")
        return action_result_str # 在干运行模式下仍然返回action_type以便统计

def clean_dataset(image_base_dir: Path, label_base_dir: Path, 
                  moved_images_root: Path, moved_labels_root: Path,
                  kept_images_root: Path, kept_labels_root: Path,
                  move_action: bool):
    """
    主处理函数：遍历图片目录，根据JSON内容清洗并移动图片及标签。
    """
    total_pairs = 0
    moved_pairs_count = 0
    kept_pairs_count = 0
    skipped_json_missing = 0
    skipped_json_invalid = 0
    error_operating_count = 0
    
    logging.info(f"开始清洗数据集...")
    logging.info(f"图片源目录: {image_base_dir}")
    logging.info(f"标签源目录: {label_base_dir}")
    logging.info(f"移动图片目标根目录: {moved_images_root}")
    logging.info(f"移动标签目标根目录: {moved_labels_root}")
    logging.info(f"保留图片目标根目录: {kept_images_root}")
    logging.info(f"保留标签目标根目录: {kept_labels_root}")
    logging.info(f"操作模式: {'实际移动文件' if move_action else '干运行 (不移动文件)'}")
    logging.info("=" * 80)
    
    # 遍历图片根目录
    for root_str, _, files in os.walk(image_base_dir):
        root = Path(root_str)
        # 确保不会遍历到目标输出目录（如果它们是子目录）
        if root.is_relative_to(moved_images_root) or \
           root.is_relative_to(moved_labels_root) or \
           root.is_relative_to(kept_images_root) or \
           root.is_relative_to(kept_labels_root):
            logging.debug(f"跳过目标目录及其子目录: {root}")
            continue

        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                total_pairs += 1
                image_source_path = root / filename
                
                # 构建对应的JSON文件路径：通过替换根目录和改变后缀实现
                relative_path_from_image_base = image_source_path.relative_to(image_base_dir)
                json_source_path = (label_base_dir / relative_path_from_image_base).with_suffix('.json')

                logging.debug(f"\n[{total_pairs}] 处理文件对: {image_source_path.name}")
                
                result = process_file_pair(
                    image_source_path, json_source_path, image_base_dir,
                    moved_images_root, moved_labels_root,
                    kept_images_root, kept_labels_root,
                    move_action
                )

                if result == 'moved':
                    moved_pairs_count += 1
                elif result == 'kept':
                    kept_pairs_count += 1
                elif result == 'skipped_json_missing':
                    skipped_json_missing += 1
                elif result == 'skipped_json_invalid':
                    skipped_json_invalid += 1
                elif result == 'error_operating':
                    error_operating_count += 1

    logging.info("\n" + "=" * 80)
    logging.info("清洗处理完成！统计信息:")
    logging.info(f"总图片-标签对数: {total_pairs}")
    logging.info(f"因不符合条件被移动的对数: {moved_pairs_count} (目标: {moved_images_root.name} / {moved_labels_root.name})")
    logging.info(f"因符合条件被保留的对数: {kept_pairs_count} (目标: {kept_images_root.name} / {kept_labels_root.name})")
    logging.info(f"因JSON文件缺失而跳过: {skipped_json_missing}")
    logging.info(f"因JSON文件无效而跳过: {skipped_json_invalid}")
    logging.info(f"文件操作失败数: {error_operating_count}")
    logging.info("=" * 80)

def main():
    """主函数"""
    args = parse_arguments()
    
    image_base_dir = Path(args.image_base_dir)
    label_base_dir = Path(args.label_base_dir)
    moved_images_root = Path(args.moved_images_root)
    moved_labels_root = Path(args.moved_labels_root)
    kept_images_root = Path(args.kept_images_root)
    kept_labels_root = Path(args.kept_labels_root)
    
    # 验证输入目录是否存在
    if not image_base_dir.exists():
        logging.critical(f"错误: 图片源目录 '{image_base_dir}' 不存在!")
        return
    if not label_base_dir.exists():
        logging.critical(f"错误: 标签源目录 '{label_base_dir}' 不存在!")
        return
    
    # 创建目标目录（如果 move_action 为 True）
    if args.move_action:
        moved_images_root.mkdir(parents=True, exist_ok=True)
        moved_labels_root.mkdir(parents=True, exist_ok=True)
        kept_images_root.mkdir(parents=True, exist_ok=True)
        kept_labels_root.mkdir(parents=True, exist_ok=True)
    else:
        logging.info("提示: 处于干运行模式 (--move_action 未设置)，不会实际移动文件。")
    
    clean_dataset(image_base_dir, label_base_dir, 
                  moved_images_root, moved_labels_root,
                  kept_images_root, kept_labels_root,
                  args.move_action)

if __name__ == "__main__":
    main()