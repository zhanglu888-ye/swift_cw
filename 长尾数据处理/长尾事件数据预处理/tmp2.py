import json
import os
import sys

# ================= 配置区 =================
# 输入文件夹路径 (包含 jsonl 的根目录)
INPUT_DIR = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p1"

# 输出文件夹路径 (清洗后的文件将保存在这里，保持原有的目录结构)
# 建议新建一个目录，不要和输入目录完全一样，防止覆盖原始数据
OUTPUT_DIR = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/train_data/p1_cleaned"
# =========================================

def process_single_file(file_path, save_path):
    """
    处理单个文件的函数
    返回: (total, kept, removed, missing_examples_list)
    """
    total_count = 0
    kept_count = 0
    removed_count = 0
    missing_examples = []
    
    # 确保输出文件的父目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(file_path, 'r', encoding='utf-8') as fin, \
         open(save_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line_content = line.strip()
            if not line_content:
                continue

            total_count += 1
            try:
                data = json.loads(line_content)
            except json.JSONDecodeError:
                print(f"  [警告] JSON解析错误: 行 {line_num}")
                removed_count += 1
                continue

            images = data.get("images", [])
            
            # 检查图片是否存在
            all_images_exist = True
            missing_img_path = ""

            for img_path in images:
                if not os.path.exists(img_path):
                    all_images_exist = False
                    missing_img_path = img_path
                    break 
            
            if all_images_exist:
                fout.write(line_content + "\n")
                kept_count += 1
            else:
                removed_count += 1
                if len(missing_examples) < 5: # 每个文件只记录前5个错误作为示例
                    missing_examples.append(f"行 {line_num}: 缺失 {missing_img_path}")

    return total_count, kept_count, removed_count, missing_examples

def batch_clean(input_dir, output_dir):
    print(f"开始批量处理...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)

    if not os.path.exists(input_dir):
        print("错误: 输入目录不存在")
        return

    global_total = 0
    global_kept = 0
    global_removed = 0
    processed_files = 0

    # 遍历目录
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                # 获取源文件的完整路径
                src_path = os.path.join(root, file)
                
                # 计算相对路径，以便在输出目录保持结构
                rel_path = os.path.relpath(src_path, input_dir)
                dst_path = os.path.join(output_dir, rel_path)

                print(f"正在处理: {rel_path}")
                
                # 处理单个文件
                t, k, r, missing = process_single_file(src_path, dst_path)
                
                # 累加统计
                global_total += t
                global_kept += k
                global_removed += r
                processed_files += 1

                # 如果该文件有删除记录，打印少量日志
                if r > 0:
                    print(f"  -> 删除 {r} 条 (总 {t} 条)")
                    for msg in missing:
                        print(f"     [示例] {msg}")
                    if r > 5:
                        print(f"     ... 等")

    print("\n" + "=" * 50)
    print("全部处理完成")
    print("=" * 50)
    print(f"处理文件数 : {processed_files}")
    print(f"原始总行数 : {global_total}")
    print(f"保留总行数 : {global_kept}")
    print(f"删除总行数 : {global_removed}")
    
    if global_total > 0:
        print(f"总保留比例 : {global_kept/global_total:.2%}")

if __name__ == "__main__":
    # 如果命令行传入参数则使用命令行参数，否则使用配置区的路径
    if len(sys.argv) == 3:
        batch_clean(sys.argv[1], sys.argv[2])
    else:
        batch_clean(INPUT_DIR, OUTPUT_DIR)