import os
import json
import shutil
import pandas as pd
import argparse

# 定义风险等级映射
# 集合中的状态映射为 ["森林火灾"] -> 正样本
FIRE_RISKS = {
    "高风险", "中风险", "低风险",
    "疑似高风险", "疑似中风险", "疑似低风险"
}

# 集合中的状态映射为 [] (空标签) -> 负样本
NORMAL_RISKS = {
    "正常"
}

def prepare_data(xlsx_folder, source_image_root, output_root):
    # 1. 创建输出目录结构
    images_out_dir = os.path.join(output_root, "large_images")
    labels_out_dir = os.path.join(output_root, "label_json")
    
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    
    # 2. 获取所有xlsx文件
    xlsx_files = [f for f in os.listdir(xlsx_folder) if f.endswith('.xlsx')]
    
    if not xlsx_files:
        print(f"在 {xlsx_folder} 中没有找到 .xlsx 文件。")
        return

    print(f"找到 {len(xlsx_files)} 个Excel文件，开始处理...")
    
    # 初始化统计数据
    stats = {
        "processed": 0,      # 总成功处理数
        "positive": 0,       # 正样本 (森林火灾)
        "negative": 0,       # 负样本 (正常/空标签)
        "skipped": 0         # 跳过 (文件丢失或格式错误)
    }
    
    # 遍历每个 Excel 文件
    for xlsx_file in xlsx_files:
        xlsx_path = os.path.join(xlsx_folder, xlsx_file)
        print(f"正在读取: {xlsx_file}")
        
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"读取 Excel {xlsx_file} 失败: {e}")
            continue
        
        # 格式化列名
        df.columns = [c.strip() for c in df.columns]
        required_cols = ['图片路径', '人工风险等级']
        if not all(col in df.columns for col in required_cols):
            print(f"跳过 {xlsx_file}: 缺少必要列 '图片路径' 或 '人工风险等级'")
            continue

        # 使用 tqdm 显示进度条 (如果可用)
        rows = df.iterrows()
        try:
            from tqdm import tqdm
            rows = tqdm(df.iterrows(), total=df.shape[0], desc=xlsx_file)
        except ImportError:
            pass

        for index, row in rows:
            rel_path = str(row['图片路径']).strip()
            risk_level = str(row['人工风险等级']).strip()
            
            # 1. 确定标签与样本类型
            labels = []
            is_positive_sample = False
            
            if risk_level in FIRE_RISKS:
                labels = ["森林火灾"]
                is_positive_sample = True
            elif risk_level in NORMAL_RISKS:
                labels = []
                is_positive_sample = False
            else:
                # 遇到未知等级
                stats["skipped"] += 1
                continue
            
            # 2. 构建完整图片路径
            clean_rel_path = rel_path.replace('\\', '/')
            full_src_path = os.path.join(source_image_root, clean_rel_path)
            
            # 3. 检查图片是否存在
            if not os.path.exists(full_src_path):
                stats["skipped"] += 1
                continue
            
            # 4. 准备目标文件名
            file_name = os.path.basename(clean_rel_path)
            dst_image_path = os.path.join(images_out_dir, file_name)
            
            # 5. 复制图片 & 生成 JSON
            try:
                # 复制图片
                shutil.copy2(full_src_path, dst_image_path)
                
                # 生成 JSON
                json_data = {
                    "file_name": file_name,
                    "labels": labels
                }
                json_filename = os.path.splitext(file_name)[0] + ".json"
                json_out_path = os.path.join(labels_out_dir, json_filename)
                
                with open(json_out_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                # 更新统计
                stats["processed"] += 1
                if is_positive_sample:
                    stats["positive"] += 1
                else:
                    stats["negative"] += 1
                    
            except Exception as e:
                print(f"处理失败 {full_src_path}: {e}")
                stats["skipped"] += 1
                continue

    # 打印最终统计结果
    print("\n" + "="*50)
    print(f"{'数据准备统计结果':^46}")
    print("="*50)
    print(f"{'项目':<25} | {'数量':<15}")
    print("-" * 50)
    print(f"{'正样本 (森林火灾)':<25} | {stats['positive']:<15}")
    print(f"{'负样本 (正常/无标签)':<25} | {stats['negative']:<15}")
    print(f"{'总成功处理':<25} | {stats['processed']:<15}")
    print(f"{'跳过/缺失/错误':<25} | {stats['skipped']:<15}")
    print("-" * 50)
    
    if stats["negative"] > 0:
        ratio = stats["positive"] / stats["negative"]
        print(f"正负样本比例 (Pos:Neg) ≈ {ratio:.2f} : 1")
    else:
        print("无负样本。")
    print("="*50)
    print(f"输出目录: {output_root}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将Excel标注数据转换为训练集格式")
    
    # 参数配置
    parser.add_argument("--xlsx_folder", type=str, default="/host/ssddata/disk1/wyd/data/fire_test/fire_rank2/tmp_训练长尾9类模型用/data", help="存放xlsx文件的文件夹路径")
    parser.add_argument("--src_root", type=str, default="/host/ssddata/disk1/wyd/data/fire_test/fire_rank2/images", help="Excel中图片路径的基础根目录（例如 D-Fire 数据集的根目录）")
    parser.add_argument("--output", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/森林火灾_tmp", help="输出的数据集路径")
    
    args = parser.parse_args()
    
    prepare_data(args.xlsx_folder, args.src_root, args.output)