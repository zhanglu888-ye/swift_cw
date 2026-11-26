# -*- coding: utf-8 -*-
import os
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置请求头，防止被服务器拒绝
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def sanitize_filename(name):
    """清理文件名中的非法字符"""
    if not isinstance(name, str):
        return str(name)
    return name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '').strip()

def download_one_image(url, save_dir, filename):
    """
    单张图片下载函数
    :return: (status, message/path)
    """
    if not url or pd.isna(url):
        return False, "URL为空"

    try:
        # 清理文件名
        safe_name = sanitize_filename(filename)
        # 如果CSV里的name没有后缀，尝试从URL推断（可选，这里假设name已有后缀）
        if not os.path.splitext(safe_name)[1]:
            safe_name += ".jpg"
            
        save_path = os.path.join(save_dir, safe_name)

        # 检查文件是否已存在且不为空
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True, "已存在 (跳过)"

        # 发起请求
        response = requests.get(url, headers=HEADERS, stream=True, timeout=15)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return True, save_path
        else:
            return False, f"HTTP状态码: {response.status_code}"

    except Exception as e:
        return False, f"异常: {str(e)}"

def main(csv_path, output_dir, max_workers=8):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 2. 读取CSV文件
    print(f"正在读取 CSV: {csv_path} ...")
    try:
        # 尝试读取 CSV。如果你的是 Excel 导出的标准CSV，用逗号分隔。
        # 如果你的文件真的是用 '|' 分隔的文本文件，请将 sep=',' 改为 sep='|'
        # 这里的 engine='python' 和 on_bad_lines='skip' 是为了防止格式错误导致读取中断
        df = pd.read_csv(csv_path, engine='python')
        
        # 剔除列名可能存在的空格
        df.columns = df.columns.str.strip()
        
        # 检查必要的列
        if 'image_url' not in df.columns or 'name' not in df.columns:
            # 有时候CSV里是 ' image_url ' 或 'name '，尝试模糊匹配
            print("警告: 未找到标准的 'image_url' 或 'name' 列，尝试自动查找...")
            for col in df.columns:
                if 'image_url' in col: df.rename(columns={col: 'image_url'}, inplace=True)
                if 'name' in col: df.rename(columns={col: 'name'}, inplace=True)
        
        if 'image_url' not in df.columns:
            raise ValueError("CSV中找不到图片链接列 (image_url)")

    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    total_images = len(df)
    print(f"共发现 {total_images} 条数据，准备开始下载...")
    print(f"线程数: {max_workers}")

    # 3. 多线程下载
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_img = {}
        for index, row in df.iterrows():
            url = row.get('image_url')
            name = row.get('name')
            
            # 如果 name 为空，用 data_id 替代，如果 data_id 也没有，用文件名
            if pd.isna(name) or str(name).strip() == "":
                name = row.get('data_id', f"image_{index}.jpg")

            future = executor.submit(download_one_image, url, output_dir, name)
            future_to_img[future] = name

        # 处理结果与进度条
        with tqdm(total=total_images, desc="下载进度", unit="img") as pbar:
            for future in as_completed(future_to_img):
                img_name = future_to_img[future]
                try:
                    is_success, msg = future.result()
                    if is_success:
                        success_count += 1
                    else:
                        fail_count += 1
                        # 可以取消注释下面这行来查看具体失败原因
                        # tqdm.write(f"失败 [{img_name}]: {msg}")
                except Exception as exc:
                    fail_count += 1
                    tqdm.write(f"任务异常 [{img_name}]: {exc}")
                
                pbar.update(1)

    print("-" * 30)
    print(f"处理完成！")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"图片保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV图片批量下载工具")
    parser.add_argument("--csv_path", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/p0算法测试集_260114/dms文件/767_非机动车乱停放_本地.csv", help="CSV文件的路径")
    parser.add_argument("--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/p0算法测试集_260114/images/非机动车乱停放", help="图片保存的文件夹路径")
    parser.add_argument("--workers", type=int, default=10, help="并行下载的线程数 (默认10)")

    args = parser.parse_args()

    main(args.csv_path, args.output_dir, args.workers)