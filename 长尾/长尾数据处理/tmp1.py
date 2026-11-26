import os

def delete_matching_jsons(img_dir, json_dir):
    # 定义图片文件的扩展名，根据需要添加
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # 检查文件夹是否存在
    if not os.path.exists(img_dir) or not os.path.exists(json_dir):
        print("错误：请检查文件夹路径是否正确。")
        return

    print(f"正在扫描文件夹1: {img_dir}")
    print(f"目标删除文件夹2: {json_dir}")
    print("-" * 30)

    count = 0
    
    # 遍历文件夹1
    for filename in os.listdir(img_dir):
        # 1. 检查是否是图片文件
        if filename.lower().endswith(valid_extensions):
            
            # 2. 获取不带后缀的文件名 (例如 'photo.jpg' -> 'photo')
            name_without_ext = os.path.splitext(filename)[0]
            
            # 3. 构建对应的 json 文件路径
            json_filename = name_without_ext + ".json"
            json_path = os.path.join(json_dir, json_filename)
            
            # 4. 检查文件夹2中是否存在同名的json文件
            if os.path.exists(json_path):
                try:
                    # 5. 删除文件
                    os.remove(json_path)
                    print(f"[已删除] {json_filename}")
                    count += 1
                except Exception as e:
                    print(f"[删除失败] {json_filename}: {e}")

    print("-" * 30)
    print(f"处理完成，共删除 {count} 个 JSON 文件。")

if __name__ == "__main__":
    # --- 请在这里修改你的文件夹路径 ---
    # 建议使用绝对路径，如果路径中有反斜杠 \，请在引号前加 r
    
    folder_1 = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第一批_260202/第一批260202/images/拉条幅/389/无效样本"
    folder_2 = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202601/imagesandlabels/p1_第一批_260202/第一批260202/images_results_0126_p1_qwen3vl235B_dirbased/横幅事件"

    delete_matching_jsons(folder_1, folder_2)