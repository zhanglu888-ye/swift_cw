import base64
import requests
import json
import os

# =================配置区域=================
# 1. API 地址 (已更新)
API_URL = "https://arms-referring-matthew-appendix.trycloudflare.com/v1/chat/completions"

# 2. 认证信息
API_KEY = "McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ"
PROXY_SECRET = "my_secret_password_sensoro_123" # 新增的 Secret

# 3. 本地图片路径 (请确保此路径在运行脚本的机器上真实存在)
IMAGE_PATH = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/负样本/人员倒地/367/ffda3ccf-3bc0-4bc1-a108-126276681319.jpg"

# =================功能逻辑=================

# 检查文件是否存在
if not os.path.exists(IMAGE_PATH):
    print(f"❌ 错误: 找不到文件 {IMAGE_PATH}")
    print("请检查路径是否正确。")
    exit(1)

# 将本地图片转换为 Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

print(f"正在读取并编码图片: {IMAGE_PATH} ...")
base64_image = encode_image(IMAGE_PATH)

# 构造请求头 (已添加 x-proxy-secret)
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "x-proxy-secret": PROXY_SECRET
}

# 构造请求体
# 注意：提示词是 "Please describe this image."
# 如果模型只是单纯描述，可能会返回纯文本而不是图片。
# 如果需要模型进行编辑或画框，请修改 text 内容。
payload = {
    "model": "Qwen-Image-Edit-2511",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "A beautiful sunset over the ocean."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
}

print("正在发送请求到 API ...")

try:
    # 发送请求
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # 检查 HTTP 状态码
    if response.status_code != 200:
        print(f"❌ API 请求失败，状态码: {response.status_code}")
        print("错误详情:", response.text)
        exit(1)

    result_json = response.json()
    
    # 获取 choices 里的第一条结果
    if 'choices' in result_json and len(result_json['choices']) > 0:
        message_content = result_json['choices'][0]['message']['content']
        
        # === 情况 A: 返回的是图片 (Base64) ===
        if isinstance(message_content, list) and message_content[0].get('type') == 'image_url':
            image_data_url = message_content[0]['image_url']['url']
            
            # 处理 Base64 字符串头 (data:image/xxx;base64,...)
            header, encoded = image_data_url.split(",", 1)
            file_ext = "png" if "png" in header else "jpg"
            
            # 解码并保存
            image_bytes = base64.b64decode(encoded)
            output_filename = f"output_image.{file_ext}"
            
            with open(output_filename, "wb") as f:
                f.write(image_bytes)
                
            print(f"✅ 成功! AI 生成的图片已保存为: {output_filename}")
            
        # === 情况 B: 返回的是纯文本描述 ===
        else:
            print("\n✅ 成功! AI 返回了文本描述:")
            print("-" * 30)
            # 有些模型返回 list 形式的 text，有些直接返回 string，这里做个兼容
            if isinstance(message_content, list):
                for item in message_content:
                    if item.get('type') == 'text':
                        print(item.get('text'))
            else:
                print(message_content)
            print("-" * 30)
            
    else:
        print("❌ 返回数据格式异常，未找到 choices 字段。")
        print(json.dumps(result_json, indent=2))

except Exception as e:
    print(f"❌ 发生程序错误: {e}")