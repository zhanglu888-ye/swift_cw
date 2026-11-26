import requests
import base64

# ================= 配置 =================
# 1. 这里先填 localhost，排除 Cloudflare 的干扰
API_URL = "https://arms-referring-matthew-appendix.trycloudflare.com/v1/chat/completions" 
# 2. 如果 localhost 能通，再换成你的 https://....trycloudflare.com 测试
# API_URL = "https://arms-referring-matthew-appendix.trycloudflare.com/v1/chat/completions"

API_KEY = "McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ"

# 一个 1x1 像素的红色图片
BASE64_IMG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
# =======================================

def test():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "Qwen-Image-Edit-2511",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "make it snow"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{BASE64_IMG}"}}
                ]
            }
        ]
    }

    print(f"🚀 正向 {API_URL} 发起测试请求...")
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        print(f"📡 状态码: {resp.status_code}")
        
        if resp.status_code == 200:
            print("✅ 成功！接口通了。")
            print(f"返回部分内容: {str(resp.json())[:100]}...")
        else:
            print("❌ 失败！")
            print(f"错误内容: {resp.text}")
            
    except Exception as e:
        print(f"💥 发生异常: {e}")

if __name__ == "__main__":
    test()
