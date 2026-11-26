import requests
import json

url = "https://arms-referring-matthew-appendix.trycloudflare.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer McclEyj2WoJiotWZOT05quXqnSl0ZQLUeSK14ebrioQ",
    "x-proxy-secret": "my_secret_password_sensoro_123"
}
data = {
    "model": "Qwen3-VL-235B-A22B-Instruct",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "测试一下自动脚本"}]}]
}

try:
    response = requests.post(url, headers=headers, json=data, timeout=60) # 尝试与你的OpenAI客户端相同的超时
    response.raise_for_status() # 如果请求失败，抛出HTTPError
    print("请求成功！")
    print(response.json())
except requests.exceptions.Timeout:
    print("请求超时！")
except requests.exceptions.RequestException as e:
    print(f"请求发生错误: {e}")
