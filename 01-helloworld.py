# 需要代码解释了复制去问 豆包
from openai import OpenAI
import os
deepseek_key = os.getenv("DEEPSEEK_KEY")

# 1. 初始化客户端（连接 DeepSeek）
client = OpenAI(
    api_key=deepseek_key,
    base_url="https://api.deepseek.com"
)

# 2. 发送请求，获取模型返回结果
res = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "hello world"}]
)

# 3. 提取并打印模型的文字回答
print(res.choices[0].message.content)