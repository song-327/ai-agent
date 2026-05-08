# 0 忘记了就问豆包

# 1. 导入 OpenAI 客户端库，用于调用大模型 API
from openai import OpenAI

# 2. 导入 os 系统库，专门用来读取环境变量（隐藏 API Key）
import os

# 3. 从系统环境变量中读取我们提前设置好的 DeepSeek API Key
# 好处：代码里不暴露真实密钥，绝对安全
deepseek_key = os.getenv("DEEPSEEK_KEY")

# 4. 创建大模型连接客户端
client = OpenAI(
    api_key=deepseek_key,       # 使用环境变量里的密钥，不写死在代码
    base_url="https://api.deepseek.com"  # DeepSeek 官方接口地址
)

# 5. 向 AI 发送问题，并把返回的所有结果保存到 res 变量中
res = client.chat.completions.create(
    model='deepseek-v4-flash',          # 指定使用的 AI 模型版本
    messages=[{"role":"user","content": "hello world"}]  # 发送的对话内容
)

# 6. 从返回结果里，提取 AI 的纯文字回答并打印出来
print(res.choices[0].message.content)