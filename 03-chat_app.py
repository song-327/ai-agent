# ==============================================
# AI 聊天机器人（Streamlit + DeepSeek）
# 功能：网页聊天 + 记忆上下文 + 安全隐藏API Key
# 运行命令：streamlit run 03-chat_app.py
# ==============================================

# 1. 导入需要的工具
from openai import OpenAI  # 调用大模型
import streamlit as st  # 做网页界面
import os  # 读取环境变量（安全拿Key）

# 2. 从电脑环境变量读取 DeepSeek API Key（不会泄露）
dp_key = os.getenv("DEEPSEEK_KEY")

# 3. 创建客户端，连接 DeepSeek 大模型
client = OpenAI(
    api_key=dp_key,  # 你的密钥（安全读取）
    base_url="https://api.deepseek.com"  # 大模型服务器地址
)

# 4. 设置网页标题
st.title("第一个AI聊天助手")

# 5. 初始化聊天记录（用来保存对话历史）
if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. 显示历史聊天消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. 获取用户输入的问题
if prompt := st.chat_input("请输入你的问题"):
    # 8. 把用户的问题保存到聊天记录
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 9. 在网页上显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 10. AI 思考中（显示加载动画）
    with st.spinner("正在思考..."):
        # 11. 发送请求给大模型，获取回答
        res = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=st.session_state.messages
        )
        # 12. 提取AI的文字回答
        reply = res.choices[0].message.content

    # 13. 在网页上显示AI的回答
    with st.chat_message("assistant"):
        st.markdown(reply)

    # 14. 把AI的回答也保存到聊天记录
    st.session_state.messages.append({"role": "assistant", "content": reply})