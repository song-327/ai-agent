import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
import requests,os

Api_Key = os.getenv("DEEPSEEK_API_KEY")
Base_Url = "https://api.deepseek.com"

# --- 页面配置 ---
st.set_page_config(page_title="AI智能助手", page_icon="🤖")
st.title("🤖 多工具AI智能助手")
st.caption("我能用计算器、查天气、搜网页——Agent会自动选择工具")

# --- 初始化模型和Agent（缓存，只加载一次）---
@st.cache_resource
def load_agent():
    # 定义工具
    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式。用户问数学计算时必须用这个工具。"""
        try:
            allowed = set("0123456789+-*/().%^ ")
            if not all(c in allowed for c in expression):
                return "表达式包含不允许的字符"
            # 安全计算
            result = eval(expression, {"__builtins__": {}})
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算出错: {str(e)}"

    @tool
    def get_weather(city: str) -> str:
        """查询城市实时天气。用户问天气时必须用这个工具。"""
        try:
            url = f"https://wttr.in/{city}?format=%C+%t+%h+%w&lang=zh"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return f"{city}天气: {response.text.strip()}"
            return f"查询{city}天气失败"
        except:
            return "无法连接到天气服务"

    @tool
    def web_search(query: str) -> str:
        """搜索互联网获取最新信息。用户问实时信息时必须用这个工具。"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            abstract = data.get("AbstractText", "")
            if abstract:
                return f"搜索结果: {abstract}"
            related = data.get("RelatedTopics", [])
            if related:
                results = [t["Text"] for t in related[:3] if "Text" in t]
                if results:
                    return "搜索结果:\n" + "\n".join(results)
            return f"没有找到关于'{query}'的结果"
        except:
            return "搜索服务暂时不可用，请稍后重试"

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=Api_Key,
        base_url=Base_Url,
        # 方法1：通过 default_headers
        default_headers={
            "X-DashScope-SSE": "disable"  # 禁用流式思考
        }
    )
    return create_react_agent(llm, [calculator, get_weather, web_search])

agent = load_agent()

# --- 初始化聊天记录 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 显示历史消息 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 接收用户输入 ---
if prompt := st.chat_input("问我任何事，比如：南宁今天天气怎么样？125*37等于多少？"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent思考中..."):
            result = agent.invoke({"messages": [("user", prompt)]})
            final_msg = result["messages"][-1]
            reply = final_msg.content if hasattr(final_msg, 'content') else str(final_msg)
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})