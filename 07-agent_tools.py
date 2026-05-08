from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
import requests
import os

# --- 1. 配置 ---
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"


# --- 2. 定义工具：给Agent的武器库 ---

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入如 '2+3*4' 或 'sqrt(16)'，返回计算结果。
    当用户问数学计算问题时，必须用这个工具。"""
    try:
        # 安全的数学计算，只允许数字和基本运算符
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "表达式包含不允许的字符，请只使用数字和 + - * / ( ) . %"
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算出错: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """查询城市的实时天气。输入城市名称（中文），返回天气信息。
    当用户问天气相关问题时，必须用这个工具。"""
    try:
        # 使用免费的 wttr.in API，不需要Key
        url = f"https://wttr.in/{city}?format=%C+%t+%h+%w&lang=zh"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return f"{city}天气: {response.text.strip()}"
        return f"查询{city}天气失败"
    except:
        return f"无法连接到天气服务"


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。输入搜索关键词，返回搜索结果摘要。
    当用户问时事、新闻、最新消息，或者你不知道答案的实时信息时，必须用这个工具。"""
    try:
        # 使用 DuckDuckGo 的免费搜索API（不需要Key）
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        # 提取摘要
        abstract = data.get("AbstractText", "")
        if abstract:
            return f"搜索结果: {abstract}"

        # 如果没有摘要，返回相关主题
        related = data.get("RelatedTopics", [])
        if related:
            results = []
            for topic in related[:3]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(topic["Text"])
            if results:
                return "搜索结果:\n" + "\n".join(results)

        return f"没有找到关于'{query}'的搜索结果"
    except Exception as e:
        return f"搜索失败: {str(e)}"


# --- 3. 初始化模型 ---
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=API_KEY,
    base_url=BASE_URL,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"}
        }
    }
)

# --- 4. 创建Agent，装上工具 ---
tools = [calculator, get_weather, web_search]

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手，可以使用工具来帮助用户解决问题。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 使用最新的 create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建 AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # 显示详细执行过程
)

# --- 5. 交互式对话 ---
print("=" * 60)
print("🤖 多工具智能Agent已启动！")
print("   我能用: 🧮计算器  🌤️天气查询  🔍网页搜索")
print("=" * 60)

while True:
    user_input = input("\n👤 你: ")
    if user_input.lower() in ['q', 'quit', '退出']:
        print("👋 再见！")
        break
    if not user_input.strip():
        continue

    print("\n🧠 Agent执行中...")
    result = agent_executor.invoke({"input": user_input})

    # 直接显示最终输出
    print(f"\n🤖 最终回答: {result['output']}")