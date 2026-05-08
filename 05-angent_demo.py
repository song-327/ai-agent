# 导入 LangChain OpenAI 聊天模型，用来调用大模型（这里是 DeepSeek）
from langchain_openai import ChatOpenAI

# 导入 LangGraph 内置的创建 ReAct 智能体函数
# ReAct 是一种让 AI 思考 + 调用工具的模式
from langgraph.prebuilt import create_react_agent

# 导入 LangChain 工具装饰器，用来把普通函数变成 AI 能用的工具
from langchain.tools import tool

# 导入警告模块，屏蔽不需要的提示信息
import warnings
warnings.filterwarnings("ignore")

# 导入系统环境模块，用来读取电脑里的环境变量
import os

# 从环境变量中获取 DeepSeek 的 API Key（密钥）
# 你需要提前在电脑里配置好这个密钥
deepseek_key = os.getenv("DEEPSEEK_KEY")

# ---------------------- 定义 AI 能用的工具：计算器 ----------------------
# @tool 是装饰器，表示这个函数是给 AI 调用的工具
@tool
def calculator(expression: str) -> str:
    """
    工具说明：计算数学表达式
    输入格式：字符串，例如 '2+3*4'
    返回结果：计算结果字符串
    """
    try:
        # 执行数学表达式计算（eval 可以直接算字符串里的数学公式）
        return str(eval(expression))
    except:
        # 如果计算出错（比如格式不对），返回错误提示
        return "计算错误"

# ---------------------- 初始化大模型（DeepSeek） ----------------------
llm = ChatOpenAI(
    model="deepseek-chat",  # 使用的模型名称：DeepSeek 聊天模型
    api_key=deepseek_key,   # 使用刚才获取的 API 密钥
    base_url="https://api.deepseek.com"  # DeepSeek 的官方接口地址
)

# ---------------------- 创建智能体（AI 助手） ----------------------
# 给 AI 绑定：大模型 + 工具列表（这里只有计算器）
agent = create_react_agent(llm, [calculator])

# ---------------------- 让 AI 执行任务 ----------------------
# 调用智能体，传入用户问题
result = agent.invoke({
    "messages": [("user", "123乘以456等于多少？请用计算器算一下")]
})

# ---------------------- 打印 AI 的完整思考过程 ----------------------
# 遍历返回的所有消息（包含 AI 思考、调用工具、工具返回结果、最终回答）
for msg in result["messages"]:
    msg_type = type(msg).__name__  # 获取消息类型（用户/AI/工具）
    print(f"[{msg_type}] {msg.content}")  # 打印消息内容
    print("---")  # 分割线