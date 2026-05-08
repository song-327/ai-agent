"""
LangChain链式调用跑通	prompt | llm 这行代码，就是现在工业界搭建大模型应用的骨架
提示词工程实践	你让AI扮演“毒舌代码审查员”，它真的用犀利风格回复了——这就是提示词控制模型行为的威力
"""

# ===================== 1. 导入需要的库 =====================
# 用 LangChain 调用大模型（ChatOpenAI 兼容 DeepSeek）
from langchain_openai import ChatOpenAI

# 提示词模板（固定说话格式，不用每次手写）
from langchain_core.prompts import ChatPromptTemplate

# 读取环境变量，安全拿 API Key（不泄露）
import os

# ===================== 2. 读取 API Key =====================
# 从你电脑系统里拿密钥
dp_key = os.getenv("DEEPSEEK_KEY")

# ===================== 3. 初始化大模型 =====================
llm = ChatOpenAI(
    model="deepseek-v4-flash",    # 用的模型
    api_key=dp_key,               # 你的密钥
    base_url="https://api.deepseek.com"  # 官方接口
)

# ===================== 4. 做一个【提示词模板】 =====================
# 作用：固定 AI 的身份 + 说话风格
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个毒舌但专业的代码审查员，说话要犀利但一针见血。"),
    ("human", "帮我审查这段代码：\n{code}")
])

# ===================== 5. 链式调用：模板 + 模型 =====================
chain = prompt | llm

# ===================== 6. 准备一段有问题的代码 =====================
bad_code = """
def f(x):
    return x + 1
    print("done")
"""

# ===================== 7. 把代码丢给 AI 审查 =====================
result = chain.invoke({"code": bad_code})

# ===================== 8. 输出 AI 的审查结果 =====================
print(result.content)