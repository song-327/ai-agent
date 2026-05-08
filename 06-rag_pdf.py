import os

# 设置 HuggingFace 镜像（放在最前面）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. 配置区 ---
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
PDF_FILE = "./data/ML.pdf"  # 你的PDF文件名

# --- 2. 复用你成功的关键决策：本地Embedding模型 ---
print("正在加载本地Embedding模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)
print("模型加载完成")

# --- 3. 加载并处理PDF ---
print(f"正在加载PDF: {PDF_FILE} ...")
loader = PyPDFLoader(PDF_FILE)
documents = loader.load()
print(f"加载完成，共{len(documents)}页")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)
print(f"文档已切分为 {len(docs)} 个片段")

# --- 4. 创建向量数据库 ---
print("正在创建向量数据库...")
vectordb = Chroma.from_documents(docs, embeddings)
print("向量数据库创建完成")

# --- 5. 准备大模型 ---
llm = ChatOpenAI(
    model="deepseek-v4-pro",  # 修正模型名
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 6. 创建问答链(开启来源返回) ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 8}),  # 增加到8个片段，提高准确度
    return_source_documents=True
)

# --- 7. 交互式问答（增强版）---
print("\n" + "=" * 50)
print("🤖 RAG知识库问答系统已就绪！")
print("=" * 50 + "\n")

# 可选：先测试一个问题
print("📖 测试提问：这本书讲了什么？")
result = qa_chain.invoke({"query": "这本书讲了什么？用中文回答"})

print("\n🤖 回答：")
print(result["result"])
print("\n📖 答案来源页数：")
seen_pages = set()
for doc in result["source_documents"]:
    page = doc.metadata.get("page", 0)
    if page not in seen_pages:
        print(f"  📄 第 {page + 1} 页")
        seen_pages.add(page)
print("-" * 50)

# 进入交互式问答循环
while True:
    question = input("\n请输入你的问题 (输入 'q' 退出): ")
    if question.lower() == 'q':
        break

    if not question.strip():
        continue

    print("🤔 思考中...")

    # 执行问答
    result = qa_chain.invoke({"query": question})


    # 打印回答
    print(f"\n🤖 回答: {result['result']}\n")
    # 打印来源 (优化显示)
    print("📖 答案参考来源:")
    seen_pages = set()  # 防止重复显示同一页
    for doc in result["source_documents"]:
        page = doc.metadata.get("page", 0)
        if page not in seen_pages:
            # 只显示文档片段的前100字，而不是全部
            snippet = doc.page_content[:100].replace("\n", " ")
            print(f"  📄 第 {page + 1} 页: \"{snippet}...\"")
            seen_pages.add(page)
    print("-" * 50)