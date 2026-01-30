from dotenv import load_dotenv
import os

# 加载.env文件
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFPlumberLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PDFPlumberLoader("doc/Book_20250121.pdf")
documents = loader.load()

doc_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
)
all_splits = doc_splitter.split_documents(documents)
print("\n=== 文档分块结果 ===")
for i, chunk in enumerate(all_splits, 1):
    print(f"\n--- 第 {i} 个文档块 ---")
    print(f"内容: {chunk.page_content}")
    print(f"元数据: {chunk.metadata}")
    print("-" * 50)

# 设置嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 构建用户查询
question = "什么是房颤？"

# 在向量存储中搜索相关文档，并准备上下文内容
retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

print(docs_content + "\n")

# 构建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
                基于以下上下文，回答问题。如果上下文中没有相关信息，
                请说"我无法从提供的上下文中找到相关信息"。
                上下文: {context}
                问题: {question}
                回答:"""
                                          )

# 使用大语言模型生成
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    temperature=0.7,        # 控制输出的随机性
    max_tokens=2048,        # 最大输出长度
    api_key=os.getenv("DEEPSEEK_API_KEY")  # API-key
)
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)