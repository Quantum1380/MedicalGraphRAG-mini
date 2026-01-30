from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import re
import os

def load_and_split_by_page(file_path: str, encoding: str = "utf-8") -> list[Document]:
    """
    直接读取文本文件，按页码标识拆分，返回带page_number元数据的Document列表
    :param file_path: 文本文件路径
    :param encoding: 文件编码
    :return: 带页码的Document列表
    """
    # 1. 直接读取文本文件（替代TextLoader）
    with open(file_path, 'r', encoding=encoding) as f:
        full_text = f.read().strip()

    # 2. 按页码标识拆分（适配## 第 N 页格式）
    page_docs = []
    page_pattern = re.compile(r'## 第\s*(\d+)\s*页', re.M)
    parts = page_pattern.split(full_text)

    # 3. 构建带页码的Document对象
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        page_num = int(parts[i].strip())
        page_content = parts[i + 1].strip()
        # 手动创建Document，添加页码元数据（核心）
        doc = Document(
            page_content=page_content,
            metadata={
                "source": file_path,  # 保留文件来源（和TextLoader一致）
                "page_number": page_num  # 新增页码元数据
            }
        )
        page_docs.append(doc)
    return page_docs

documents = load_and_split_by_page("mark/Book_20250121.txt", encoding="utf-8")
# 定义分割符列表，按优先级依次使用
separators = ["\n\n", ".", "，", " "] # . 是句号，， 是逗号， 是空格
# 创建递归分块器，并传入分割符列表
text_splitter = CharacterTextSplitter(
    chunk_size=100,  # 每个文本块的大小为50个字符
    chunk_overlap=10,  # 文本块之间没有重叠部分
)
all_splits = text_splitter.split_documents(documents)
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
    temperature=0.7,  # 控制输出的随机性
    max_tokens=2048,  # 最大输出长度
    api_key=os.getenv("DEEPSEEK_API_KEY")  # API-key
)
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)