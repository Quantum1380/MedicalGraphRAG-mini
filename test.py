from dotenv import load_dotenv
import os

# 加载.env文件
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

loader = DirectoryLoader("./doc")
documents = loader.load()

doc_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
)
all_splits = doc_splitter.split_documents(documents)