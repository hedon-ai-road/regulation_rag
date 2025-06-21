"""
uv add rank_bm25
uv add jieba
"""

import warnings
warnings.filterwarnings('ignore')

import jieba
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 从文件中加载原始文档
import pickle
with open('./data/zhidu_db.pickl', 'rb') as f:
    doc_txts = pickle.load(f)

doc_ids = list(doc_txts.keys())
docs = list(doc_txts.values())

# jieba 分词处理函数
def jieba_preprocessing_func(text: str):
    return [word for word in jieba.cut(text) if word and word.strip()]

text = "我要请病假 100 天"
print(jieba_preprocessing_func(text=text))

# 稀疏索引：bm25 检索器
bm25_retriever = BM25Retriever.from_documents(documents=docs,
                                              preprocess_func=jieba_preprocessing_func)
bm25_retriever.k = 3

ret_docs = bm25_retriever.invoke(text)
for doc in ret_docs:
    print(doc)

# 稠密索引
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import chromadb
from model import RagEmbedding
chroma_client = chromadb.PersistentClient(path='./chroma')
embedding_model = RagEmbedding()
zhidu_db = Chroma("zhidu_db",
                  embedding_model.get_embedding_fun(),
                  client=chroma_client)

embedding_retriever = zhidu_db.as_retriever(search_kwargs={"k": 3}) # 转为检索器

# 融合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, embedding_retriever],
    weights=[0.3, 0.7],
)

query = "公司丧假有什么规定？"

docs = embedding_retriever.invoke(query, k=3)
print("#"* 100)
print("ensemble:")
for doc in docs:
    print(doc)
print()

docs = bm25_retriever.invoke(query)
print("#"* 100)
print("bm25:")
for doc in docs:
    print(doc)
print()

docs = embedding_retriever.invoke(query)
print("#"* 100)
print("embedding:")
for doc in docs:
    print(doc)
print()