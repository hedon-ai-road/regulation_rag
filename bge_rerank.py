import torch
import warnings
warnings.filterwarnings('ignore')

import jieba
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def jieba_preprocessing_func(text: str):
    return [word for word in jieba.cut(text) if word and word.strip()]

def get_optimal_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 使用 Apple Metal Performance Shaders (MPS) 加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 使用 CUDA 加速")
    else:
        device = torch.device("cpu")
        print("💻 使用 CPU")
    return device

# 从文件中加载原始文档
import pickle
with open('./data/zhidu_db.pickl', 'rb') as f:
    doc_txts = pickle.load(f)

doc_ids = list(doc_txts.keys())
docs = list(doc_txts.values())

# 初始化模型
model_name = 'BAAI/bge-reranker-v2-m3'
tokenizer = AutoTokenizer.from_pretrained(model_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 启动加速
device = get_optimal_device()
rerank_model = rerank_model.to(device)

# bm25
from langchain_community.retrievers import BM25Retriever
bm25_retriever = BM25Retriever.from_documents(documents=docs,
                                              preprocess_func=jieba_preprocessing_func)
bm25_retriever.k = 3

# 单独使用 bm25 检索
query = "我要请病假100天"
ret_docs = bm25_retriever.invoke(query)
for doc in ret_docs:
    print("#"*50)
    print(doc)
    print()

# 准备排序数据
pairs = []
for doc in ret_docs:
    pairs.append([query, doc.page_content])
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)

# 执行排序
with torch.no_grad():
    inputs = {key: inputs[key].to(device) for key in inputs.keys()}
    scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
doc_sort_ids = list((scores.cpu().numpy() * -1).argsort())
print(doc_sort_ids)

# 获取最相关的
import numpy as np
print(np.array(ret_docs)[doc_sort_ids][0])