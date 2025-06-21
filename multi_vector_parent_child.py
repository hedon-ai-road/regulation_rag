import warnings
warnings.filterwarnings('ignore')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from model import RagEmbedding
from langchain_chroma import Chroma
import chromadb

# 从文件中加载原始文档
import pickle
with open('./data/zhidu_db.pickl', 'rb') as f:
    doc_txts = pickle.load(f)

doc_ids = list(doc_txts.keys())
docs = list(doc_txts.values())
print(f"original docs count: {len(docs)}")


# 定义递归文本分块器
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=64,
                                                     chunk_overlap=15,
                                                     separators=["\n\n",
                                                                 "\n",
                                                                 ".",
                                                                 "\uff0e",
                                                                 "\u3002",
                                                                 ",",
                                                                 "\uff0c",
                                                                 "\u3001",
                                                                 ])
sub_docs = []
id_key = "doc_id"
index_type = 'small_chunk'

# 拆分子文档
for i, doc in enumerate(docs):
    _id = doc_ids[i]

    # 表格保持不变
    if doc.metadata["is_table"] == 1:
        _doc = Document(page_content=doc.page_content,
                        metadata={"type": index_type, id_key: _id})
        sub_docs.extend([_doc])
        continue

    # 对文本进行递归拆分
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
        _doc.metadata["type"] = index_type
    sub_docs.extend(_sub_docs)

print(f"sub docs count: {len(sub_docs)}")

# 使用子文档进行 embedding
chroma_client = chromadb.PersistentClient(path='./chroma')
embedding_model = RagEmbedding()
sm_chunk_db = Chroma.from_documents(sub_docs,
                  embedding_model.get_embedding_fun(),
                  client=chroma_client,
                  collection_name="zhidu_db_sm_chunk")



from langchain.storage import InMemoryByteStore # 存储原始文档分块和 id
from langchain.retrievers.multi_vector import MultiVectorRetriever # 多向量检索器

# 构建多索引检索器
store = InMemoryByteStore()
retriever = MultiVectorRetriever(
    vectorstore=sm_chunk_db,
    byte_store=store,
    id_key=id_key,
)
retriever.docstore.mset(list(zip(doc_ids, docs)))

query = "出差交通怎么算？"
docs = retriever.invoke(query)
print("#" * 100)
print("multi ventor:")
for doc in docs:
    print(doc)

print("#" * 100)
print("single vector:")
docs = sm_chunk_db.similarity_search(query=query, k=2)
for doc in docs:
    print(doc)