from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from model import RagLLM

# 从文件中加载原始文档
import pickle
with open('./data/zhidu_db.pickl', 'rb') as f:
    doc_txts = pickle.load(f)

doc_ids = list(doc_txts.keys())
docs = list(doc_txts.values())

# 构建摘要处理链路
llm = RagLLM()
prompt_template = "你是企业员工助手，熟悉公司考勤和报销标准等规章制度。请根据下面的文档：\n\n{doc}\n\n 在保持语义完整的前提下，做一个简短的概括摘要改写，50~100 个字左右，并给出关键词，以提高 RAG 检索质量"
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template(prompt_template)
    | llm
    | StrOutputParser()
)

# 处理摘要
summaries = chain.batch(docs, {"max_concurrency": 1})
for i, doc in enumerate(docs):
    print("#" * 100)
    print(f"original: {doc.page_content}")
    print(f"summary: {summaries[i]}")

# 聚合成 langchain Document
summary_docs = []
id_key = "doc_id"
index_type = "summary"
for i, s in enumerate(summaries):
    _id = doc_ids[i]
    doc = docs[i]
    if doc.metadata["is_table"] == 1:
        _doc = Document(page_content=doc.page_content,
                        metadata={"type": index_type, id_key: _id})
        summary_docs.extend([_doc])
        continue

    _s = Document(page_content=s,
                  metadata={"type": index_type, id_key: _id})
    summary_docs.extend([_s])

# embedding
from model import RagEmbedding
from langchain_chroma import Chroma
import chromadb

chroma_client = chromadb.PersistentClient(path='./chroma')
embedding_model = RagEmbedding()
summary_chunk_db = Chroma.from_documents(summary_docs,
                  embedding_model.get_embedding_fun(),
                  client=chroma_client,
                  collection_name="zhidu_db_summary")

# 多向量检索
from langchain.storage import InMemoryByteStore # 存储原始文档分块和 id
from langchain.retrievers.multi_vector import MultiVectorRetriever # 多向量检索器

store = InMemoryByteStore()
id_key = "doc_id"

summary_retriever = MultiVectorRetriever(
    vectorstore=summary_chunk_db,
    byte_store=store,
    id_key=id_key
)
summary_retriever.docstore.mset(list(zip(doc_ids, docs)))

# 检索
query = "出差交通费怎么算？"
docs = summary_retriever.invoke(query)
print("#" * 100)
print("multi ventor:")
for doc in docs:
    print(doc)

print("#" * 100)
print("single vector:")
docs = summary_chunk_db.similarity_search(query=query, k=2)
for doc in docs:
    print(doc)