from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic.v1 import BaseModel, Field, validator
from langchain_core.output_parsers import PydanticOutputParser
from model import RagLLM

# 从文件中加载原始文档
import pickle
with open('./data/zhidu_db.pickl', 'rb') as f:
    doc_txts = pickle.load(f)

doc_ids = list(doc_txts.keys())
docs = list(doc_txts.values())


# 定义处理链路
llm = RagLLM()
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template(
        """
        你是企业员工助手，熟悉公司考勤制度和报销标准等规章制度。
        你的任务是提出可以在下面文档的内容中可以找到答案的 3 个假设性问题。
        \n\n{doc}\n\n
        要求输出为中文，不包含解释性内容，格式位列表格式，如 ['问题1', '问题2', '问题3']
        """
    )
    | llm
)

# 生成假设性问题
class HypotheticalQuestion(BaseModel):
    """Generate hypothetical questions."""

    questions: List[str] = Field(..., description="List of questions")

question_docs = []
id_key = "doc_id"
index_type = 'hq'
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    for _ in range(3):
        try:
            hq = chain.invoke(doc)

            # 通过这种方式来强制检查 LLM 返回的数据格式是合法的
            res = eval(hq)
            q = HypotheticalQuestion(questions=res)

            for i, question in enumerate(q.questions):
                question_docs.extend([
                    Document(page_content=question, metadata={"type": index_type, id_key: _id})
                ])
            break
        except:
            continue


# embedding
from model import RagEmbedding
from langchain_chroma import Chroma
import chromadb

chroma_client = chromadb.PersistentClient(path='./chroma')
embedding_model = RagEmbedding()
hq_chunk_db = Chroma.from_documents(question_docs,
                  embedding_model.get_embedding_fun(),
                  client=chroma_client,
                  collection_name="zhidu_db_hq")

# 构建多向量检索器
from langchain.storage import InMemoryByteStore # 存储原始文档分块和 id
from langchain.retrievers.multi_vector import MultiVectorRetriever # 多向量检索器

store = InMemoryByteStore()
id_key = "doc_id"

hq_retriever = MultiVectorRetriever(
    vectorstore=hq_chunk_db,
    byte_store=store,
    id_key=id_key
)
hq_retriever.docstore.mset(list(zip(doc_ids, docs)))


# 检索
query = "出差交通费怎么算？"
docs = hq_retriever.invoke(query)
print("#" * 100)
print("multi ventor:")
for doc in docs:
    print(doc)

print("#" * 100)
print("single vector:")
docs = hq_chunk_db.similarity_search(query=query, k=2)
for doc in docs:
    print(doc)