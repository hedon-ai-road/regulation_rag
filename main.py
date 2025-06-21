import warnings
warnings.filterwarnings('ignore')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from model import RagEmbedding, RagLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import chromadb
import numpy as np

prompt_template = """
你是企业员工助手，熟悉公司考勤和报销标准等规章制度，需要根据提供的上下文信息context来回答员工的提问。\
请直接回答问题，如果上下文信息context没有和问题相关的信息，请直接回答[不知道,请咨询HR] \
问题：{question}
"{context}"
回答：
"""

chroma_client = chromadb.PersistentClient(path='./chroma')
embedding_model = RagEmbedding()

zhidu_db = Chroma("zhidu_db",
                  embedding_model.get_embedding_fun(),
                  client=chroma_client)

llm = RagLLM()

def run_rag_pipeline(query, context_query, k=3, context_query_type="query",
                     stream=True, prompt_template=prompt_template,
                     temperature=0.1):
    if context_query_type == "vector":
        related_docs = zhidu_db.similarity_search_by_vector(context_query, k=k)
    elif context_query_type == "query":
        related_docs = zhidu_db.similarity_search(context_query, k=k)
    elif context_query_type == "doc":
        related_docs = context_query
    else:
        related_docs = zhidu_db.similarity_search(context_query, k=k)
    context = "\n".join([f"上下文{i+1}: {doc.page_content} \n" \
                         for i, doc in enumerate(related_docs)])
    print()
    print()
    print("#"*100)
    print(f"query: {query}")
    print(f"context: {context}")
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template,
    )
    llm_prompt = prompt.format(question=query, context=context)

    if stream:
        response = llm.stream(llm_prompt)
        print("stream response:")
        for chunk in response:
            print(chunk, end='', flush=True)
        return ""
    else:
        response = llm(llm_prompt, stream=True, temperature=temperature)
        print(f"response: \n{response}")
        return response
    
if __name__ == "__main__":
    run_rag_pipeline("请假有什么规定?", "请假有什么规定?")