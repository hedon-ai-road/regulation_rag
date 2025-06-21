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
    
def query2doc(query):
    prompt = f"你是一名公司员工制度的问答助手，熟悉公司规章制度，请简短回答以下问题：{query}"
    doc_info = llm(prompt)
    context_query = f"{query}, {doc_info}"
    print("#"*20, 'query2doc')
    print(context_query)
    print("#"*20)
    return context_query

from langchain.chains.hyde.base import HypotheticalDocumentEmbedder

def hyde(query, include_query=True):
    prompt_template = """你是一名公司员工制度的问答助手，熟悉公司规章制度，请简短回答以下问题：
    Question: {question}
    Answer:"""

    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    embeddings = HypotheticalDocumentEmbedder(llm_chain= prompt | llm,
                                 base_embeddings=embedding_model.get_embedding_fun())
    hyde_embedding = embeddings.embed_query(query)

    if include_query:
        query_embeddings = embedding_model.get_embedding_fun().embed_query(query)
        result = (np.array(query_embeddings) + np.array(hyde_embedding)) / 2
        result = list(result)
    else:
        result = hyde_embedding
    result = list(map(float, result))
    return result

def sun_question(query):
    prompt_template = """你是一名公司员工制度的问答助手，熟悉公司规章制度。
    你的任务是对复杂问题继续拆解，以便理解员工的意图。
    请根据以下问题创建一个子问题列表：
    
    复杂问题：{question}

    请执行以下步骤：
    1. 识别主要问题：找出问题中的核心概念或主题。
    2. 分解成子问题：将主要问题分解成可以独立理解和解决的多个子问题。
    3. 只返回子问题列表，不包含其他解释信息，格式为：
        1. 子问题1
        2. 子问题2
        3. 子问题3
        ...
    
    """

    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

    llm_chain = prompt | llm
    sub_queries = llm_chain.invoke(query).split('\n')
    print("#"*20, 'sun_question')
    print(sub_queries)
    print("#"*20)
    return sub_queries

def question_rewrite(query):
    prompt_template = """你是一名公司员工制度的问答助手，熟悉公司规章制度。
    你的任务是需要为给定的问题，从不同层次生成这个问题的转述版本，使其更易于检索，转述的版本增加一些公司规章制度的关键词。
    问题：{question}
    请直接给出转述后的问题列表，不包含其他解释信息，格式为：
        1. 转述问题1
        2. 战术问题2
        3. 转述问题3
        ..."""

    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

    llmchain = prompt | llm
    rewrote_question = llmchain.invoke(query)
    print("#"*20, 'question_rewrite')
    print(rewrote_question)
    print("#"*20)
    return rewrote_question

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate

# 将复杂问题抽象化，使其更聚焦在本质问题上
def take_step_back(query):
    examples = [
        {
            "input": "我祖父去世了，我要回去几天",
            "output": "公司丧葬假有什么规定？",
        },
        {
            "input": "我去北京出差，北京的消费高，有什么额外的补助？",
            "output": "员工出差的交通费、住宿费、伙食补助费的规定是什么？"
        },
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一名公司员工制度的问答助手，熟悉公司规章制度。
                你的任务是将输入的问题通过归纳、提炼，转换为关于公司规章制度制定相关的一般性问题，使得问题更容易捕捉问题的意图。
                请参考下面的例子，按照同样的风格直接返回一个转述后的问题："""
            ),
            # few shot exmaples,
            few_shot_prompt,
            # new question
            ("user", "{question}")
        ]
    )

    question_gen = prompt | llm | StrOutputParser()
    res = question_gen.invoke({"question": query}).removeprefix("AI: ")
    print("#"*20, 'take_step_back')
    print(res)
    print("#"*20)
    return res

def run_rag_pipeline_basic(query):
    run_rag_pipeline(query=query, context_query=query)

def run_rag_pipeline_with_query2doc(query):
    context_query = query2doc(query=query)
    run_rag_pipeline(query=query, context_query=context_query)

def run_rag_pipeline_with_hyde(query):
    run_rag_pipeline(query=query, context_query=hyde(query=query), context_query_type="vector")

def run_rag_pipeline_with_sub_question(query):
    sub_queries = sun_question(query=query)
    for sq in sub_queries:
        run_rag_pipeline(sq, sq, context_query_type="query")

def run_rag_pipeline_with_question_rewrite(query):
    run_rag_pipeline(query=query, context_query=question_rewrite(query=query), context_query_type="query")

def run_rag_pipeline_with_take_step_back(query):
    run_rag_pipeline(query=query, context_query=take_step_back(query=query), context_query_type="query")


if __name__ == "__main__":
    query = "那个，我们公司有什么规定来着？"

    """
    根据提供的上下文信息，公司的主要规定包括：  

    1. **企业形象维护**：  
    - 员工必须严格遵守企业文化、经营理念和管理制度，不得有损学校形象和荣誉的行为。  
    - 如有违反，情节严重者可能面临警告、处分、解除劳动合同甚至法律追究。  

    2. **考勤制度**：  
    - 迟到或早退60分钟以上视为缺勤1天。  
    - 迟到、早退、脱岗累计超过3次（含），每次扣减工资50元。  
    - 连续旷工超过3天或全年累计旷工超过7天，无条件辞退。  

    3. **学校财产保护**：  
    - 损坏学校财产需酌情赔偿，盗窃行为将立即解除劳动合同并移交公安部门处理。  

    4. **值班要求**： 
    - 行政岗及教辅岗需参与法定节假日轮流值班。
    """
    # run_rag_pipeline_basic(query=query)

    """
    根据提供的上下文信息，公司的主要规定包括：

    1. **考勤规定**：
    - 迟到或早退60分钟以上视同缺勤1天。
    - 迟到、早退、脱岗累计超过3次（含），每次扣减工资50元。
    - 缺打卡或无有效签注且无证明的，视为旷工或早退半天。
    - 无工作理由超过上班时间到岗视为迟到，提前离校视为早退，未经批准中途离校视为旷工。

    2. **旷工情形**：
    - 未办理请假手续或请假未批准擅自离岗。
    - 提供虚假证明获得准假。
    - 迟到、早退或擅离岗位达1小时以上。
    - 谎报请假原因或伪造证明。
    - 事假未经批准。

    3. **休假类型**：
    - 包括事假、病假、婚假、丧假、产假、哺乳假、工伤假、调休。

    4. **其他规定**：
    - 行政岗及教辅岗需参与法定节假日轮流值班。
    - 考勤员徇私舞弊或弄虚作假的，按奖惩规定处理，严重者可能解聘。

    如需更详细的规定（如具体休假天数、报销标准等），请咨询HR。
    """
    # run_rag_pipeline_with_query2doc(query=query)

    """
    根据提供的上下文信息，公司有以下规定：

    1. **考勤规定**：
    - 工作时间：星期一至星期四 7:55-16:55，星期五 7:55-16:15。
    - 迟到或早退60分钟以上（含60分钟），每次视同缺勤1天。
    - 职工迟到、早退、脱岗累计超过3次的（含），从第1次起，每次扣减工资50元。
    - 无工作理由，超过上班时间到岗的，视为迟到；未到下班时间提前离校的，视为早退；中途未经批准离校，视为旷工。

    2. **休假规定**：
    - 休假分为以下八种：事假、病假、婚假、丧假、产假、哺乳假、工伤假、调休。

    3. **值班规定**：
    - 行政岗及教辅岗都需要参与法定节假日轮流值班。

    4. **违规处理**：
    - 考勤员徇私舞弊、弄虚作假的，按学校奖惩规定处理。情节严重的，予以通报批评直至解聘。

    如果还有其他具体问题，请进一步说明。
    """
    # run_rag_pipeline_with_hyde(query=query)

    # query = "最近发生了很多事情，有点感冒发烧，还要出差去上海，我可以请什么假？"
    # run_rag_pipeline_with_sub_question(query=query)

    # query = "我想了解一下，临时外出需要怎么申请？"
    # run_rag_pipeline_with_question_rewrite(query=query)

    query = "我有事外出，要怎么办？"
    run_rag_pipeline_with_take_step_back(query=query)