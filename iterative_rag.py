iter_retgen_prompt_template = """
你是企业员工助手，熟悉公司考勤和报销标准等规章制度，需要参考提供的上下文信息context来回答员工的提问。\
请直接回答问题，如果上下文信息context没有和问题相关的信息，可尝试回答 \
问题：{question}
"{context}"
回答：
"""

from query_enhance import run_rag_pipeline, prompt_template

def iter_retgen(query, iter_num=2):
    iter_answer = None
    for i in range(iter_num):
        context_query = f"{query}, {iter_answer}" if iter_answer else query
        print("="*50, "context_query-", i, "="*50)
        print(context_query)

        if i < iter_num -1:
            iter_answer = run_rag_pipeline(query=query,
                                           context_query=context_query,
                                           prompt_template=iter_retgen_prompt_template,
                                           temperature=1.1)
            print("="*50, "iter-", i, "="*50)
            print(iter_answer)
        else:
            iter_answer = run_rag_pipeline(
                query=query,
                context_query=context_query,
                stream=True,
                prompt_template=prompt_template,
                temperature=0.1,
            )


query = "公司的假有哪些，规定是什么？"
iter_retgen(query)
    