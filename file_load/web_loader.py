from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(web_path="https://hedon.top/2025/04/13/ai-rag-tech-overview/")
docs = loader.load()

for doc in docs:
    print("#"*100)
    print(doc)