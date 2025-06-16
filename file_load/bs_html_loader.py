from langchain_community.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("./file_load/fixtures/test.html")
docs = loader.load()

for doc in docs:
    print("#"*100)
    print(doc)