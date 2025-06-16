from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader(file_path="./file_load/fixtures/zhidu_travel.pdf")

docs = loader.load()

for doc in docs:
    print(doc)
    print("#" * 100)