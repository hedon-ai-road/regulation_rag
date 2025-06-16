from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(file_path="./file_load/fixtures/zhidu_travel.pdf")

docs = loader.load()

for doc in docs:
    print(doc)
    print("#" * 100)