from langchain_community.document_loaders import UnstructuredPowerPointLoader

loader = UnstructuredPowerPointLoader(file_path="./file_load/fixtures/test_ppt.pptx")

docs = loader.load()

for doc in docs:
    print(doc)
    print("#" * 100)