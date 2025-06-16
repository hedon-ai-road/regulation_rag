from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("./file_load/fixtures/test_word.docx")
docs = loader.load()

for doc in docs:
    print(doc)
    print("#"*100)