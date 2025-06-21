from ragflow.rag.app import manual as deepdoc_manual

# should download wordnet first
# import nltk
# nltk.set_proxy('http://127.0.0.1:7890')
# nltk.download('wordnet')

def dummy(prog=None, msg=""):
    print(prog, msg)

res = deepdoc_manual.chunk(filename="./file_load/fixtures/zhidu_travel.pdf",
                        callback=dummy)
for data in res:
    print("="*100)
    print(data)
    print("="*100)
    print(data['content_with_weight'])
