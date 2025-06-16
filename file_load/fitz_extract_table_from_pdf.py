import fitz

doc = fitz.open("./file_load/fixtures/zhidu_travel.pdf")

table_data = []
text_data = []

doc_tables = []
for idx, page in enumerate(doc):
    text = page.get_text()
    text_data.append(text)
    tabs = page.find_tables()
    for i, tab in enumerate(tabs):
        ds = tab.to_pandas()
        table_data.append(ds.to_markdown())

for tab in table_data:
    print(tab)
    print("="*100)