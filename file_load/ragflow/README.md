# RAGFlow PDF Parser (Simplified Version)

This is a simplified version of [RAGFlow](https://github.com/infiniflow/ragflow) that focuses on PDF manual chunk processing functionality.

## What's Included

- PDF parsing with OCR
- Layout recognition
- Table structure recognition
- Text merging algorithms
- Core deepdoc functionality

## What's Removed

- Web UI and API services
- Database components
- LLM integration
- Non-PDF parsers (Excel, PPT, etc.)
- Benchmark and demo files

## Original License

This work is based on RAGFlow, which is licensed under the Apache License 2.0.
See the [LICENSE](LICENSE) file for details.

## Usage

```python
from ragflow.rag.app import manual as deepdoc_manual

def dummy(prog=None, msg=""):
    print(prog, msg)

res = deepdoc_manual.chunk(filename="your_pdf_file.pdf", callback=dummy)
for data in res:
    print(data['content_with_weight'])
```
