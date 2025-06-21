# 企业制度问答 RAG 系统

一个基于多种先进 RAG 技术的企业制度智能问答系统，支持多种文档格式解析和多维度检索增强。

## 🚀 核心特性

### 📊 多样化文档处理

- **PDF 文档**: 支持文本、表格、图像的智能解析
- **Excel 表格**: 结构化数据提取与处理
- **Word 文档**: 富文本内容解析
- **PowerPoint**: 演示文稿内容提取
- **HTML 网页**: 在线内容爬取与解析

### 🧠 先进的 RAG 优化技术

#### 1. Self-RAG (自反思检索增强生成)

- **技术原理**: 通过多步骤判断流程优化检索和生成质量
- **核心组件**:
  - 文档相关性评估 (grade_documents)
  - 答案支撑度验证 (answer_supported)
  - 答案有用性检查 (answer_useful)
  - 查询重写机制 (query_rewrite)
- **工作流**: 检索 → 文档评估 → 生成 → 质量评估 → 迭代优化
- **实现文件**: `self_rag.py`

#### 2. 多向量检索 (Multi-Vector Retrieval)

- **父子文档检索**: 小块索引，大块返回，平衡检索精度和上下文完整性
- **文档摘要检索**: 基于摘要进行检索，返回原始文档
- **假设问题生成**: 为每个文档生成假设问题，提高问答匹配度
- **实现文件**:
  - `multi_vector_parent_child.py` - 父子文档策略
  - `multi_vector_summary.py` - 摘要检索策略
  - `multi_vector_hypothetical_question.py` - 假设问题策略

#### 3. 查询增强技术 (Query Enhancement)

- **Query2Doc**: 将查询转换为文档形式，增强语义匹配
- **HyDE**: 假设性文档嵌入，生成理想答案的向量表示
- **查询重写**: 多角度重新表述问题，提高检索召回率
- **子问题分解**: 复杂问题拆解为多个子问题
- **Step-Back Prompting**: 将具体问题抽象为一般性问题
- **实现文件**: `query_enhance.py`

#### 4. 集成检索 (Ensemble Retrieval)

- **稀疏检索**: BM25 关键词匹配，基于 jieba 分词
- **稠密检索**: BGE-M3 语义向量检索
- **权重融合**: 0.3 (BM25) + 0.7 (向量检索) 的最优权重配置
- **实现文件**: `ensemble_retriever.py`

#### 5. 重排序优化 (Re-ranking)

- **模型**: BAAI/bge-reranker-v2-m3
- **功能**: 对初步检索结果进行精细化排序
- **硬件加速**: 支持 CUDA/MPS/CPU 多种运行环境
- **实现文件**: `bge_rerank.py`

#### 6. 迭代检索生成 (Iterative RAG)

- **多轮优化**: 基于前一轮答案优化后续检索
- **温度调节**: 探索阶段使用高温度(1.1)，最终生成使用低温度(0.1)
- **实现文件**: `iterative_rag.py`

### 🛠 技术架构

#### 核心模块

- **模型层** (`model.py`): 统一的 LLM 和 Embedding 接口
  - LLM: DeepSeek Chat V3 (通过 OpenRouter)
  - Embedding: BAAI/bge-m3
- **文档解析** (`file_load/`): 多格式文档处理能力
- **向量存储**: ChromaDB 持久化存储
- **评估系统** (`eval.ipynb`): RAGAS 评估框架

#### 评估指标

- **Context Relevancy (CR)**: 检索相关性
- **Answer Relevancy (AR)**: 答案相关性
- **Faithfulness (F)**: 可信度评估

## 📦 安装与配置

### 环境要求

```bash
# 安装文档解析依赖
brew install poppler tesseract libmagic ghostscript pandoc

# Python 环境 (使用 uv 包管理器)
uv add rank_bm25 jieba langchain chromadb transformers torch
```

### 环境变量配置

```bash
# .env 文件
OPENROUTER_API_KEY=your_openrouter_api_key
```

## 🚀 使用方法

### 基础 RAG

```python
from query_enhance import run_rag_pipeline_basic
run_rag_pipeline_basic("公司的考勤制度是什么？")
```

### Self-RAG

```python
from self_rag import app
inputs = {"keys": {"question": "我要请病假100天，有什么规定？"}}
for output in app.stream(inputs):
    pass
print(output['generate']['keys']['generation'])
```

### 查询增强 RAG

```python
from query_enhance import (
    run_rag_pipeline_with_query2doc,
    run_rag_pipeline_with_hyde,
    run_rag_pipeline_with_sub_question,
    run_rag_pipeline_with_question_rewrite,
    run_rag_pipeline_with_take_step_back
)

# Query2Doc 增强
run_rag_pipeline_with_query2doc("出差的交通费报销标准？")

# HyDE 增强
run_rag_pipeline_with_hyde("公司的请假流程？")

# 子问题分解
run_rag_pipeline_with_sub_question("新员工入职需要了解哪些制度？")
```

### 集成检索

```python
from ensemble_retriever import ensemble_retriever
docs = ensemble_retriever.invoke("公司丧假有什么规定？")
```

### 多向量检索

```python
# 父子文档检索
from multi_vector_parent_child import retriever as parent_child_retriever
docs = parent_child_retriever.invoke("出差交通费怎么算？")

# 摘要检索
from multi_vector_summary import summary_retriever
docs = summary_retriever.invoke("出差交通费怎么算？")

# 假设问题检索
from multi_vector_hypothetical_question import hq_retriever
docs = hq_retriever.invoke("出差交通费怎么算？")
```

## 📊 项目结构

```
regulation_rag/
├── README.md                          # 项目文档
├── model.py                          # 统一模型接口
├── self_rag.py                       # Self-RAG 实现
├── query_enhance.py                  # 查询增强技术
├── ensemble_retriever.py             # 集成检索器
├── bge_rerank.py                     # BGE 重排序
├── iterative_rag.py                  # 迭代 RAG
├── multi_vector_*.py                 # 多向量检索策略
├── doc_parse.py                      # 文档解析
├── eval.ipynb                        # 评估实验
├── baseline.ipynb                    # 基线对比
├── file_load/                        # 文档加载器
│   ├── ragflow/                      # RAGFlow 深度文档解析
│   ├── bs_html_loader.py            # HTML 解析
│   ├── openpyxl_excel.py            # Excel 处理
│   ├── pymupdf_loader.py            # PDF 解析
│   └── unstructured_*.py            # 非结构化文档处理
├── data/                             # 数据目录
│   ├── zhidu_db.pickl               # 处理后的文档数据
│   └── *.pdf, *.xlsx                # 原始文档
├── chroma/                           # 向量数据库
└── tests/                            # 测试用例
```

## 🎯 技术亮点

1. **模块化设计**: 每种 RAG 技术独立实现，便于组合使用
2. **多策略融合**: 支持多种检索和生成策略的灵活配置
3. **企业级适配**: 针对企业制度文档的专门优化
4. **评估体系**: 完整的 RAG 效果评估流程
5. **硬件优化**: 支持多种硬件加速方案

## 📈 性能优化

- **向量检索**: BGE-M3 多语言语义理解
- **关键词检索**: jieba 分词 + BM25 精确匹配
- **重排序**: BGE-reranker-v2-m3 二次精排
- **缓存机制**: ChromaDB 持久化存储
- **并行处理**: 多文档并发处理能力

## 🔧 扩展指南

### 添加新的文档类型

1. 在 `file_load/` 目录下创建新的加载器
2. 实现统一的文档解析接口
3. 更新 `doc_parse.py` 中的调度逻辑

### 集成新的 RAG 技术

1. 创建独立的技术实现文件
2. 继承现有的接口规范
3. 在 `query_enhance.py` 中添加调用入口

### 优化检索策略

1. 调整 `ensemble_retriever.py` 中的权重配置
2. 在 `bge_rerank.py` 中试验不同的重排序模型
3. 通过 `eval.ipynb` 评估效果改进

---

## Help

### unstructured:PDF text extraction failed, skip text extraction

```bash
brew install poppler tesseract libmagic ghostscript pandoc
```
