# Godot 文档 RAG

基于 Godot 文档构建本地检索索引的实验项目。处理流程为：将 RST 文档转换为 MyST Markdown，按结构切分为 chunks，再构建 BM25、API 符号和 Qdrant 向量索引。

## 项目结构

```text
.
├── docs-source/          # 文档源文件；RST 转换后也写入 Markdown
├── scripts/
│   ├── convert.py        # RST -> MyST Markdown
│   ├── chunk.py          # Markdown -> JSONL chunks
│   └── build_index.py    # 构建 BM25、符号和向量索引
├── data/
│   ├── chunks/chunk.jsonl
│   ├── bm25_index/
│   ├── symbol_index/
│   └── qdrant_data/
├── models/               # 本地模型缓存
└── logs/                 # 转换和切分日志
```

`docs-source/`、`data/`、`models/` 和日志目录属于本地生成或输入数据，不纳入版本控制。

## 环境准备

使用 Python 环境安装项目依赖：

```bash
pip install rst-to-myst chunkana rank-bm25 qdrant-client modelscope sentence-transformers
```

向量索引脚本使用 `BAAI/bge-m3`，并指定 `device="cuda"`。请先安装与本机 CUDA 环境匹配的 PyTorch，并确认 GPU 可用；首次构建索引时会从 ModelScope 下载模型。

## 运行流程

在仓库根目录依次运行：

```bash
python scripts/convert.py
python scripts/chunk.py
python scripts/build_index.py
```

转换和切分目录由脚本中的 `DOCS_DIRS` 配置。运行前请确认它与 `docs-source/` 下的实际目录名称一致。

切分脚本将结果追加写入 `data/chunks/chunk.jsonl`。重新切分前，请先备份或清理旧输出，避免重复数据。每行是一个 chunk 的 JSON 对象，包含 `chunk_id`、`source`、`header_path`、`content`、`start_line`、`end_line`、`metadata` 和 `chunk_type`。

索引脚本读取 `data/chunks/chunk.jsonl`，在 `data/` 下生成 BM25、符号和 Qdrant 本地索引。重建时会删除并重新创建名为 `godot_docs` 的 Qdrant collection；执行前请确认其中没有需要保留的数据。
