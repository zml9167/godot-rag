# Godot Documentation RAG

[中文](README.md) | [English](README.en.md)

An experimental project for building a local retrieval index from Godot documentation. The pipeline converts RST documents to MyST Markdown, splits them into structural chunks, and builds BM25, API symbol, and Qdrant vector indexes.

## Project Structure

```text
.
├── docs-source/          # Source documentation; converted Markdown is also written here
├── scripts/
│   ├── convert.py        # RST -> MyST Markdown
│   ├── chunk.py          # Markdown -> JSONL chunks
│   └── build_index.py    # Build BM25, symbol, and vector indexes
├── data/
│   ├── chunks/chunk.jsonl
│   ├── bm25_index/
│   ├── symbol_index/
│   └── qdrant_data/
├── models/               # Local model cache
└── logs/                 # Conversion and chunking logs
```

`docs-source/`, `data/`, `models/`, and log files contain local inputs or generated data and are not tracked by Git.

## Environment Setup

Install the project dependencies in a Python environment:

```bash
pip install rst-to-myst chunkana rank-bm25 qdrant-client modelscope sentence-transformers
```

The vector indexing script uses `BAAI/bge-m3` with `device="cuda"`. Install a PyTorch build compatible with your local CUDA environment and make sure a GPU is available. The model is downloaded from ModelScope when the index is built for the first time.

## Run the Pipeline

Run these commands from the repository root, in order:

```bash
python scripts/convert.py
python scripts/chunk.py
python scripts/build_index.py
```

The conversion and chunking directories are configured through `DOCS_DIRS` in the scripts. Make sure the configured names match the actual directories under `docs-source/`.

The chunking script appends results to `data/chunks/chunk.jsonl`. Back up or clear the previous output before rerunning it to avoid duplicate records. Each line is a JSON object representing one chunk, with fields including `chunk_id`, `source`, `header_path`, `content`, `start_line`, `end_line`, `metadata`, and `chunk_type`.

The indexing script reads `data/chunks/chunk.jsonl` and writes local BM25, symbol, and Qdrant indexes under `data/`. Rebuilding deletes and recreates the Qdrant collection named `godot_docs`; make sure it does not contain data you need to keep.
