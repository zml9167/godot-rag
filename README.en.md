# Godot Documentation RAG

[中文](README.md) | [English](README.en.md)

An experimental project that builds a local retrieval index from the official Godot documentation. The pipeline converts RST documents to MyST Markdown, splits them into structural chunks, and builds BM25, API symbol, and Qdrant vector indexes. Retrieval combines symbol lookup, dense vectors, and BM25 to provide context for an LLM, with a companion evaluation script for both retrieval and generation.

## Project Structure

```text
.
├── godot-docs-master/         # Documentation cloned from godotengine/godot-docs
├── evaluate/
│   └── questions.jsonl        # Evaluation question set
├── scripts/
│   ├── run_pipeline.py        # Run conversion, chunking, and indexing
│   ├── rst2md.py              # RST -> MyST Markdown
│   ├── chunk_documents.py     # Markdown -> data/chunks.jsonl
│   ├── build_index.py         # Build vector, BM25, and symbol indexes
│   ├── index_builder/
│   │   ├── qdrant.py          # Qdrant vector index
│   │   ├── bm25.py            # BM25 index
│   │   └── symbol.py          # API symbol index
│   ├── retrieve.py            # Hybrid retrieval and RRF fusion
│   ├── generate.py            # LLM answer generation
│   ├── ask.py                 # Retrieval + generation entry point
│   ├── evaluate.py            # Retrieval and generation evaluation
│   ├── ip.py                  # Detect the current IP's country
│   ├── tokenizer.py           # BM25 tokenizer
│   └── paths.py               # Project root resolution
├── data/
│   ├── chunks.jsonl
│   ├── bm25_index/
│   ├── symbol_index/
│   └── qdrant_data/
├── models_cache/              # Embedding model cache
└── logs/                      # Stage logs
```

`godot-docs-master/`, `data/`, `models_cache/`, `logs/`, and `chunk_mismatch_reports/` contain local inputs or generated data and are not tracked by Git.

## Environment Setup

Create a virtual environment at the repository root and install the dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` does not include PyTorch, so install a build compatible with your local CUDA environment separately. The device is chosen automatically when building indexes: `cuda` when a GPU is available, otherwise `cpu`.

The embedding model is `BAAI/bge-m3`. On first use, the download source depends on the current IP's country: mainland China (`CN`) uses ModelScope, other regions use Hugging Face, and files are cached under `models_cache/`.

> Run every command from the repository root using `python -m scripts.<module>`. The scripts use package-relative imports, so running them as `python scripts/xxx.py` will fail.

## Building the Index

```bash
python -m scripts.run_pipeline
```

The script first verifies that every pipeline script exists and is non-empty, then runs these stages in order:

1. `scripts.rst2md`: clones `godotengine/godot-docs` (shallow) into `godot-docs-master/` if it is missing, then converts RST files under `DOCS_DIRS` to `.md` files alongside them.
2. `scripts.chunk_documents`: splits the Markdown structurally and writes `data/chunks.jsonl`.
3. `scripts.build_index`: builds the Qdrant, BM25, and symbol indexes.

If any stage fails, the remaining stages are skipped.

Notes:

- The documentation repository is cloned only on the first run and is **not** updated automatically. To update the docs, run `git pull` inside `godot-docs-master/`.
- The directories to convert and chunk are configured by `DOCS_DIRS` in `scripts/rst2md.py` and `scripts/chunk_documents.py`. The defaults are `about`, `classes`, `community`, `engine_details`, `getting_started`, and `tutorials`; make sure they exist under `godot-docs-master/`.
- Chunking uses the `chunkana` `structural` strategy with `max_chunk_size` set to 8192. `data/chunks.jsonl` is rebuilt on every run.
- If the reconstructed content differs from the source, a warning is logged and the output is saved to `chunk_mismatch_reports/` for inspection.

Each chunk is one JSON line in `data/chunks.jsonl` with these fields:

| Field | Description |
| --- | --- |
| `chunk_id` | Unique ID in the form `source:start_line-end_line` |
| `source` | Document path relative to `godot-docs-master/` |
| `header_path` | Heading path of the section |
| `content` | Chunk body |
| `start_line` / `end_line` | Line range in the source file |
| `metadata` | Chunking strategy, content type, whether it contains code, and similar metadata |
| `chunk_type` | Top-level documentation directory |

## Index Details

`scripts.build_index` calls the three modules under `scripts/index_builder/`:

- **Qdrant**: generates 1024-dimensional normalized vectors with `bge-m3`, uses cosine distance, and persists to `data/qdrant_data` under the collection name `godot_docs`. Rebuilding deletes and recreates the collection, so make sure it does not contain data you need to keep. Chunk IDs are hashed with UUIDv5 to form point IDs.
- **BM25**: uses `rank_bm25`'s `BM25Okapi`. Tokenization lives in `scripts/tokenizer.py` and keeps only lowercase English identifiers and numbers, matching the retrieval side. Outputs `bm25.pkl`, `chunk_ids.json`, and `meta.json` under `data/bm25_index/`.
- **Symbol index**: extracts API symbols from the last heading level, backticked identifiers, and CamelCase names, then builds a lowercase "symbol -> chunk list" mapping in `data/symbol_index/symbol_index.json`.

Each stage writes its log under `logs/`.

## Retrieval and Question Answering

The retrieval flow in `scripts/retrieve.py`:

1. Detect symbols in the query and look them up in the symbol index, taking up to 3 chunks per symbol.
2. Fetch the top 30 vector results and the top 30 BM25 results.
3. Fuse the two rankings with RRF (`k=60`), place symbol hits first, and return `top_k` chunks (default 6).

`scripts/ask.py` refuses to answer when retrieval returns nothing. Otherwise it builds a context with source labels from the chunks, calls the LLM, and returns a de-duplicated source list.

Command-line example:

```bash
python -m scripts.ask "How should I pass the delta parameter to move_and_slide?"
```

Generation uses an OpenAI-compatible API configured in `scripts/generate.py`: `api_key` is currently a placeholder and must be replaced before running; `base_url` defaults to `https://api.deepseek.com`, and the model defaults to `deepseek-chat`. The prompt requires answers to rely only on the retrieved chunks, cite `[来源: section path]`, and state clearly when the context is insufficient.

## Evaluation

```bash
python -m scripts.evaluate
```

The evaluation first validates that `expected_source` and `expected_section` in `evaluate/questions.jsonl` actually exist, and stops if validation fails. It then reports:

- **Retrieval**: per language, `Hit@6` (whether the expected source appears in the results) and `Symbol Hit@1` (whether the first result for API questions contains the expected symbol).
- **Generation**: per language, source citation rate, API name retention rate, and refusal accuracy on unanswerable questions.

Failures are written to `evaluate/misses.json`. Each line of the question set is a JSON object with fields including `id`, `lang`, `query`, `type` (such as `api`, `mixed`, or `unanswerable`), `expected_symbol`, `expected_source`, and `expected_section`.

## Notes

- This is a local experimental project with no web service or frontend; retrieval and QA are invoked through Python modules.
- Scripts locate data through `PROJECT_DIR` in `scripts/paths.py`, so keep the relative paths and directory layout consistent.
- Documentation content is copyright its Godot Engine documentation contributors; follow their license when reusing it.
