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
│   ├── build_all_index.py     # Build vector, BM25, and symbol indexes in parallel, then validate
│   ├── retrieve.py            # Hybrid retrieval and RRF fusion
│   ├── generate.py            # LLM answer generation
│   ├── ask.py                 # Retrieval + generation entry point
│   ├── evaluate.py            # Retrieval and generation evaluation
│   └── libs/
│       ├── utils.py           # Path constants and API client singletons
│       ├── siliconflow.py     # SiliconFlow embeddings/rerank API client
│       ├── tokenizer.py       # BM25 tokenizer
│       ├── pattern_utils.py   # API symbol extraction patterns
│       └── index_builder/
│           ├── qdrant.py      # Qdrant vector index
│           ├── bm25.py        # BM25 index
│           ├── symbol.py      # API symbol index
│           └── validate.py    # Index integrity validation
├── data/
│   ├── chunks.jsonl
│   ├── bm25_index/
│   ├── symbol_index/
│   └── qdrant_data/
└── logs/                      # Stage logs
```

`godot-docs-master/`, `data/`, and `logs/` contain local inputs or generated data and are not tracked by Git.

## Environment Setup

Create a virtual environment at the repository root and install the dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` does not include PyTorch: both embedding and reranking now go through the SiliconFlow API, so this project no longer depends on a GPU, PyTorch, or local model weights.

**Embedding and reranking depend on the SiliconFlow API; export the key first:**

```bash
export SILICONFLOW_API_KEY="your-api-key"
```

If the key is missing, index building or retrieval fails when the client is initialized. Related environment variables (all optional, defaults in parentheses): embedding model name `GODOT_RAG_EMBEDDING_MODEL` (`BAAI/bge-m3`), reranker model name `GODOT_RAG_RERANKER_MODEL` (`BAAI/bge-reranker-v2-m3`), API base URL `SILICONFLOW_API_BASE` (`https://api.siliconflow.cn/v1`), embedding items per request `GODOT_RAG_API_BATCH_SIZE` (`32`, hard API limit), per-request token budget `GODOT_RAG_API_BATCH_TOKENS` (`24576`), per-text token limit `GODOT_RAG_EMBEDDING_MAX_TOKENS` (`8176`; bge-m3 accepts up to 8192 tokens per item — oversized chunks are truncated precisely by token using the bge-m3 tokenizer before encoding, with a logged warning; `tokenizer.json` is downloaded automatically from ModelScope/HF into `models_cache/tokenizers/` on first use (~9 MB, reused from an existing local bge-m3 model cache when available)), rerank documents per request `GODOT_RAG_RERANK_BATCH_SIZE` (`64`), timeout `GODOT_RAG_API_TIMEOUT` (`60` seconds), max retries `GODOT_RAG_API_MAX_RETRIES` (`8`), initial retry delay `GODOT_RAG_API_RETRY_BASE_DELAY` (`2.0` seconds), retry delay cap `GODOT_RAG_API_RETRY_MAX_DELAY` (`60.0` seconds), and writer-side upsert batching `GODOT_RAG_UPSERT_BATCH` (`512`). 429 (rate limit) / 503 (overloaded) / 504 (timeout) are retried automatically with exponential backoff; 400/401/403/404 are configuration errors and abort immediately.

> Run every command from the repository root using `python -m scripts.<module>`. The scripts use package-relative imports, so running them as `python scripts/xxx.py` will fail.

## Building the Index

```bash
python -m scripts.run_pipeline
```

The script first verifies that every pipeline script exists and is non-empty, then runs these stages in order:

1. `scripts.rst2md`: clones `godotengine/godot-docs` (shallow) into `godot-docs-master/` if it is missing, then converts RST files under `DOCS_DIRS` to `.md` files alongside them.
2. `scripts.chunk_documents`: splits the Markdown structurally and writes `data/chunks.jsonl`.
3. `scripts.build_all_index`: builds the Qdrant, BM25, and symbol indexes in three parallel subprocesses and runs index integrity validation (`scripts.libs.index_builder.validate`) once all succeed.

If any stage fails, the remaining stages are skipped.

Notes:

- The documentation repository is cloned only on the first run and is **not** updated automatically. To update the docs, run `git pull` inside `godot-docs-master/`.
- The directories to convert and chunk are configured by `DOCS_DIRS` in `scripts/rst2md.py` and `scripts/chunk_documents.py`. The defaults are `about`, `classes`, `community`, `engine_details`, `getting_started`, and `tutorials`; make sure they exist under `godot-docs-master/`.
- Chunking uses the `chunkana` `structural` strategy with `max_chunk_size` set to 8192. `data/chunks.jsonl` is rebuilt on every run.
- If the reconstructed content differs from the source, a warning is logged per document, distinguishing "content loss" from "blank-line-only differences" (blank-line-only documents are counted in the run summary instead of warned individually). Nothing is written to disk.

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

`scripts.build_all_index` builds the three modules under `scripts/index_builder/` in three parallel subprocesses (they are independent of each other; validation runs only after all succeed):

- **Qdrant**: generates 1024-dimensional normalized vectors with `bge-m3` through the SiliconFlow API, uses cosine distance, and persists to `data/qdrant_data` under the collection name `godot_docs`. Chunk IDs are hashed with UUIDv5 to form point IDs. Rebuilding physically empties the entire `data/qdrant_data` storage directory before recreating it (`delete_collection` in qdrant-client local mode does not actually purge old points in some versions), and after writing, the total point count is reconciled against the number of points written in this run; make sure it does not contain data you need to keep.
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

Generation uses an OpenAI-compatible API configured through environment variables: `OPENAI_BASE_URL` (**required**, service address; cloud DeepSeek is `https://api.deepseek.com`, local Ollama is `http://localhost:11434/v1`), `OPENAI_API_KEY` (service key; when missing a placeholder is used instead — local deployments don't check it, cloud endpoints return 401), and `OPENAI_MODEL` (model name, default `deepseek-chat`). If `OPENAI_BASE_URL` is not set, generation fails immediately with a clear error. The prompt requires answers to rely only on the retrieved chunks, cite `[来源: section path]`, and state clearly when the context is insufficient.

## Evaluation

```bash
python -m scripts.evaluate
```

The evaluation first validates that every line of `evaluate/questions.jsonl` is valid JSON with the required fields (`id`, `query`, `type`, `expected_source`) and that `expected_source` / `expected_section` actually exist, stopping if validation fails. It then reports:

- **Retrieval**: per language, `Hit@6` (whether the expected source appears in the results) and `Symbol Hit@1` (whether the first result for API questions contains the expected symbol).
- **Generation**: per language, source citation rate, API name retention rate, and refusal accuracy on unanswerable questions.

Failures are written to `evaluate/misses.json`. Each line of the question set is a JSON object with fields including `id`, `lang`, `query`, `type` (such as `api`, `mixed`, or `unanswerable`), `expected_symbol`, `expected_source`, and `expected_section`.

## Notes

- This is a local experimental project with no web service or frontend; retrieval and QA are invoked through Python modules.
- Scripts locate data through `PROJECT_DIR` in `scripts/libs/utils.py`, so keep the relative paths and directory layout consistent.
- Documentation content is copyright its Godot Engine documentation contributors; follow their license when reusing it.
- This project is licensed under the [MIT License](LICENSE).
