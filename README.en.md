# Godot Documentation RAG

[中文](README.md) | [English](README.en.md)

An experimental project that builds a local retrieval index from the official Godot documentation. The pipeline converts RST documents to MyST Markdown, splits them into structural chunks, and builds BM25, API symbol, and Qdrant vector indexes. Retrieval combines symbol lookup, dense vectors, and BM25 to provide context for an LLM, with a companion evaluation script for both retrieval and generation.

## Project Structure

```text
.
├── godot-docs-master/            # Documentation cloned from godotengine/godot-docs
├── docs/                         # Process notes
│   ├── GPU_TUNING.md             # GPU memory tuning notes from the local-GPU era (historical)
│   └── RETRIEVAL_OPTIMIZATION.md # Retrieval optimization notes
├── evaluate/
│   ├── questions.jsonl           # Evaluation question set
│   └── misses.json               # Failure cases written by the evaluation (generated at runtime)
├── scripts/
│   ├── run_pipeline.py           # Run conversion, chunking, and indexing
│   ├── rst2md.py                 # RST -> MyST Markdown
│   ├── chunk_documents.py        # Markdown -> data/chunks.jsonl
│   ├── build_all_index.py        # Build vector, BM25, and symbol indexes in parallel, then validate
│   ├── retrieve.py               # Symbol + dual-vector + BM25 hybrid retrieval with reranking
│   ├── generate.py               # LLM answer generation
│   ├── ask.py                    # Retrieval + generation entry point
│   ├── evaluate.py               # Retrieval and generation evaluation
│   └── libs/
│       ├── utils.py              # Path constants and API client singletons
│       ├── siliconflow.py        # SiliconFlow embeddings/rerank API client
│       ├── tokenizer.py          # BM25 tokenizer
│       ├── pattern_utils.py      # API symbol extraction patterns
│       └── index_builder/
│           ├── qdrant.py         # Qdrant vector index
│           ├── bm25.py           # BM25 index
│           ├── symbol.py         # API symbol index
│           └── validate.py       # Index integrity validation
├── data/
│   ├── chunks.jsonl
│   ├── bm25_index/
│   ├── symbol_index/
│   └── qdrant_data/
├── models_cache/                 # tokenizer.json and other local caches (downloaded on first use)
└── logs/                         # Stage logs
```

`godot-docs-master/`, `data/`, `logs/`, and `models_cache/` contain local inputs or generated data and are not tracked by Git.

## Environment Setup

Create a virtual environment at the repository root and install the dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` does not include PyTorch: embedding and reranking only support the SiliconFlow API, so this project no longer depends on a GPU, PyTorch, or local model weights.

**Embedding and reranking only support the SiliconFlow API (no other providers or local models); export the key first:**

```bash
export SILICONFLOW_API_KEY="your-api-key"
```

If the key is missing, index building or retrieval fails when the client is initialized. Related environment variables (all optional, defaults in parentheses): embedding model name `GODOT_RAG_EMBEDDING_MODEL` (`BAAI/bge-m3`), reranker model name `GODOT_RAG_RERANKER_MODEL` (`BAAI/bge-reranker-v2-m3`), API base URL `SILICONFLOW_API_BASE` (`https://api.siliconflow.cn/v1`), embedding items per request `GODOT_RAG_API_BATCH_SIZE` (`32`, hard API limit), per-request token budget `GODOT_RAG_API_BATCH_TOKENS` (`24576`), per-text token limit `GODOT_RAG_EMBEDDING_MAX_TOKENS` (`8176`; bge-m3 accepts up to 8192 tokens per item — oversized chunks are truncated precisely by token using the bge-m3 tokenizer before encoding, with a logged warning; `tokenizer.json` is downloaded automatically from ModelScope/HF into `models_cache/tokenizers/` on first use (~9 MB, reused from an existing local bge-m3 model cache when available)), tokenizer file path `GODOT_RAG_TOKENIZER_FILE` (`models_cache/tokenizers/bge-m3/tokenizer.json`), rerank documents per request `GODOT_RAG_RERANK_BATCH_SIZE` (`64`), timeout `GODOT_RAG_API_TIMEOUT` (`60` seconds), max retries `GODOT_RAG_API_MAX_RETRIES` (`8`), initial retry delay `GODOT_RAG_API_RETRY_BASE_DELAY` (`2.0` seconds), retry delay cap `GODOT_RAG_API_RETRY_MAX_DELAY` (`60.0` seconds), and writer-side upsert batching `GODOT_RAG_UPSERT_BATCH` (`512`). 429 (rate limit) / 503 (overloaded) / 504 (timeout) are retried automatically with exponential backoff; 400/401/403/404 are configuration errors and abort immediately.

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
- The directories to convert and chunk are configured by `DOCS_DIRS` in `scripts/libs/utils.py` (shared by `rst2md.py` and `chunk_documents.py`). The defaults are `about`, `classes`, `community`, `engine_details`, `getting_started`, and `tutorials`; make sure they exist under `godot-docs-master/`.
- Chunking uses the `chunkana` `structural` strategy with adaptive sizing enabled (`AdaptiveSizeConfig(base_size=2048)`). `data/chunks.jsonl` is rebuilt on every run, and blank chunks are skipped instead of written.
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

- **Qdrant**: generates 1024-dimensional normalized vectors with `bge-m3` through the SiliconFlow API, uses cosine distance, and persists to `data/qdrant_data` under the collection name `godot_docs`. Each point carries two named vectors: `title` (the `header_path` joined with ` > `) and `content` (the chunk body); chunks without a valid `header_path` only get the `content` vector. Chunk IDs are hashed with UUIDv5 to form point IDs. Rebuilding physically empties the entire `data/qdrant_data` storage directory before recreating it (`delete_collection` in qdrant-client local mode does not actually purge old points in some versions), and after writing, the total point count is reconciled against the number of points written in this run; make sure it does not contain data you need to keep.
- **BM25**: uses `rank_bm25`'s `BM25Okapi`. Tokenization lives in `scripts/tokenizer.py` and keeps only lowercase English identifiers and numbers, matching the retrieval side. Outputs `bm25.pkl`, `chunk_ids.json`, and `meta.json` under `data/bm25_index/`.
- **Symbol index**: extracts API symbols from `classes` chunks only — class names come from `header_path[0]`, while method/property/constant/signal declarations are extracted from the chunk body (patterns in `scripts/libs/pattern_utils.py`; capture groups include digits so names like `CharacterBody2D` are covered). Non-class chunks such as tutorials produce no symbols; their retrieval coverage is left to vectors and BM25. Outputs `symbol_index.json` (exact symbols) and `symbol_lower_index.json` (lowercased symbols) under `data/symbol_index/`.

Each stage writes its log under `logs/`.

## Retrieval and Question Answering

The retrieval flow in `scripts/retrieve.py`:

1. **Symbol retrieval**: detect API symbols in the query with regular expressions (exact symbol index first, then the lowercased index with stopwords filtered out), rank the hit chunks by priority (whether the section heading / source path / chunk body matches the symbol, with `classes` ranked above `tutorials`), score them with the reranker, and keep the top `symbol_top_k` (default 3) chunks.
2. **Hybrid recall**: fetch the top 30 results from the `title` vectors, the top 30 from the `content` vectors, and the top 30 from BM25, then fuse the three rankings with RRF (`k=60`).
3. **Reranking**: score the fused candidates with the reranker, re-sort them, and keep the top `fuse_top_k` (default 6) chunks.
4. **Merge**: symbol hits go first; the remaining results are de-duplicated by `chunk_id` and appended, for up to 3 + 6 = 9 chunks in total. Each result carries a `retrieve_method` marker (`symbol` / `hybrid`).

`scripts/ask.py` refuses to answer when retrieval returns nothing. Otherwise it builds a context with source labels from the chunks, calls the LLM, and returns a de-duplicated source list.

Command-line example:

```bash
python -m scripts.ask "How should I pass the delta parameter to move_and_slide?"
```

Generation uses an OpenAI-compatible API (only generation supports a custom service address; embedding and reranking only support SiliconFlow) configured through environment variables: `OPENAI_BASE_URL` (**required**, service address; cloud DeepSeek is `https://api.deepseek.com`, local Ollama is `http://localhost:11434/v1`), `OPENAI_API_KEY` (service key; when missing a placeholder is used instead — local deployments don't check it, cloud endpoints return 401), and `OPENAI_MODEL` (model name, **required**). Generation runs with a low temperature (`temperature=0.1`) and `max_tokens=1500` to reduce free-form drift. The prompt requires answers to rely only on the retrieved chunks, cite `[来源: section path]`, and state clearly when the context is insufficient.

## Evaluation

```bash
python -m scripts.evaluate
```

The evaluation first validates that every line of `evaluate/questions.jsonl` is valid JSON with the required fields (`id`, `query`, `type`, `expected_source`) and that `expected_source` / `expected_section` actually exist, stopping if validation fails. It then reports:

- **Retrieval**: per language, `Hit@9` (`symbol_top_k + fuse_top_k`; whether the expected source appears in the results) and `Symbol Hit@1` (whether the first result for API questions comes from symbol retrieval).
- **Generation**: per language, source citation rate, API name retention rate, and refusal accuracy on unanswerable questions.

Failures are written to `evaluate/misses.json` under the `retrieval_misses` and `generation_failures` keys. Each line of the question set is a JSON object with fields including `id`, `lang`, `query`, `type` (such as `api`, `mixed`, or `unanswerable`), `expected_symbol`, `expected_source`, and `expected_section`.

## Notes

- This is a local experimental project with no web service or frontend; retrieval and QA are invoked through Python modules.
- Scripts locate data through `PROJECT_DIR` in `scripts/libs/utils.py`, so keep the relative paths and directory layout consistent.
- See `docs/RETRIEVAL_OPTIMIZATION.md` for retrieval optimization decisions and evaluation data; `docs/GPU_TUNING.md` is a historical record of GPU memory tuning from the local-GPU era (embedding and reranking now use the API, and the environment variables described there no longer exist).
- Documentation content is copyright its Godot Engine documentation contributors; follow their license when reusing it.
- This project is licensed under the [MIT License](LICENSE).
