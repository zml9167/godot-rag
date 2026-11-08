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
│       ├── utils.py           # Device detection, model loading, path constants
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
├── models_cache/              # Embedding model cache
└── logs/                      # Stage logs
```

`godot-docs-master/`, `data/`, `models_cache/`, and `logs/` contain local inputs or generated data and are not tracked by Git.

## Environment Setup

Create a virtual environment at the repository root and install the dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` does not include PyTorch, so install a build compatible with your local CUDA environment separately. Index building uses the GPU (`cuda`) by default and prints the active device after loading the model, before encoding; if the GPU is unavailable it fails instead of silently falling back to CPU. A short GPU smoke test runs in a subprocess at startup, and if the ROCm PyTorch build lacks kernels for the current architecture (for example `gfx1031`), a compatible `HSA_OVERRIDE_GFX_VERSION` is set automatically. CPU mode requires an explicit `GODOT_RAG_ALLOW_CPU=1`. The long-sequence truncation limit can be adjusted with `GODOT_RAG_MAX_SEQ_LENGTH` (default 4096). The embedding batch size is determined by automatic calibration at build startup: two budget gates — the attention budget `GODOT_RAG_ATTN_BUDGET` and the per-batch token budget `GODOT_RAG_TOKEN_BUDGET` — default to `auto`: starting from conservative values, real probe encodings double the budgets and binary search converges toward the VRAM safety line. `GODOT_RAG_VRAM_MARGIN` (default 0.20, VRAM kept aside), `GODOT_RAG_EMPTY_CACHE_EVERY` (default 256 batches, cache release interval), `GODOT_RAG_REFINE_ROUNDS` (default 7, binary refinement rounds), and `GODOT_RAG_UPSERT_BATCH` (default 512, writer-side upsert batching) can all be overridden.

The embedding model is `BAAI/bge-m3`. On first use, the download source depends on the current IP's country: mainland China (`CN`) uses ModelScope, other regions use Hugging Face, and files are cached under `models_cache/`.

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

- **Qdrant**: generates 1024-dimensional normalized vectors with `bge-m3`, uses cosine distance, and persists to `data/qdrant_data` under the collection name `godot_docs`. Chunk IDs are hashed with UUIDv5 to form point IDs. Rebuilding physically empties the entire `data/qdrant_data` storage directory before recreating it (`delete_collection` in qdrant-client local mode does not actually purge old points in some versions), and after writing, the total point count is reconciled against the number of points written in this run; make sure it does not contain data you need to keep.
- **BM25**: uses `rank_bm25`'s `BM25Okapi`. Tokenization lives in `scripts/tokenizer.py` and keeps only lowercase English identifiers and numbers, matching the retrieval side. Outputs `bm25.pkl`, `chunk_ids.json`, and `meta.json` under `data/bm25_index/`.
- **Symbol index**: extracts API symbols from the last heading level, backticked identifiers, and CamelCase names, then builds a lowercase "symbol -> chunk list" mapping in `data/symbol_index/symbol_index.json`.

Each stage writes its log under `logs/`. To debug batch calibration on its own, run `python -m scripts.libs.index_builder.qdrant --validate-calibration`: stage one checks the batching invariants (no GPU needed, finishes in seconds), stage two calibrates on the real GPU and trial-encodes one batch to report peak VRAM; without CUDA the VRAM probe is skipped automatically.

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
