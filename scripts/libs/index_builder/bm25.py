import pickle
import json
import time
from loguru import logger
from pathlib import Path
from rank_bm25 import BM25Okapi

from ..utils import PROJECT_DIR, format_duration
from ..tokenizer import tokenize

CURRENT_PATH = Path(__file__)
BM25_INDEX_DIR = PROJECT_DIR / "data/bm25_index"
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"


def build(chunks_path):
    logger.add(LOG_PATH, encoding="utf-8")
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    chunk_ids, tokenized_corpus = [], []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_ids.append(c["chunk_id"])
            tokenized_corpus.append(tokenize(c["content"]))

    # 与 qdrant.build 的空文件守卫对齐：把 rank_bm25 深处无上下文的
    # ZeroDivisionError 提前变成带原因和路径的明确报错
    if not chunk_ids:
        raise ValueError(f"chunks 文件为空：{chunks_path}")
    if not any(tokenized_corpus):
        raise ValueError(
            "chunks 分词后全为空（tokenize 只保留英文标识符和数字，中文会被丢弃），"
            f"无法建 BM25 索引：{chunks_path}"
        )

    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_INDEX_DIR / "chunk_ids.json", "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)
    with open(BM25_INDEX_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"tokenizer": "custom_v1", "count": len(chunk_ids)}, f)
    elapsed = time.perf_counter() - started
    logger.info(f"bm25 index done，总耗时 {format_duration(elapsed)}")
    return elapsed


if __name__ == "__main__":
    chunks_path = PROJECT_DIR / "data/chunks.jsonl"
    build(chunks_path)
