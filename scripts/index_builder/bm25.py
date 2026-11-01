import pickle
import json
from loguru import logger
from pathlib import Path
from rank_bm25 import BM25Okapi

from ..paths import PROJECT_DIR
from ..tokenizer import tokenize

CURRENT_PATH = Path(__file__)
BM25_INDEX_DIR = PROJECT_DIR / "data/bm25_index"
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"


def build(chunks_path):
    logger.add(LOG_PATH)
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunk_ids, tokenized_corpus = [], []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_ids.append(c["chunk_id"])
            tokenized_corpus.append(tokenize(c["content"]))

    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_INDEX_DIR / "chunk_ids.json", "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)
    with open(BM25_INDEX_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"tokenizer": "custom_v1", "count": len(chunk_ids)}, f)
    logger.info("bm25 index done")


if __name__ == "__main__":
    chunks_path = PROJECT_DIR / "data/chunks.jsonl"
    build(chunks_path)
