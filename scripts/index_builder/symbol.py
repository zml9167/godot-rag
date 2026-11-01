import re
import json
from loguru import logger
from pathlib import Path

from ..paths import PROJECT_DIR

CURRENT_PATH = Path(__file__)
SYMBOL_INDEX_DIR = PROJECT_DIR / "data/symbol_index"
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"

# 过滤掉太通用的词，避免符号索引被噪音淹没
STOPWORDS = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "on",
             "for", "with", "this", "that", "it", "as", "be", "or", "if"}



def extract_symbols(chunk):
    symbols = set()

    # 1. header_path 最后一级
    if chunk["header_path"]:
        last = chunk["header_path"][-1]
        # 去掉括号、参数等，只保留标识符
        for m in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', last):
            if m.lower() not in STOPWORDS:
                symbols.add(m)

    # 2. content 中形如 `Class.method` 或独立 API 名的 token
    for m in re.findall(r'`([A-Za-z_][A-Za-z0-9_.]*)`', chunk["content"]):
        symbols.add(m)
    for m in re.findall(r'\b([A-Z][a-z]+[A-Za-z0-9]*)\b', chunk["content"]):
        symbols.add(m)

    return {s for s in symbols if s.lower() not in STOPWORDS and len(s) >= 3}


def build(chunks_path: str):
    logger.add(LOG_PATH)
    SYMBOL_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    symbol_to_chunks = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            for sym in extract_symbols(c):
                symbol_to_chunks.setdefault(sym.lower(), []).append(c["chunk_id"])

    with open(SYMBOL_INDEX_DIR / "symbol_index.json", "w", encoding="utf-8") as f:
        json.dump(symbol_to_chunks, f, ensure_ascii=False)
    logger.info(f"symbol index done: {len(symbol_to_chunks)} symbols")


if __name__ == "__main__":
    chunks = PROJECT_DIR / "data/chunks.jsonl"
    build(chunks)