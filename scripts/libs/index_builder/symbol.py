import re
import json
from loguru import logger
from pathlib import Path

from ..utils import PROJECT_DIR, SYMBOL_INDEX_DIR, STOPWORDS, DOCS_DIRS
from ..pattern_utils import get_symbol_from_clases_content

CURRENT_PATH = Path(__file__)
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"


def extract_symbols_from_header_path(chunk: dict, index: int) -> set[str]:
    """从 chunk 的 header_path 中提取符号"""
    symbols = set()
    if not isinstance(chunk.get("header_path"), list):
        return symbols
    if len(chunk["header_path"]) > 0 and index < len(chunk["header_path"]):
        symbols.add(chunk["header_path"][index])
    return symbols


def extract_symbols(chunk: dict) -> set[str]:
    """从 chunk 的 source 和 content 中提取符号"""
    symbols = set()
    if chunk["chunk_type"] == "classes":
        symbols.update(extract_symbols_from_header_path(chunk, 0))
        symbols.update(get_symbol_from_clases_content(chunk))
    else:
        symbols.update(extract_symbols_from_header_path(chunk, -1))
    return {s for s in symbols if s.lower() not in STOPWORDS and len(s) >= 3}


def build(chunks_path: str):
    logger.add(LOG_PATH)
    SYMBOL_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    symbol_to_chunks = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            # 保留完整引用；查询时在 symbol_lookup 中按 chunk_id 去重
            for sym in extract_symbols(c):
                symbol_to_chunks.setdefault(sym, []).append(c["chunk_id"])

    with open(SYMBOL_INDEX_DIR / "symbol_index.json", "w", encoding="utf-8") as f:
        json.dump(symbol_to_chunks, f, ensure_ascii=False)
    logger.info(f"symbol index done: {len(symbol_to_chunks)} symbols")


if __name__ == "__main__":
    chunks = PROJECT_DIR / "data/chunks.jsonl"
    build(chunks)