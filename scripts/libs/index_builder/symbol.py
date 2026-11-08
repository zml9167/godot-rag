import json
import time
from loguru import logger
from pathlib import Path

from ..utils import PROJECT_DIR, SYMBOL_INDEX_DIR, STOPWORDS, CHUNKS_PATH, format_duration
from ..pattern_utils import get_symbol_from_clases_content

CURRENT_PATH = Path(__file__)
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"


def extract_symbols_from_header_path(header_path, index: int = 0) -> set[str]:
    """从 chunk 的 header_path 中提取符号"""
    symbols = set()
    if not isinstance(header_path, list):
        return symbols
    if len(header_path) > 0 and index < len(header_path):
        symbols.add(header_path[index])
    return symbols


def extract_symbols(chunk: dict) -> set[str]:
    """从 chunk 的 source 和 content 中提取符号"""
    symbols = set()
    if chunk["chunk_type"] == "classes":
        symbols.update(extract_symbols_from_header_path(chunk["header_path"], 0))
        symbols.update(get_symbol_from_clases_content(chunk["content"]))
    else:
        symbols.update(extract_symbols_from_header_path(chunk["header_path"], -1))
    return {s for s in symbols if s.lower() not in STOPWORDS and len(s) >= 3}


def build(chunks_path: Path = CHUNKS_PATH):
    logger.add(LOG_PATH, encoding="utf-8")
    SYMBOL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    symbol_to_chunks = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            # 保留完整引用；查询时在 symbol_lookup 中按 chunk_id 去重。
            # key 统一小写：检索端（detect_symbols/symbol_lookup）一律用
            # 小写 key 查询，混合大小写的 key 是永远查不到的死条目。
            for sym in extract_symbols(c):
                symbol_to_chunks.setdefault(sym.lower(), []).append(c["chunk_id"])

    with open(SYMBOL_INDEX_DIR / "symbol_index.json", "w", encoding="utf-8") as f:
        json.dump(symbol_to_chunks, f, ensure_ascii=False)
    elapsed = time.perf_counter() - started
    logger.info(
        f"symbol index done: {len(symbol_to_chunks)} symbols，总耗时 {format_duration(elapsed)}"
    )
    return elapsed


if __name__ == "__main__":
    chunks = PROJECT_DIR / "data/chunks.jsonl"
    build(chunks)