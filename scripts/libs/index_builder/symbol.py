import re
import json
import time
from loguru import logger
from pathlib import Path

from ..utils import PROJECT_DIR, SYMBOL_INDEX_DIR, SYMBOL_INDEX_PATH, SYMBOL_LOWER_INDEX_PATH, STOPWORDS, CHUNKS_PATH, format_duration
from ..pattern_utils import get_symbol_from_clases_content

CURRENT_PATH = Path(__file__)
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"


def extract_symbols(chunk: dict) -> set[str]:
    """只从 classes chunk 提取符号：header_path[0] = 类名，content = 成员声明。

    - classes chunk：header_path[0]（类文档页首 header 即类名，如 CharacterBody2D；
      排除 __preamble__ 等内部标记）
    - classes chunk 的 content：get_symbol_from_clases_content
      （方法/属性/常量/信号声明，不含类名链接文本——类名已由 header_path 覆盖）
    - 教程等非类文档 chunk 不产符号，检索覆盖交给 BM25/向量
    - 过滤：STOPWORDS、长度 <3
    """
    symbols = set()
    if chunk["chunk_type"] == "classes":
        header_path = chunk.get("header_path")
        if isinstance(header_path, list) and header_path:
            name = str(header_path[0])
            if not name.startswith("__"):
                symbols.add(name)
        symbols.update(get_symbol_from_clases_content(chunk["content"]))
    return {s for s in symbols if len(s) >= 3 and s.lower() not in STOPWORDS}


def build(chunks_path: Path = CHUNKS_PATH):
    logger.add(LOG_PATH, encoding="utf-8")
    SYMBOL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    symbol_to_chunks = {}
    symbol_lower_to_chunks = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            # 保留完整引用；查询时在 symbol_lookup 中按 chunk_id 去重。
            for sym in extract_symbols(c):
                symbol_to_chunks.setdefault(sym, []).append(c["chunk_id"])

    for key, ids in symbol_to_chunks.items():
        if re.search(r'[A-Z]+', key):
            symbol_lower_to_chunks.setdefault(key.lower(), []).extend(ids)

    with open(SYMBOL_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(symbol_to_chunks, f, ensure_ascii=False)
    with open(SYMBOL_LOWER_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(symbol_lower_to_chunks, f, ensure_ascii=False)
    elapsed = time.perf_counter() - started
    logger.info(
        f"symbol index done: {len(symbol_to_chunks)} symbols，总耗时 {format_duration(elapsed)}"
    )
    return elapsed


if __name__ == "__main__":
    chunks = PROJECT_DIR / "data/chunks.jsonl"
    build(chunks)