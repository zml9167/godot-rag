import re
import json
from loguru import logger
from pathlib import Path

from ..utils import PROJECT_DIR, SYMBOL_INDEX_DIR, STOPWORDS, DOCS_DIRS
from ..pattern_utils import ref_with_target_re, ref_target_bold_ident_re, void_mark_bold_ident_re, file_line_range_re

CURRENT_PATH = Path(__file__)
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"


def extract_symbols(chunk: dict) -> set[str]:
    symbols = set()

    # classes
    if chunk["chunk_type"] == "clsses":
        source_split = chunk["source"].split("_")
        if len(source_split) > 1:
            symbols.add(j)

    # content 中的标题
    for m in re.findall(ref_with_target_re(), chunk["content"]):
        symbols.add(m)
    for m in re.findall(ref_target_bold_ident_re(), chunk["content"]):
        symbols.add(m)
    for m in re.findall(void_mark_bold_ident_re(), chunk["content"]):
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