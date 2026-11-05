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
    source_match = re.search(file_line_range_re(), chunk["source"])
    for i in source_match.groups() if source_match else []:
        for j in i.split('_'):
            if j and j.lower() not in DOCS_DIRS:
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
            # extract_symbols 区分大小写；转小写后再去重，避免同一个 chunk 重复挂到同一符号下
            for sym in {s.lower() for s in extract_symbols(c)}:
                symbol_to_chunks.setdefault(sym, []).append(c["chunk_id"])

    with open(SYMBOL_INDEX_DIR / "symbol_index.json", "w", encoding="utf-8") as f:
        json.dump(symbol_to_chunks, f, ensure_ascii=False)
    logger.info(f"symbol index done: {len(symbol_to_chunks)} symbols")


if __name__ == "__main__":
    chunks = PROJECT_DIR / "data/chunks.jsonl"
    build(chunks)