from .paths import PROJECT_DIR
from scripts.index_builder import bm25
from scripts.index_builder import symbol
from scripts.index_builder import qdrant

LOG_PATH = PROJECT_DIR / "logs/build_index.log"

chunks = str(PROJECT_DIR / "data/chunks.jsonl")
qdrant.build(chunks)
bm25.build(chunks)
symbol.build(chunks)