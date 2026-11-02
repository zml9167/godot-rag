from .libs.utils import PROJECT_DIR
from .libs.index_builder import bm25
from .libs.index_builder import symbol
from .libs.index_builder import qdrant

LOG_PATH = PROJECT_DIR / "logs/build_index.log"

chunks = PROJECT_DIR / "data/chunks.jsonl"
qdrant.build(chunks)
bm25.build(chunks)
symbol.build(chunks)