from .libs.utils import PROJECT_DIR
from .libs.index_builder import bm25
from .libs.index_builder import symbol
from .libs.index_builder import qdrant
from .libs.index_builder import validate

chunks = PROJECT_DIR / "data/chunks.jsonl"
qdrant.build(chunks)
bm25.build(chunks)
symbol.build(chunks)
validate.validate_indexes(chunks)
