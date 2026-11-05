from loguru import logger

from .libs.utils import PROJECT_DIR
from .libs.index_builder import bm25
from .libs.index_builder import symbol
from .libs.index_builder import qdrant
from .libs.index_builder import validate

LOG_PATH = PROJECT_DIR / "logs/build_index.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger.add(LOG_PATH)

chunks = PROJECT_DIR / "data/chunks.jsonl"
qdrant.build(chunks)
bm25.build(chunks)
symbol.build(chunks)
validate.validate_indexes(chunks)
