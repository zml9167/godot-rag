import multiprocessing
from multiprocessing import Process
from .libs.utils import PROJECT_DIR
from .libs.index_builder import bm25
from .libs.index_builder import symbol
from .libs.index_builder import qdrant
from .libs.index_builder import validate


if __name__ == "__main__":
    # CUDA 不兼容 fork，统一用 spawn 启动子进程
    multiprocessing.set_start_method("spawn", force=True)
    chunks = PROJECT_DIR / "data/chunks.jsonl"
    qdrant_process = Process(target=qdrant.build, args=(chunks,))
    qdrant_process.start()
    bm25_process = Process(target=bm25.build, args=(chunks,))
    bm25_process.start()
    symbol_process = Process(target=symbol.build, args=(chunks,))
    symbol_process.start()
    qdrant_process.join()
    bm25_process.join()
    symbol_process.join()
    validate.validate_indexes(chunks)
