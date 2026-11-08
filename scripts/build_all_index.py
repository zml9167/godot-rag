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
    jobs = [
        ("qdrant", Process(target=qdrant.build, args=(chunks,))),
        ("bm25", Process(target=bm25.build, args=(chunks,))),
        ("symbol", Process(target=symbol.build, args=(chunks,))),
    ]
    for _, process in jobs:
        process.start()
    for _, process in jobs:
        process.join()

    # 任一构建进程失败就直接报错退出，跳过校验：
    # 避免校验阶段的"索引缺失/数量不一致"掩盖真实的构建错误
    failed = [(name, process.exitcode) for name, process in jobs if process.exitcode != 0]
    if failed:
        detail = ", ".join(f"{name}（exitcode={code}）" for name, code in failed)
        raise SystemExit(
            f"索引构建失败：{detail}，详见 logs/ 下对应阶段的日志，跳过索引校验"
        )

    validate.validate_indexes(chunks)
