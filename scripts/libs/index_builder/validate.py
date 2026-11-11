import json
import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient

from ..utils import PROJECT_DIR, QURANT_DATA_DIR, SYMBOL_INDEX_PATH, SYMBOL_LOWER_INDEX_PATH
from .bm25 import BM25_INDEX_DIR
from .qdrant import COLLECTION_NAME, VECTOR_DIM, chunk_id_to_point_id


class IndexValidationError(RuntimeError):
    pass


class ValidationErrors:
    """收集错误，日志最多打印前 limit 条，避免刷屏。"""

    def __init__(self, limit: int = 100):
        self.limit = limit
        self.count = 0
        self.messages = []

    def add(self, message: str) -> None:
        self.count += 1
        if len(self.messages) < self.limit:
            self.messages.append(message)

    def raise_if_any(self) -> None:
        if not self.count:
            return
        for message in self.messages:
            logger.error(message)
        if self.count > len(self.messages):
            logger.error(f"... 其余 {self.count - len(self.messages)} 条错误已省略")
        raise IndexValidationError(f"索引校验失败：{self.count} 个问题")


def _check_vector(vector, name: str, chunk_id: str, errors: ValidationErrors) -> None:
    """校验单个命名向量：维度、有限性、归一化。"""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.shape != (VECTOR_DIM,):
        errors.add(f"Qdrant {name} 向量维度异常：{chunk_id} shape={arr.shape}")
    elif not np.isfinite(arr).all():
        errors.add(f"Qdrant {name} 向量含 NaN/Inf：{chunk_id}")
    else:
        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > 1e-3:
            errors.add(f"Qdrant {name} 向量未归一化：{chunk_id} norm={norm:.6f}")


def _load_chunks(chunks_path: Path, errors: ValidationErrors):
    chunks = {}
    order = []
    with open(chunks_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                errors.add(f"chunks.jsonl:{lineno}: 空行")
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.add(f"chunks.jsonl:{lineno}: JSON 解析失败：{exc}")
                continue

            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                errors.add(f"chunks.jsonl:{lineno}: chunk_id 缺失或不是字符串")
                continue
            if chunk_id in chunks:
                errors.add(f"chunks.jsonl:{lineno}: 重复 chunk_id：{chunk_id}")
                continue

            for field, field_type in (
                ("source", str),
                ("header_path", list),
                ("chunk_type", str),
                ("content", str),
            ):
                if not isinstance(chunk.get(field), field_type):
                    errors.add(
                        f"chunks.jsonl:{lineno}: 字段 {field} 缺失或类型错误"
                        f"（期望 {field_type.__name__}）"
                    )

            if isinstance(chunk.get("content"), str) and not chunk["content"].strip():
                errors.add(f"chunks.jsonl:{lineno}: content 为空白")

            chunks[chunk_id] = chunk
            order.append(chunk_id)

    if not order:
        errors.add("chunks.jsonl: 没有任何有效 chunk")

    return chunks, order


def _validate_qdrant(chunks, errors: ValidationErrors) -> int:
    client = QdrantClient(path=str(QURANT_DATA_DIR))
    try:
        if not client.collection_exists(COLLECTION_NAME):
            errors.add(f"Qdrant collection 不存在：{COLLECTION_NAME}")
            return 0

        info = client.get_collection(COLLECTION_NAME)
        vector_params = info.config.params.vectors
        if isinstance(vector_params, dict):
            # named vectors：title/content 配置都必须存在且维度一致
            for name in ("content", "title"):
                params = vector_params.get(name)
                if params is None:
                    errors.add(f"Qdrant collection 缺少 {name} 向量配置")
                elif params.size != VECTOR_DIM:
                    errors.add(
                        f"Qdrant {name} 向量维度 {params.size} != 期望 {VECTOR_DIM}"
                    )
        else:
            # 旧的单未命名向量索引：检索侧按 named vectors（using=...）查询会失败
            errors.add(
                f"Qdrant collection 是单一未命名向量格式（维度 {vector_params.size}），"
                "缺少 title/content 配置，请重建索引"
            )

        count = client.count(COLLECTION_NAME, exact=True).count
        if count != len(chunks):
            errors.add(f"Qdrant point 数 {count} != chunks 数 {len(chunks)}")

        seen = set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    errors.add(f"Qdrant point {point.id}: payload 缺少 chunk_id")
                    continue
                if chunk_id in seen:
                    errors.add(f"Qdrant point {point.id}: 重复 chunk_id {chunk_id}")
                seen.add(chunk_id)

                chunk = chunks.get(chunk_id)
                if chunk is None:
                    errors.add(f"Qdrant point {point.id}: chunks 中不存在 {chunk_id}")
                    continue

                if str(point.id) != chunk_id_to_point_id(chunk_id):
                    errors.add(f"Qdrant point id 与 chunk_id 不匹配：{chunk_id}")

                # payload 合同：只存 chunk_id（正文等字段检索时经 CHUNK_MAP
                # 从 chunks.jsonl 取，Qdrant 不做第二份存储），因此这里
                # 不比对 source/header_path/chunk_type/content。

                # named vectors 合同：content 必须存在；title 可缺省
                # （空 header_path 的 chunk 建索引时不生成 title 向量），
                # 存在则一并校验维度/有限性/归一化
                vector = point.vector
                if not isinstance(vector, dict):
                    errors.add(f"Qdrant 向量缺失：{chunk_id}")
                    continue
                if "content" not in vector:
                    errors.add(f"Qdrant 缺少 content 向量：{chunk_id}")
                else:
                    _check_vector(vector["content"], "content", chunk_id, errors)
                if "title" in vector:
                    _check_vector(vector["title"], "title", chunk_id, errors)

            if offset is None:
                break

        missing = set(chunks) - seen
        if missing:
            examples = ", ".join(sorted(missing)[:5])
            errors.add(f"Qdrant 缺少 {len(missing)} 个 chunk，例如：{examples}")
        extra = seen - set(chunks)
        if extra:
            examples = ", ".join(sorted(extra)[:5])
            errors.add(f"Qdrant 多出 {len(extra)} 个 chunk，例如：{examples}")

        return len(seen)
    finally:
        client.close()


def _validate_bm25(order, errors: ValidationErrors) -> int:
    pkl_path = BM25_INDEX_DIR / "bm25.pkl"
    ids_path = BM25_INDEX_DIR / "chunk_ids.json"
    meta_path = BM25_INDEX_DIR / "meta.json"

    missing = []
    for path in (pkl_path, ids_path, meta_path):
        if not path.exists():
            errors.add(f"BM25 文件不存在：{path}")
            missing.append(path)
        elif path.stat().st_size == 0:
            errors.add(f"BM25 文件为空：{path}")
            missing.append(path)
    if missing:
        return 0

    with open(ids_path, encoding="utf-8") as f:
        bm25_ids = json.load(f)

    if bm25_ids != order:
        mismatch = next(
            (
                (i, a, b)
                for i, (a, b) in enumerate(zip(bm25_ids, order))
                if a != b
            ),
            None,
        )
        if mismatch:
            i, got, expected = mismatch
            detail = f"第 {i} 项 {got!r} != {expected!r}"
        else:
            detail = f"长度 {len(bm25_ids)} != {len(order)}"
        errors.add(f"BM25 chunk_ids 与 chunks 顺序/内容不一致：{detail}")

    with open(pkl_path, "rb") as f:
        bm25 = pickle.load(f)

    corpus_size = getattr(bm25, "corpus_size", None)
    if corpus_size != len(order):
        errors.add(f"BM25 corpus_size {corpus_size} != chunks 数 {len(order)}")
    doc_freqs = getattr(bm25, "doc_freqs", None)
    if doc_freqs is None or len(doc_freqs) != len(order):
        errors.add("BM25 doc_freqs 数量与 chunks 数不一致")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("count") != len(order):
        errors.add(f"BM25 meta.count {meta.get('count')} != chunks 数 {len(order)}")

    return len(bm25_ids)


def _validate_symbol(chunks, errors: ValidationErrors):
    for path in (SYMBOL_INDEX_PATH, SYMBOL_LOWER_INDEX_PATH):
        if not path.exists():
            errors.add(f"符号索引文件不存在：{path}")
            return 0, 0
        if path.stat().st_size == 0:
            errors.add(f"符号索引文件为空：{path}")
            return 0, 0

    with open(SYMBOL_INDEX_PATH, encoding="utf-8") as f:
        symbol_index = json.load(f)

    with open(SYMBOL_LOWER_INDEX_PATH, encoding="utf-8") as f:
        symbol_lower_index = json.load(f)

    for index in (symbol_index, symbol_lower_index):
        if not isinstance(index, dict) or not index:
            errors.add("符号索引不是非空 JSON 对象")
            return 0, 0

    referenced = set()
    index_path = [SYMBOL_INDEX_PATH, SYMBOL_LOWER_INDEX_PATH]
    for i, index in enumerate((symbol_index, symbol_lower_index)):
        for symbol, chunk_ids in index.items():
            if not isinstance(chunk_ids, list) or not chunk_ids:
                errors.add(f"索引 {index_path[i]} 符号索引 {symbol} 的 value 不是非空列表")
                continue
            for chunk_id in chunk_ids:
                if not isinstance(chunk_id, str):
                    errors.add(f"索引 {index_path[i]} 符号索引 {symbol} 含非字符串 chunk_id")
                    continue
                if chunk_id not in chunks:
                    errors.add(f"索引 {index_path[i]} 符号索引 {symbol} 引用了不存在的 chunk_id：{chunk_id}")
                referenced.add(chunk_id)

    for symbol, chunk_ids in symbol_lower_index.items():
        if symbol != symbol.lower():
            errors.add(f"符号索引 key 不是小写：{symbol}")

    return len(symbol_index), len(referenced)


def validate_indexes(chunks_path) -> dict:
    chunks_path = Path(chunks_path)
    errors = ValidationErrors()

    chunks, order = _load_chunks(chunks_path, errors)
    qdrant_count = _validate_qdrant(chunks, errors)
    bm25_count = _validate_bm25(order, errors)
    symbol_count, symbol_refs = _validate_symbol(chunks, errors)

    errors.raise_if_any()

    summary = {
        "chunks": len(order),
        "qdrant": qdrant_count,
        "bm25": bm25_count,
        "symbols": symbol_count,
        "symbol_refs": symbol_refs,
    }
    logger.info(
        "索引校验通过：chunks={chunks}, qdrant={qdrant}, bm25={bm25}, "
        "symbols={symbols}, symbol_refs={symbol_refs}".format(**summary)
    )
    return summary


if __name__ == "__main__":
    validate_indexes(PROJECT_DIR / "data/chunks.jsonl")
