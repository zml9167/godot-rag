import json
import os
import uuid
from loguru import logger
from pathlib import Path
from qdrant_client import QdrantClient, models

from ..utils import (
    PROJECT_DIR,
    QURANT_DATA_DIR,
    MODELS_CACHE_DIR,
    get_model,
)

NAMESPACE = uuid.NAMESPACE_DNS

COLLECTION_NAME = "godot_docs"
VECTOR_DIM = 1024  # bge-m3 的输出维度，请根据实际模型确认

CURRENT_PATH = Path(__file__)
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"


def _chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def _to_point(chunk: dict, vector: list[float]) -> models.PointStruct:
    return models.PointStruct(
        id=_chunk_id_to_point_id(chunk["chunk_id"]),
        vector=vector,
        payload={
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "header_path": chunk["header_path"],
            "chunk_type": chunk["chunk_type"],
            "content": chunk["content"],
        },
    )


def _estimate_tokens(text: str) -> int:
    # 粗略估算，用于控制单批次的显存占用；模型自身仍会截断到 max_seq_length
    return min(max(1, len(text) // 4), 8192)


def _batched_chunks(handle, max_batch_size: int, max_batch_tokens: int):
    batch, tokens = [], 0
    for line in handle:
        chunk = json.loads(line)
        chunk_tokens = _estimate_tokens(chunk["content"])
        if batch and tokens + chunk_tokens > max_batch_tokens:
            yield batch
            batch, tokens = [], 0
        batch.append(chunk)
        tokens += chunk_tokens
        if len(batch) >= max_batch_size:
            yield batch
            batch, tokens = [], 0
    if batch:
        yield batch


def build(chunks_path: str):
    qdrant_path = str(QURANT_DATA_DIR)
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOG_PATH)
    # 1. 初始化本地持久化客户端
    client = QdrantClient(path=qdrant_path)

    # 2. 如果 Collection 已存在，先删除（方便重建）
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # 3. 创建 Collection，指定向量维度和余弦距离
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_DIM,
            distance=models.Distance.COSINE,
        ),
    )

    # 4. 加载 embedding 模型
    model = get_model()
    is_gpu = model.device.type == "cuda"
    batch_size = int(
        os.environ.get("GODOT_RAG_EMBED_BATCH_SIZE", "8" if is_gpu else "16")
    )
    # 单个批次的总 token 预算：GPU 上注意力显存随序列长度平方增长，预算更保守；
    # CPU 没有显存限制，可以放大预算以提高吞吐。
    max_batch_tokens = int(
        os.environ.get(
            "GODOT_RAG_EMBED_MAX_BATCH_TOKENS", "8192" if is_gpu else "16384"
        )
    )
    logger.info(
        f"开始生成向量（device={model.device}, batch_size={batch_size}, "
        f"max_batch_tokens={max_batch_tokens}）"
    )

    # 5. 分批读取 chunks、生成向量并写入，避免一次性占用过多内存
    total = 0
    last_logged = 0
    with open(chunks_path, encoding="utf-8") as f:
        for batch in _batched_chunks(f, batch_size, max_batch_tokens):
            vectors = model.encode(
                [c["content"] for c in batch],
                batch_size=len(batch),
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[_to_point(c, v) for c, v in zip(batch, vectors)],
                wait=True,
            )
            total += len(batch)
            if total - last_logged >= 1000:
                logger.info(f"已写入 {total} 个 chunk")
                last_logged = total

    logger.info(f"已写入 {total} 个 chunk 到 Qdrant")
    client.close()


if __name__ == "__main__":
    chunks = PROJECT_DIR / "data/chunks.jsonl"
    build(chunks)
