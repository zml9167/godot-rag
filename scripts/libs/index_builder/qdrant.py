import json
import uuid
from loguru import logger
from pathlib import Path
from qdrant_client import QdrantClient, models

import numpy as np

from ..utils import PROJECT_DIR, QURANT_DATA_DIR, MODELS_CACHE_DIR, get_model

NAMESPACE = uuid.NAMESPACE_DNS

COLLECTION_NAME = "godot_docs"
VECTOR_DIM = 1024  # bge-m3 的输出维度，请根据实际模型确认
UPSERT_BATCH_SIZE = 256
# 按 token 预算组 batch，兼顾吞吐和显存；8192 tokens 的单条会单独成批
TOKEN_BUDGET = 4096
MAX_BATCH_SIZE = 32

CURRENT_PATH = Path(__file__)
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


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

    # 5. 读取 chunks，先统计 token 长度
    logger.info('读取 chunks 并统计 token 长度')
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    if not chunks:
        raise ValueError(f"chunks 文件为空：{chunks_path}")

    token_lengths = [
        len(model.tokenizer.encode(
            c["content"],
            add_special_tokens=True,
            truncation=True,
            max_length=model.max_seq_length,
        ))
        for c in chunks
    ]

    # 按长度排序后贪心分组，保证 max_len * batch_count <= TOKEN_BUDGET。
    # 用 max_len 而不是 token 总和，避免长短混排时 padding 把显存撑爆。
    order = sorted(range(len(chunks)), key=lambda i: token_lengths[i])
    groups = []
    current = []
    current_max = 0
    for idx in order:
        length = token_lengths[idx]
        new_max = max(current_max, length)
        if current and (
            new_max * (len(current) + 1) > TOKEN_BUDGET
            or len(current) + 1 > MAX_BATCH_SIZE
        ):
            groups.append(current)
            current = []
            current_max = 0
            new_max = length
        current.append(idx)
        current_max = new_max
    if current:
        groups.append(current)

    logger.info(
        f"共 {len(chunks)} 个 chunk，{len(groups)} 个编码批次，"
        f"token budget={TOKEN_BUDGET}，max batch={MAX_BATCH_SIZE}"
    )

    points = []
    written = 0
    skipped = 0
    cpu_model = None

    def encode_batch(texts):
        try:
            return model.encode(
                texts,
                batch_size=len(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.warning(
                f"批量编码失败（{type(exc).__name__}: {exc}），改为逐条编码"
            )
            return np.stack([
                model.encode(t, normalize_embeddings=True, show_progress_bar=False)
                for t in texts
            ])

    def encode_on_cpu(text):
        nonlocal cpu_model
        if cpu_model is None:
            cpu_model = get_model(device='cpu')
        return cpu_model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )

    def flush_points():
        nonlocal points, written
        if not points:
            return
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,  # 等待写入完成，确保数据立即可查
        )
        written += len(points)
        points = []
        logger.info(f"已写入 {written} 个 chunk")

    # 6. 按批次编码并写入
    for group_no, group in enumerate(groups, start=1):
        texts = [chunks[i]["content"] for i in group]
        vectors = encode_batch(texts)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        for row, idx in enumerate(group):
            c = chunks[idx]
            vector = vectors[row]

            # GPU 在个别 token 长度上会输出 NaN，失败时用 CPU 重算该 chunk
            if not np.isfinite(vector).all():
                logger.warning(f"GPU 生成了非有限向量，改用 CPU 重试：{c['chunk_id']}")
                vector = encode_on_cpu(c["content"])
                if not np.isfinite(vector).all():
                    logger.error(f"CPU 仍生成非有限向量，已跳过：{c['chunk_id']}")
                    skipped += 1
                    continue

            # 7. 构造 Point：id + vector + payload（元数据）
            points.append(
                models.PointStruct(
                    id=chunk_id_to_point_id(c["chunk_id"]),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": c["chunk_id"],
                        "source": c["source"],
                        "header_path": c["header_path"],
                        "chunk_type": c["chunk_type"],
                        "content": c["content"],
                    },
                )
            )

            # 8. 分批写入，避免中途失败时丢失全部进度
            if len(points) >= UPSERT_BATCH_SIZE:
                flush_points()

        if group_no % 50 == 0:
            logger.info(f"编码进度 {group_no}/{len(groups)} 批")

    flush_points()
    logger.info(f"已写入 {written} 个 chunk 到 Qdrant，跳过 {skipped} 个")
    client.close()


if __name__ == "__main__":
    chunks = PROJECT_DIR / "data/chunks.jsonl"
    build(chunks)
