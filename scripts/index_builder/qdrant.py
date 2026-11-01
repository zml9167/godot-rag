import json
import uuid
from loguru import logger
from torch import cuda
from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from ..paths import PROJECT_DIR

NAMESPACE = uuid.NAMESPACE_DNS

COLLECTION_NAME = "godot_docs"
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_DIM = 1024  # bge-m3 的输出维度，请根据实际模型确认

CURRENT_PATH = Path(__file__)
QURANT_DATA_DIR = PROJECT_DIR / "data/qdrant_data"
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"
MODELS_CACHE_DIR = PROJECT_DIR / "models_cache"


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
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(MODELS_CACHE_DIR),
        device=device
    )

    # 5. 读取 chunks 并生成向量
    logger.info('开始生成向量')
    points = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            vector = model.encode(c["content"], normalize_embeddings=True).tolist()

            # 6. 构造 Point：id + vector + payload（元数据）
            points.append(
                models.PointStruct(
                    id=chunk_id_to_point_id(c["chunk_id"]),
                    vector=vector,
                    payload={
                        "chunk_id": c["chunk_id"],
                        "source": c["source"],
                        "header_path": c["header_path"],
                        "chunk_type": c["chunk_type"],
                        "content": c["content"],
                    },
                )
            )

    # 7. 批量写入
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,  # 等待写入完成，确保数据立即可查
    )
    logger.info(f"已写入 {len(points)} 个 chunk 到 Qdrant")


if __name__ == "__main":
    chunks = PROJECT_DIR / "data/chunks.jsonl"
    build(chunks)
