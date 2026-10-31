import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "godot_docs"
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_DIM = 1024  # bge-m3 的输出维度，请根据实际模型确认

PROJECT_DIR = Path(__file__).parents[1]
BM25_INDEX_DIR = PROJECT_DIR / "data" / "bm25_index"
SYMBOL_INDEX_DIR = PROJECT_DIR / "data" / "symbol_index"
QURANT_DATA_DIR = PROJECT_DIR / "data" / "qdrant_data"
LOG_PATH = PROJECT_DIR / "logs" / "build_index.log"
MODEL_DIR = PROJECT_DIR / "models"

# 匹配 Godot 风格的符号：CharacterBody2D、move_and_slide、velocity 等
SYMBOL_RE = re.compile(r'\b([A-Z][A-Za-z0-9]*|[a-z_][a-z0-9_]*)\b')

# 过滤掉太通用的词，避免符号索引被噪音淹没
STOPWORDS = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "on",
             "for", "with", "this", "that", "it", "as", "be", "or", "if"}

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text):
    # 保留 API 符号整体：字母数字下划线组成的 token 不拆
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]', text.lower())
    return tokens


def build_bm25_index(chunks_path, index_dir):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids, tokenized_corpus = [], []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_ids.append(c["chunk_id"])
            tokenized_corpus.append(tokenize(c["content"]))

    bm25 = BM25Okapi(tokenized_corpus)

    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(index_dir / "chunk_ids.json", "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)
    with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"tokenizer": "custom_v1", "count": len(chunk_ids)}, f)
    print("bm25 index done")


def extract_symbols(chunk):
    symbols = set()

    # 1. section_path 最后一级
    if chunk["header_path"]:
        last = chunk["header_path"][-1]
        # 去掉括号、参数等，只保留标识符
        for m in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', last):
            if m.lower() not in STOPWORDS:
                symbols.add(m)

    # 2. content 中形如 `Class.method` 或独立 API 名的 token
    for m in re.findall(r'`([A-Za-z_][A-Za-z0-9_.]*)`', chunk["content"]):
        symbols.add(m)
    for m in re.findall(r'\b([A-Z][a-z]+[A-Za-z0-9]*)\b', chunk["content"]):
        symbols.add(m)

    return {s for s in symbols if s.lower() not in STOPWORDS and len(s) >= 3}


def build_symbol_index(chunks_path: str, index_dir: str):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    symbol_to_chunks = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            for sym in extract_symbols(c):
                symbol_to_chunks.setdefault(sym.lower(), []).append(c["chunk_id"])

    with open(index_dir / "symbol_index.json", "w", encoding="utf-8") as f:
        json.dump(symbol_to_chunks, f, ensure_ascii=False)
    print(f"symbol index done: {len(symbol_to_chunks)} symbols")


def build_qdrant_index(chunks_path: str, qdrant_path: str):
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
    model_dir = snapshot_download(EMBEDDING_MODEL, cache_dir=str(MODEL_DIR))
    model = SentenceTransformer(
        model_dir,
        cache_folder=str(MODEL_DIR),
        device="cuda"
    )

    # 5. 读取 chunks 并生成向量
    points = []
    with open(chunks_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            c = json.loads(line)
            vector = model.encode(c["content"], normalize_embeddings=True).tolist()

            # 6. 构造 Point：id + vector + payload（元数据）
            points.append(
                models.PointStruct(
                    id=idx,  # 或使用稳定的 chunk_id 映射为整数
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
    print(f"已写入 {len(points)} 个 chunk 到 Qdrant")


if __name__ == "__main__":
    chunks = str(PROJECT_DIR / "data" / "chunks" / "chunk.jsonl")
    build_qdrant_index(chunks, QURANT_DATA_DIR)
    build_bm25_index(chunks, BM25_INDEX_DIR)
    build_symbol_index(chunks, SYMBOL_INDEX_DIR)