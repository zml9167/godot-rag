import re
import json
import pickle
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .libs.tokenizer import tokenize
from .libs.utils import PROJECT_DIR, QURANT_DATA_DIR, get_model_name

# ============ 全局加载（进程启动时加载一次） ============
PROJECT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_DIR / "data/chunks.jsonl"
QDRANT_PATH = PROJECT_DIR / "data/qdrant_data"
BM25_DIR = PROJECT_DIR / "data/bm25_index"
SYMBOL_INDEX_PATH = PROJECT_DIR / "data/symbol_index/symbol_index.json"
MODELS_CACHE_DIR = PROJECT_DIR / "models_cache"
COLLECTION_NAME = "godot_docs"
EMBEDDING_MODEL = "BAAI/bge-m3"


SYMBOL_PATTERN = re.compile(
    r'`([A-Za-z_][A-Za-z0-9_.]*)`'          # 反引号包裹的
    r'|\b([A-Z][a-zA-Z0-9]+(?:\.[a-z_][a-z0-9_]*)?)\b'  # CamelCase 或 Class.method
    r'|\b([a-z_][a-z0-9_]{2,})\b'           # snake_case（至少3字符）
)


# chunk_id -> chunk 完整内容，用于最终补全
CHUNK_MAP = {}
with open(CHUNKS_PATH, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        CHUNK_MAP[c["chunk_id"]] = c

# BM25 索引
with open(BM25_DIR / "bm25.pkl", "rb") as f:
    BM25 = pickle.load(f)
with open(BM25_DIR / "chunk_ids.json", encoding="utf-8") as f:
    BM25_CHUNK_IDS = json.load(f)

# 符号索引
with open(SYMBOL_INDEX_PATH, encoding="utf-8") as f:
    SYMBOL_INDEX = json.load(f)

MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Qdrant 客户端与 embedding 模型
QDRANT = QdrantClient(path=QURANT_DATA_DIR)

model = SentenceTransformer(get_model_name())


def detect_symbols(query: str) -> list[str]:
    candidates = set()
    for m in SYMBOL_PATTERN.finditer(query):
        sym = next(g for g in m.groups() if g)
        candidates.add(sym)
    # 只保留在符号索引里真实存在的
    return [s for s in candidates if s.lower() in SYMBOL_INDEX]


def symbol_lookup(symbols: list[str], top_k: int = 3) -> list[dict]:
    hits = []
    for sym in symbols:
        chunk_ids = SYMBOL_INDEX.get(sym.lower(), [])
        # API 参考块优先，教程块次之
        ranked = sorted(
            chunk_ids,
            key=lambda cid: (CHUNK_MAP[cid]["chunk_type"] != "api_reference",
                             len(CHUNK_MAP[cid]["content"]))
        )
        for cid in ranked[:top_k]:
            hits.append({
                "chunk_id": cid,
                "source": "symbol",
                "score": 1.0,  # 符号直达给最高分
                **CHUNK_MAP[cid],
            })
    return hits

def vector_search(query: str, top_n: int = 30) -> list[tuple[str, float]]:
    qv = MODEL.encode(query, normalize_embeddings=True).tolist()
    res = QDRANT.query_points(
        collection_name=COLLECTION_NAME, query=qv,
        limit=top_n, with_payload=True,
    )
    return [(p.payload["chunk_id"], p.score) for p in res.points]


def bm25_search(query: str, top_n: int = 30) -> list[tuple[str, float]]:
    tokens = tokenize(query)  # 必须和建索引时同一个分词函数
    scores = BM25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [(BM25_CHUNK_IDS[i], s) for i, s in ranked if s > 0]


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: 多路检索结果，每路是 [(chunk_id, score), ...] 按分数降序"""
    fused = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def retrieve(query: str, top_k: int = 6) -> list[dict]:
    # 1. 符号直达
    symbols = detect_symbols(query)
    symbol_hits = symbol_lookup(symbols, top_k=3)

    # 2. 混合检索
    vec = vector_search(query, top_n=30)
    bm = bm25_search(query, top_n=30)
    fused = rrf_fuse([vec, bm])

    # 3. 合并：符号直达置顶，混合结果按 RRF 排序补充
    seen = set()
    final = []
    for h in symbol_hits:
        if h["chunk_id"] not in seen:
            seen.add(h["chunk_id"])
            final.append(h)

    for cid, rrf_score in fused:
        if len(final) >= top_k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        final.append({
            "chunk_id": cid,
            "source": "hybrid",
            "score": rrf_score,
            **CHUNK_MAP[cid],
        })

    return final[:top_k]


if __name__ == "__main__":
    res = retrieve("什么是CharacterBody2D")
    print("检索结果：")
    print(res)