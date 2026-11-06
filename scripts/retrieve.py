import re
import json
import pickle
from pathlib import Path
from qdrant_client import QdrantClient

from .libs.tokenizer import tokenize
from .libs.utils import PROJECT_DIR, QURANT_DATA_DIR, STOPWORDS, SYMBOL_INDEX_PATH, get_model
from .libs.pattern_utils import match_sybol_from_classes_content

# ============ 全局加载（进程启动时加载一次） ============
PROJECT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_DIR / "data/chunks.jsonl"
BM25_DIR = PROJECT_DIR / "data/bm25_index"
COLLECTION_NAME = "godot_docs"
SYMBOL_ORDER = ["classes", "tutorials"]  # 符号索引命中时的优先级顺序

SYMBOL_PATTERN = re.compile(
    r'`([A-Za-z_][A-Za-z0-9_.]*)`'          # 反引号包裹的
    r'|\b([A-Z][A-Za-z0-9]*)\.([a-z_][A-Za-z0-9_]*)\b'  # Class.method 拆开
    r'|\b([A-Z][A-Za-z0-9]+)\b'             # CamelCase
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


# Qdrant 客户端与 embedding 模型
QDRANT = QdrantClient(path=QURANT_DATA_DIR)

model = get_model()


def get_symbol_order(sym: str, chunk) -> tuple:
    """返回符号索引的优先级顺序，越小越优先"""
    order_chunk_type = len(SYMBOL_ORDER)
    if chunk["chunk_type"] in SYMBOL_ORDER:
        order_chunk_type = SYMBOL_ORDER.index(chunk["chunk_type"])

    order_header_path = True
    if chunk["header_path"]:
        order_header_path = True
        for h in chunk["header_path"]:
            if sym.lower() in h.lower():
                order_header_path = False
                break
    order_source = sym.lower() not in chunk['source'].lower()
    order_content = match_sybol_from_classes_content(sym, chunk["content"])
    return (order_header_path, order_source, order_content, order_chunk_type)


def detect_symbols(query: str) -> list[str]:
    candidates = set()
    for m in SYMBOL_PATTERN.finditer(query):
        sym_list = [g for g in m.groups() if g]
        for sym in sym_list:
            if sym.lower() not in STOPWORDS:
                candidates.add(sym)
    # 只保留在符号索引里真实存在的
    return [s for s in candidates if s.lower() in SYMBOL_INDEX]


def symbol_lookup(symbols: list[str], top_k: int = 3) -> list[dict]:
    hits = []
    for sym in symbols:
        # 先按 chunk_id 去重，避免重复项占用 top_k 槽位
        chunk_ids = list(dict.fromkeys(SYMBOL_INDEX.get(sym.lower(), [])))
        # API 参考块优先，教程块次之
        ranked = sorted(
            chunk_ids,
            key=lambda cid: get_symbol_order(sym, CHUNK_MAP[cid]),
        )
        for cid in ranked[:top_k]:
            hits.append({
                **CHUNK_MAP[cid],
                "score": 1.0,  # 符号直达给最高分
            })
    return hits


def vector_search(query: str, top_n: int = 30) -> list[tuple[str, float]]:
    qv = model.encode(query, normalize_embeddings=True).tolist()
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
    seen = set()
    final = []

    # 1. 符号直达
    symbols = detect_symbols(query)
    symbol_hits = symbol_lookup(symbols, top_k=3)
    
    # 符号直达置顶
    for h in symbol_hits:
        if h["chunk_id"] not in seen:
            seen.add(h["chunk_id"])
            h['retrieve_method'] = 'symbol'
            final.append(h)

    # 2. 混合检索
    vec = vector_search(query, top_n=30)
    bm = bm25_search(query, top_n=30)
    fused = rrf_fuse([vec, bm])

    # 混合结果按 RRF 排序补充
    for cid, rrf_score in fused:
        if len(final) >= top_k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        final.append({
            **CHUNK_MAP[cid],
            "score": rrf_score,
            "retrieve_method": "hybrid",
        })

    return final[:top_k]


if __name__ == "__main__":
    res = retrieve("move_and_slide 的 delta 参数怎么传？")
    print("检索结果：")
    print([(h["source"], h["retrieve_method"]) for h in res])