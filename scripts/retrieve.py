import re
import json
import pickle

from .libs.tokenizer import tokenize
from .libs.utils import PROJECT_DIR, STOPWORDS, SYMBOL_INDEX_PATH, SYMBOL_LOWER_INDEX_PATH, get_embedding_client, get_qdrant_client, get_reranker_client
from .libs.pattern_utils import match_sybol_from_classes_content

# ============ 全局加载（进程启动时加载一次） ============
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

with open(SYMBOL_LOWER_INDEX_PATH, encoding="utf-8") as f:
    SYMBOL_LOWER_INDEX = json.load(f)


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
    # 内容命中集合转 bool：命中（非空）→ False → 排前，与 header/source 约定一致
    order_content = not match_sybol_from_classes_content(sym, chunk["content"])
    order_content_lower = True
    if order_content:
        order_content_lower = not match_sybol_from_classes_content(sym.lower(), chunk["content"])
    return (order_header_path, order_source, order_content, order_chunk_type, order_content_lower)


def detect_symbols(query: str) -> list[str]:
    seen = set()
    candidates = []
    for m in SYMBOL_PATTERN.finditer(query):
        sym_list = [g for g in m.groups() if g]
        for sym in sym_list:
            if sym in seen:
                continue
            seen.add(sym)
            # 只保留在符号索引里真实存在的
            if sym in SYMBOL_INDEX:
                candidates.append(sym)
                continue
            sym_lower = sym.lower()
            if sym_lower in STOPWORDS:
                continue
            if sym_lower in SYMBOL_LOWER_INDEX:
                candidates.append(sym_lower)
                continue
    return candidates


def symbol_lookup(symbols: list[str]) -> list[dict]:
    hits = []
    for sym in symbols:
        # 先按 chunk_id 去重
        chunk_ids = list(dict.fromkeys(SYMBOL_INDEX.get(sym, []))) or list(dict.fromkeys(SYMBOL_LOWER_INDEX.get(sym, [])))
        ranked = sorted(
            chunk_ids,
            key=lambda cid: get_symbol_order(sym, CHUNK_MAP[cid]),
        )
        for cid in ranked:
            hits.append({
                **CHUNK_MAP[cid],
                "retrieve_method": "symbol"
                })
    return hits


def vector_search(query: str, top_n: int = 30) -> tuple[tuple[str, float], tuple[str, float]]:
    _embedding_client = get_embedding_client()
    _qdrant_client = get_qdrant_client()
    qv = _embedding_client.encode([query])[0]
    title_res = _qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=qv,
        limit=top_n,
        with_payload=True,
        using="title"
    )
    content_res = _qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=qv,
        limit=top_n,
        with_payload=True,
        using="content"
    )
    return [(p.payload["chunk_id"], p.score) for p in title_res.points], [(p.payload["chunk_id"], p.score) for p in content_res.points]


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


def search_symbols(query: str):
    _reranker_client = get_reranker_client()
    symbols = detect_symbols(query)
    symbol_hits = symbol_lookup(symbols)
    if len(symbol_hits) == 0:
        return symbol_hits
    scores = _reranker_client.rerank(query, [hit['content'] for hit in symbol_hits])
    for i, v in enumerate(scores):
        symbol_hits[i]['reranker_score'] = v
    symbol_hits.sort(key=lambda h: h['reranker_score'], reverse=True)
    return symbol_hits


def search_fuse(query: str, top_k=60):
    _reranker_client = get_reranker_client()
    seen = set()
    result = []
    # 混合检索
    title_vec, content_vec = vector_search(query, top_k>>1)
    bm = bm25_search(query, top_k>>1)
    fused = rrf_fuse([title_vec, content_vec, bm])

    # 混合结果按 RRF 排序补充
    for cid, rrf_score in fused:
        if cid in seen:
            continue
        seen.add(cid)
        result.append({
            **CHUNK_MAP[cid],
            "score": rrf_score,
            "retrieve_method": "hybrid",
        })
    scores = _reranker_client.rerank(query, [res["content"] for res in result])
    for i, v in enumerate(scores):
        result[i]["reranker_score"] = v
    result.sort(key=lambda x: x["reranker_score"], reverse=True)
    return result[: top_k]


def retrieve(query: str, symbol_top_k: int = 3, fuse_top_k: int = 6) -> list[dict]:
    symbol_results = search_symbols(query)[: symbol_top_k]
    fuse_results = search_fuse(query)[:fuse_top_k]
    seen = set()
    final = []
    for results in (symbol_results, fuse_results):
        for result in results:
            if result["chunk_id"] in seen:
                continue
            seen.add(result["chunk_id"])
            final.append(result)
    return final


if __name__ == "__main__":
    res = retrieve("move_and_slide 的 delta 参数怎么传？")
    print("检索结果：")
    print([(h["source"], h["retrieve_method"]) for h in res])