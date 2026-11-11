from pathlib import Path

# 以本文件位置为锚点，与调用方脚本的深度无关
LIBS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = LIBS_DIR.parent
PROJECT_DIR = SCRIPTS_DIR.parent

CHUNKS_PATH = PROJECT_DIR / "data/chunks.jsonl"
QURANT_DATA_DIR = PROJECT_DIR / "data/qdrant_data"
SYMBOL_INDEX_DIR = PROJECT_DIR / "data/symbol_index"
SYMBOL_INDEX_PATH = SYMBOL_INDEX_DIR / "symbol_index.json"
SYMBOL_LOWER_INDEX_PATH = SYMBOL_INDEX_DIR / "symbol_lower_index.json"
DOCS_DIRS = ("about", "classes", "community", "engine_details", "getting_started", "tutorials")

# 过滤掉太通用的词，避免符号索引被噪音淹没
STOPWORDS = {
    "what", "the", "a", "an", "is", "are", "to", "of", "and", "in", "on",
    "why", "how", "for", "with", "this", "that", "it", "as", "be", "or", "if",
    "when", "where", "which", "can", "do", "does", "did", "from", "by", "at",
    "not", "but", "all", "any", "some", "each", "every", "other", "more", "most",
    "should", "such", "no", "nor", "too", "very", "class"}


def format_duration(seconds: float) -> str:
    """把秒数格式化为人类可读的时长，如 3.5 秒 / 37 分 12 秒 / 2 小时 05 分 03 秒"""
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m} 分 {s:02d} 秒"
    h, m = divmod(m, 60)
    return f"{h} 小时 {m:02d} 分 {s:02d} 秒"


_embedding_client = None


def get_embedding_client():
    """SiliconFlow embeddings API 客户端单例（进程内复用连接池，密钥缺失时在此报错）。"""
    global _embedding_client
    if _embedding_client is None:
        from .siliconflow import SiliconFlowEmbedding

        _embedding_client = SiliconFlowEmbedding()
    return _embedding_client


_reranker_client = None


def get_reranker_client():
    """SiliconFlow rerank API 客户端单例（与 embedding 客户端各自持有 Session）。"""
    global _reranker_client
    if _reranker_client is None:
        from .siliconflow import SiliconFlowReranker

        _reranker_client = SiliconFlowReranker()
    return _reranker_client


_qdrant_client = None


def get_qdrant_client():
    """Qdrant 本地模式只读客户端单例（进程内复用，不主动 close）"""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient

        _qdrant_client = QdrantClient(
            path=str(QURANT_DATA_DIR), force_disable_check_same_thread=True
        )
    return _qdrant_client
