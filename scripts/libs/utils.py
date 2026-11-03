from pathlib import Path

# 以本文件位置为锚点，与调用方脚本的深度无关
LIBS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = LIBS_DIR.parent
PROJECT_DIR = SCRIPTS_DIR.parent

EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNKS_PATH = PROJECT_DIR / "data/chunks.jsonl"
MODELS_CACHE_DIR = PROJECT_DIR / "models_cache"
QURANT_DATA_DIR = PROJECT_DIR / "data/qdrant_data"
SYMBOL_INDEX_DIR = PROJECT_DIR / "data/symbol_index"
SYMBOL_INDEX_PATH = SYMBOL_INDEX_DIR / "symbol_index.json"
DOCS_DIRS = ("about", "classes", "community", "engine_details", "getting_started", "tutorials")

# 过滤掉太通用的词，避免符号索引被噪音淹没
STOPWORDS = {"what", "the", "a", "an", "is", "are", "to", "of", "and", "in", "on",
             "why", "how", "for", "with", "this", "that", "it", "as", "be", "or", "if",
             "when", "where", "which", "can", "do", "does", "did", "from", "by", "at",
             "not", "but", "all", "any", "some", "each", "every", "other", "more", "most",
             "should", "such", "no", "nor", "too", "very", "class"}


def get_current_ip_country() -> str:
    from ipspot import get_public_ipv4, IPv4API
    """使用 ipspot 一步到位，获取IP并查询其国家代码"""
    # 调用 ip-api.com 作为数据源，并请求地理信息
    result = get_public_ipv4(api=IPv4API.IP_API_COM, geo=True, timeout=10)
    
    if not result.get("status"):
        print(f"Get ip information failed: {result}")
        return ''
    
    data = result["data"]
    country_code = data.get("country_code")
    return country_code


def install_modelscope():
    import subprocess
    command = ["pip", "install", "modelscope", "-i", "https://mirrors.aliyun.com/pypi/simple/"]
    subprocess.run(command, cwd=PROJECT_DIR, check=True, text=True, capture_output=False)


def load_model_from_modelscope(model_name):
    try:
        from modelscope import snapshot_download
    except ModuleNotFoundError:
        install_modelscope()
        from modelscope import snapshot_download
    model_dir = snapshot_download(model_name, cache_dir=MODELS_CACHE_DIR)
    return model_dir


def get_model():
    from torch import cuda
    from sentence_transformers import SentenceTransformer
    device = 'cuda' if cuda.is_available() else 'cpu'
    if not MODELS_CACHE_DIR.exists():
        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if get_current_ip_country() == 'CN':
            model_dir = load_model_from_modelscope(EMBEDDING_MODEL)
            return SentenceTransformer(str(model_dir), device=device)
        return SentenceTransformer(EMBEDDING_MODEL, device=device, cache_folder=str(MODELS_CACHE_DIR))

    if (MODELS_CACHE_DIR / "models/BAAI--bge-m3").exists():
        model_dir = load_model_from_modelscope(EMBEDDING_MODEL)
        return SentenceTransformer(str(model_dir), device=device)
    
    if (MODELS_CACHE_DIR / "BAAI--bge-m3").exists():
        model_dir = MODELS_CACHE_DIR / "models/BAAI--bge-m3"
        return SentenceTransformer(str(model_dir), device=device)
    
    if get_current_ip_country() == 'CN':
        model_dir = load_model_from_modelscope(EMBEDDING_MODEL)
        return SentenceTransformer(str(model_dir), device=device)
    
    return SentenceTransformer(EMBEDDING_MODEL, device=device, cache_folder=str(MODELS_CACHE_DIR))