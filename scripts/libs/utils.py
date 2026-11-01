import subprocess
from pathlib import Path
from ipspot import get_public_ipv4, IPv4API

# 以本文件位置为锚点，与调用方脚本的深度无关
LIBS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = LIBS_DIR.parent
PROJECT_DIR = SCRIPTS_DIR.parent

EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNKS_PATH = PROJECT_DIR / "data/chunks.jsonl"
MODELS_CACHE_DIR = PROJECT_DIR / "models_cache"
QURANT_DATA_DIR = PROJECT_DIR / "data/qdrant_data"


def get_current_ip_country() -> str:
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


def get_model_name():
    if not MODELS_CACHE_DIR.exists():
        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if get_current_ip_country() == 'CN':
            return load_model_from_modelscope(EMBEDDING_MODEL)
        return EMBEDDING_MODEL
    if (MODELS_CACHE_DIR / "models/BAAI--bge-m3").exists():
        return load_model_from_modelscope(EMBEDDING_MODEL)
    if (MODELS_CACHE_DIR / "BAAI--bge-m3").exists():
        return MODELS_CACHE_DIR / "models/BAAI--bge-m3"
    if get_current_ip_country() == 'CN':
        return load_model_from_modelscope(EMBEDDING_MODEL)
    return EMBEDDING_MODEL