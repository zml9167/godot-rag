import json
import os
import struct
import subprocess
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


def configure_rocm():
    """ROCm 的 PyTorch wheel 不包含 gfx1031（RX 6700 XT）的 kernel，借用 gfx1030 的。"""
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION") or not Path("/opt/rocm").exists():
        return
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    if "gfx1031" in result.stdout:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"


# 必须在 import torch / 初始化 HIP 之前设置
configure_rocm()


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


def is_valid_safetensors(path):
    """只解析 safetensors 头部，确认偏移量没有超出文件大小。"""
    path = Path(path)
    try:
        with path.open("rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return False
            (header_len,) = struct.unpack("<Q", raw_len)
            # 正常模型的头部只有几 KB；限制上限避免读到损坏的巨大长度
            if header_len <= 1 or header_len > 100 * 1024 * 1024:
                return False
            header = json.loads(f.read(header_len))

        max_end = 0
        tensor_count = 0
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            if start < 0 or end < start:
                return False
            max_end = max(max_end, end)
            tensor_count += 1

        data_start = 8 + header_len
        return tensor_count > 0 and data_start + max_end <= path.stat().st_size
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def ensure_safetensors(model_dir):
    """torch < 2.6 时 transformers 拒绝加载 .bin，这里把旧权重转成 safetensors。"""
    model_dir = Path(model_dir)
    safe_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"

    if safe_path.exists():
        if is_valid_safetensors(safe_path):
            return model_dir
        print(f"{safe_path} 已损坏或不完整，将重新生成", flush=True)
        safe_path.unlink()

    sharded_bins = sorted(model_dir.glob("pytorch_model-*-of-*.bin"))
    if sharded_bins:
        raise ValueError(
            f"暂不支持分片 .bin 权重（{sharded_bins[0].name}），"
            "请先从 Hugging Face 下载 safetensors 版本。"
        )

    if not bin_path.exists():
        return model_dir

    import torch
    from safetensors.torch import save_file

    tmp_path = model_dir / "model.safetensors.tmp"
    print(f"正在把 {bin_path} 转换为 safetensors（首次运行较慢）...", flush=True)
    try:
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

        # safetensors 不能保存共享同一块存储或非连续的 tensor
        seen = set()
        for key, tensor in state_dict.items():
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            ptr = tensor.untyped_storage().data_ptr()
            if ptr in seen:
                tensor = tensor.clone()
            else:
                seen.add(ptr)
            state_dict[key] = tensor

        save_file(state_dict, str(tmp_path), metadata={"format": "pt"})
        os.replace(tmp_path, safe_path)
    finally:
        state_dict = None
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"已生成 {safe_path}", flush=True)
    return model_dir


def resolve_model_dir():
    if not MODELS_CACHE_DIR.exists():
        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if get_current_ip_country() == 'CN':
            return load_model_from_modelscope(EMBEDDING_MODEL)
        return None

    for candidate in (
        MODELS_CACHE_DIR / "models/BAAI--bge-m3",
        MODELS_CACHE_DIR / "BAAI--bge-m3",
    ):
        if not candidate.exists():
            continue
        if (candidate / "config.json").exists():
            return candidate
        # ModelScope 的目录结构：<model>/snapshots/<revision>/
        snapshots = [
            p for p in (candidate / "snapshots").glob("*") if (p / "config.json").exists()
        ]
        if snapshots:
            return max(snapshots, key=lambda p: p.stat().st_mtime)
        return candidate

    if get_current_ip_country() == 'CN':
        return load_model_from_modelscope(EMBEDDING_MODEL)

    return None


def get_model(device=None):
    """加载 embedding 模型；device 为 None 时优先使用 GPU。"""
    from torch import cuda
    from sentence_transformers import SentenceTransformer
    if device is None:
        device = 'cuda' if cuda.is_available() else 'cpu'
    model_dir = resolve_model_dir()
    if model_dir is None:
        return SentenceTransformer(EMBEDDING_MODEL, device=device, cache_folder=str(MODELS_CACHE_DIR))
    return SentenceTransformer(str(ensure_safetensors(model_dir)), device=device)