import os
import re
import shutil
import subprocess
import sys
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


def format_duration(seconds: float) -> str:
    """把秒数格式化为人类可读的时长，如 3.5 秒 / 37 分 12 秒 / 2 小时 05 分 03 秒"""
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m} 分 {s:02d} 秒"
    h, m = divmod(m, 60)
    return f"{h} 小时 {m:02d} 分 {s:02d} 秒"


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


def resolve_model_dir():
    """在缓存目录里找已下载的模型；找到任何候选就返回，找不到返回 None。

    候选覆盖两种下载器的落盘布局：
    - models/BAAI--bge-m3：ModelScope（CN 机器实际使用的）
    - models--BAAI--bge-m3：HuggingFace / SentenceTransformer（非 CN 机器）
    两种布局的模型本体都在各自的 snapshots/<版本>/ 下；
    目录是否完整不在这里判断，交给 get_model 里的 SentenceTransformer
    实际加载来裁决（加载失败会走重新下载兜底）。
    """
    if not MODELS_CACHE_DIR.exists():
        return None

    for candidate in (
        MODELS_CACHE_DIR / "models/BAAI--bge-m3",      # ModelScope 布局
        MODELS_CACHE_DIR / "models--BAAI--bge-m3",     # HuggingFace / SentenceTransformer 布局
        MODELS_CACHE_DIR / "BAAI--bge-m3",             # 手工放置/其它
    ):
        if not candidate.is_dir():
            continue
        snapshots_dir = candidate / "snapshots"
        if snapshots_dir.is_dir():
            versions = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if versions:
                return max(versions, key=lambda p: p.stat().st_mtime)
        return candidate    # 无 snapshots 层（扁平布局）：原样返回，交给加载器裁决
    return None


GPU_SMOKE_TEST = """
import torch

assert torch.cuda.is_available()
x = torch.randn(2, 16, 64, device="cuda")
w = torch.randn(64, 64, device="cuda")
y = torch.nn.functional.layer_norm(x @ w, (64,))
y = torch.nn.functional.gelu(y)
y = torch.softmax(y, dim=-1)
table = torch.randn(128, 64, device="cuda")
e = torch.nn.functional.embedding(torch.arange(128, device="cuda"), table)
torch.cuda.synchronize()
assert torch.isfinite(y).all() and torch.isfinite(e).all()
"""


def gpu_is_usable() -> bool:
    """在子进程中探测 GPU，避免损坏的驱动/后端把主进程一起带崩。"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", GPU_SMOKE_TEST],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def detect_rocm_override() -> str | None:
    """ROCm 版 PyTorch 可能没有为当前 GPU 编译内核，例如 gfx1031。

    如果 rocminfo 报出的架构不在 torch 的编译列表里，就找一个同系列的
    架构并返回对应的 HSA_OVERRIDE_GFX_VERSION，让 HIP 使用兼容内核。
    """
    existing = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if existing:
        return existing

    rocminfo = shutil.which("rocminfo")
    if rocminfo is None:
        candidate = Path("/opt/rocm/bin/rocminfo")
        rocminfo = str(candidate) if candidate.is_file() else None
    if rocminfo is None:
        return None

    try:
        result = subprocess.run(
            [rocminfo], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None

    match = re.search(r"\bgfx[0-9a-f]{3,4}\b", result.stdout)
    if not match:
        return None
    device_arch = match.group(0)

    from torch import cuda

    compiled_arches = cuda.get_arch_list()
    if not compiled_arches or device_arch in compiled_arches:
        return None

    # gfx1031 -> gfx103, gfx1103 -> gfx110
    compatible = next(
        (
            arch
            for arch in compiled_arches
            if arch[:-1] == device_arch[:-1]
            and re.fullmatch(r"gfx[0-9]{4}", arch)
        ),
        None,
    )
    if compatible is None:
        return None

    version = re.fullmatch(r"gfx([0-9]{2})([0-9])([0-9])", compatible)
    if version is None:
        return None
    return ".".join(version.groups())


def allow_cpu_fallback() -> bool:
    """CPU 只能通过显式环境变量启用，避免悄悄回退到 CPU。"""
    return os.environ.get("GODOT_RAG_ALLOW_CPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def describe_device(device: str) -> str:
    """返回便于阅读的设备名称，例如 'CUDA (AMD Radeon Graphics)' 或 'CPU'。"""
    if not device.lower().startswith("cuda"):
        return "CPU"
    try:
        from torch import cuda

        index = cuda.current_device()
        if ":" in device:
            index = int(device.split(":", 1)[1])
        return f"CUDA ({cuda.get_device_name(index)})"
    except Exception:
        return "CUDA"


def get_device() -> str:
    override = os.environ.get("GODOT_RAG_DEVICE", "").strip().lower()
    if override == "cpu":
        print(
            "WARNING: GODOT_RAG_DEVICE=cpu was set explicitly; "
            "encoding will run on CPU.",
            flush=True,
        )
        return "cpu"

    rocm_override = detect_rocm_override()
    if rocm_override:
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", rocm_override)

    if override:
        return override

    from torch import cuda

    if not cuda.is_available():
        if allow_cpu_fallback():
            print(
                "WARNING: CUDA/ROCm is not available and GODOT_RAG_ALLOW_CPU "
                "is set; using CPU.",
                flush=True,
            )
            return "cpu"
        raise RuntimeError(
            "CUDA/ROCm is not available. Refusing to run the pipeline on CPU. "
            "Fix the GPU setup, or set GODOT_RAG_ALLOW_CPU=1 to override."
        )

    if gpu_is_usable():
        if rocm_override:
            from loguru import logger

            logger.info(
                f"Using HSA_OVERRIDE_GFX_VERSION={rocm_override} "
                "for the current ROCm GPU."
            )
        return "cuda"

    if allow_cpu_fallback():
        print(
            "WARNING: The GPU backend failed the smoke test and "
            "GODOT_RAG_ALLOW_CPU is set; using CPU.",
            flush=True,
        )
        return "cpu"

    raise RuntimeError(
        "The GPU backend failed the smoke test. Refusing to fall back to CPU. "
        "Fix the GPU setup, or set GODOT_RAG_ALLOW_CPU=1 to override."
    )


def get_model(device: str | None = None, max_seq_length: int | None = None):
    """加载 bge-m3。

    截断上限唯一在此处确定：显式参数 max_seq_length 优先，
    其次是环境变量 GODOT_RAG_MAX_SEQ_LENGTH，都不传则用默认 4096。
    """
    from sentence_transformers import SentenceTransformer

    device = device or get_device()
    print(f"Encoding device: {describe_device(device)}", flush=True)

    model = None
    candidate = resolve_model_dir()
    if candidate is not None:
        try:
            model = SentenceTransformer(str(candidate), device=device)
        except Exception as exc:
            # 缓存目录可能残缺（下载中断等）：加载失败就重新下载一次。
            # 这里不自动删除缓存目录：加载失败也可能只是暂时性的
            # （文件被占用），误删好缓存代价更大；反复失败再手动删
            print(
                f"WARNING: 缓存模型加载失败（{type(exc).__name__}: {exc}），"
                f"将重新下载模型；若反复失败可手动删除 {candidate}",
                flush=True,
            )

    if model is None:
        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_dir = None
        if get_current_ip_country() == "CN":
            model_dir = load_model_from_modelscope(EMBEDDING_MODEL)
        if model_dir is None:
            model = SentenceTransformer(
                EMBEDDING_MODEL, device=device, cache_folder=str(MODELS_CACHE_DIR)
            )
        else:
            model = SentenceTransformer(
                str(model_dir), device=device
            )

    if max_seq_length is None:
        max_seq_length = int(os.environ.get("GODOT_RAG_MAX_SEQ_LENGTH", "4096"))
    if max_seq_length > 0:
        # 只能收紧，不能超过模型自身的 token 上限
        model.max_seq_length = min(model.max_seq_length, max_seq_length)
    return model
