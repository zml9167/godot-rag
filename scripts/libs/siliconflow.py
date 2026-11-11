"""SiliconFlow API 客户端（embeddings + rerank），替代本地 SentenceTransformer。

共享同一条 API Key、Session 与重试策略：
- embeddings：POST /v1/embeddings，文本 input 为单个字符串或字符串数组（单次最多 32 条，官方硬上限）
- rerank：POST /v1/rerank，documents 为字符串数组（无官方硬上限，客户端按 RERANK_BATCH_SIZE 分批）

错误分类：
- 400/401/403/404：配置类错误（请求体/密钥/权限/路径），重试无意义，直接抛错
- 429/503/504 及网络异常：指数退避重试，耗尽后抛错中止

发送前净化（用 bge-m3 同款 tokenizer 精确计数，不依赖字符密度近似）：
- 空白文本替换为 "." 占位符（API 拒绝空字符串）
- 超过 EMBEDDING_MAX_TOKENS 的文本按 tokenizer 截断后再发送
  （bge-m3 单条上限 8192 token；tokenizer.json 首次使用时自动下载到
  models_cache/tokenizers/，约 9MB，ModelScope 优先、HF 兜底）
- 分批双闸门：≤ API_BATCH_SIZE 条 且 批内总 token ≤ API_BATCH_TOKENS
  （token 口径防御请求级总量限制——table:: 标记类文本字符少但 token 密）
"""

import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import requests
from loguru import logger

from .utils import PROJECT_DIR

API_BASE = os.environ.get("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1").rstrip("/")
EMBEDDING_MODEL = os.environ.get("GODOT_RAG_EMBEDDING_MODEL", "BAAI/bge-m3").strip()
EMBEDDING_DIMENSIONS = 1024  # bge-m3 固定输出维度（该模型不支持 dimensions 参数）
API_BATCH_SIZE = int(os.environ.get("GODOT_RAG_API_BATCH_SIZE", "32"))
API_BATCH_TOKENS = int(os.environ.get("GODOT_RAG_API_BATCH_TOKENS", "24576"))
EMBEDDING_MAX_TOKENS = int(os.environ.get("GODOT_RAG_EMBEDDING_MAX_TOKENS", "8176"))  # bge-m3 上限 8192，留 16 特殊 token 余量
TOKENIZER_FILE = Path(os.environ.get(
    "GODOT_RAG_TOKENIZER_FILE",
    str(PROJECT_DIR / "models_cache/tokenizers/bge-m3/tokenizer.json"),
))
# tokenizer.json 下载源：CN 网络走 ModelScope 更快，HF 兜底
TOKENIZER_URLS = (
    "https://modelscope.cn/models/BAAI/bge-m3/resolve/master/tokenizer.json",
    "https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json",
)
RERANKER_MODEL = os.environ.get("GODOT_RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip()
RERANK_BATCH_SIZE = int(os.environ.get("GODOT_RAG_RERANK_BATCH_SIZE", "64"))
API_TIMEOUT = float(os.environ.get("GODOT_RAG_API_TIMEOUT", "60"))
API_MAX_RETRIES = int(os.environ.get("GODOT_RAG_API_MAX_RETRIES", "8"))
RETRY_BASE_DELAY = float(os.environ.get("GODOT_RAG_API_RETRY_BASE_DELAY", "2.0"))
RETRY_MAX_DELAY = float(os.environ.get("GODOT_RAG_API_RETRY_MAX_DELAY", "60.0"))

# 仅这些状态码值得重试；其余状态码重试也不会成功
RETRYABLE_STATUS = frozenset({429, 503, 504})


class SiliconFlowAPIError(RuntimeError):
    """SiliconFlow API 调用失败（含重试耗尽）。"""


def _format_body(resp: requests.Response) -> str:
    """响应体可能是 JSON 对象（400/429/503）也可能是纯字符串（401/403/404/504），截断后统一展示。"""
    text = (resp.text or "").strip().replace("\n", " ")
    if len(text) > 300:
        text = text[:300] + "…"
    return text or "<空响应体>"


_TOKENIZER = None


def _find_cached_tokenizer() -> Path | None:
    """在旧本地模型缓存里找现成的 tokenizer.json（bge-m3 快照），免去下载。"""
    candidates = []
    for rel in (
        "models_cache/models/BAAI--bge-m3/snapshots/*/tokenizer.json",
        "models_cache/models--BAAI--bge-m3/snapshots/*/tokenizer.json",
    ):
        candidates.extend(PROJECT_DIR.glob(rel))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _download_tokenizer_file(dest: Path) -> None:
    """从 ModelScope（CN 优先）/ HF 下载 tokenizer.json（约 9MB），流式落盘。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    failures = []
    for url in TOKENIZER_URLS:
        try:
            with requests.get(url, stream=True, timeout=(10, 120), allow_redirects=True) as resp:
                resp.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            if not part.read_bytes().lstrip().startswith(b"{"):
                # 来源返回了非 JSON 内容（如 HTML 错误页），视为失败换下一个源
                raise requests.RequestException("下载内容不是 tokenizer.json")
            part.replace(dest)
            logger.info(f"tokenizer.json 下载完成: {dest}")
            return
        except requests.RequestException as exc:
            failures.append(f"{url} → {type(exc).__name__}: {exc}")
    if part.exists():
        part.unlink()
    raise RuntimeError(
        "tokenizer.json 下载失败（以下来源均不可达）：\n  " + "\n  ".join(failures) + "\n"
        "请手动下载 BAAI/bge-m3 的 tokenizer.json 并放到：\n  "
        f"{dest}\n"
        "来源：https://modelscope.cn/models/BAAI/bge-m3/files 或 "
        "https://huggingface.co/BAAI/bge-m3/tree/main"
    )


def _get_tokenizer():
    """bge-m3 tokenizer 懒加载单例。

    与 SiliconFlow 服务端使用同款分词器，token 计数才与服务端一致，
    按 token 截断才有精确保证。获取顺序：规范化缓存路径 → 旧模型缓存复用
    （bge-m3 快照里自带 tokenizer.json）→ 下载（ModelScope 优先，HF 兜底）。
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        if not TOKENIZER_FILE.exists():
            cached = _find_cached_tokenizer()
            if cached is not None:
                logger.info(f"复用本地已有 tokenizer: {cached}")
                TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, TOKENIZER_FILE)
            else:
                logger.info(f"tokenizer 不存在，开始下载 → {TOKENIZER_FILE}")
                _download_tokenizer_file(TOKENIZER_FILE)
        try:
            from tokenizers import Tokenizer

            _TOKENIZER = Tokenizer.from_file(str(TOKENIZER_FILE))
        except Exception as exc:
            raise RuntimeError(
                f"tokenizer.json 加载失败（{TOKENIZER_FILE}）：{exc}"
            ) from exc
        logger.info(f"tokenizer 已加载: {TOKENIZER_FILE}")
    return _TOKENIZER


class _SiliconFlowClient:
    """共享基类：API Key 校验、Session 连接池、带指数退避的 POST。"""

    def __init__(self):
        api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "缺少 SILICONFLOW_API_KEY 环境变量，无法调用 SiliconFlow API"
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self.total_tokens = 0  # 累计消耗的 token 数，便于观察费用

    def _post_with_retry(self, path: str, payload: dict, name: str) -> dict:
        """POST 并返回 200 响应的 JSON；不可重试错误/重试耗尽抛 SiliconFlowAPIError。"""
        delay = RETRY_BASE_DELAY
        last_error = ""
        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                resp = self._session.post(
                    f"{API_BASE}{path}", json=payload, timeout=API_TIMEOUT
                )
            except requests.RequestException as exc:
                last_error = f"网络异常：{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise SiliconFlowAPIError(
                            f"SiliconFlow {name} 响应不是合法 JSON：{_format_body(resp)}"
                        ) from exc
                if resp.status_code not in RETRYABLE_STATUS:
                    raise SiliconFlowAPIError(
                        f"SiliconFlow {name} 请求失败（HTTP {resp.status_code}，不可重试）："
                        f"{_format_body(resp)}"
                    )
                last_error = f"HTTP {resp.status_code}：{_format_body(resp)}"

            sleep_for = min(delay, RETRY_MAX_DELAY) * (0.5 + random.random())
            logger.warning(
                f"{name} 请求失败（第 {attempt}/{API_MAX_RETRIES} 次尝试），"
                f"{sleep_for:.1f}s 后重试：{last_error}"
            )
            time.sleep(sleep_for)
            delay *= 2
        raise SiliconFlowAPIError(
            f"{name} 请求重试 {API_MAX_RETRIES} 次仍失败，最后错误：{last_error}"
        )


class SiliconFlowEmbedding(_SiliconFlowClient):
    def encode(self, texts: list[str]) -> list:
        """批量编码，返回 n 个 1024 维 L2 归一化向量（纯 Python list），顺序与输入一致。

        内部用 numpy 归一化；对外返回 list——调用方可把缺失标记安全地存成 None。
        """
        if not texts:
            return []
        texts, token_counts = self._sanitize(texts)
        vectors = np.concatenate(
            [self._encode_batch(batch) for batch in self._make_batches(texts, token_counts)],
            axis=0,
        )
        return vectors.tolist()

    @staticmethod
    def _sanitize(texts: list[str]) -> tuple[list[str], list[int]]:
        """API 拒绝空字符串与超限长文本：空白串替换为.，超限按 token 精确截断。

        返回 (净化后的文本, 每条的 token 数)——token 数供分批闸门使用。
        """
        tok = _get_tokenizer()
        fixed = []
        token_counts = []
        empty_count = 0
        trunc_count = 0
        worst = 0
        for text in texts:
            if not text.strip():
                empty_count += 1
                text = "."
                token_counts.append(1)
            else:
                ids = tok.encode(text, add_special_tokens=False).ids
                if len(ids) > EMBEDDING_MAX_TOKENS:
                    trunc_count += 1
                    worst = max(worst, len(ids))
                    text = tok.decode(ids[:EMBEDDING_MAX_TOKENS])
                    ids = ids[:EMBEDDING_MAX_TOKENS]
                token_counts.append(len(ids))
            fixed.append(text)
        if empty_count:
            logger.warning(
                f"{empty_count} 条空白文本已用.替代（API 拒绝空字符串）"
            )
        if trunc_count:
            logger.warning(
                f"{trunc_count} 条文本超过 {EMBEDDING_MAX_TOKENS} token（最长 {worst}），"
                "已按 token 截断后编码，可用 GODOT_RAG_EMBEDDING_MAX_TOKENS 调整"
            )
        return fixed, token_counts

    @staticmethod
    def _make_batches(texts: list[str], token_counts: list[int]) -> list[list[str]]:
        """双闸门分批：条数 ≤ API_BATCH_SIZE，批内总 token ≤ API_BATCH_TOKENS。

        条数是官方硬限制；token 闸门按服务端计费口径防御请求级总量限制
        （table:: 标记类文本字符少但 token 密，字符口径防不住）。
        单条超预算的文本独立成批（截断后 ≤ 8176 token，不会触发）。"""
        batches = []
        current: list[str] = []
        total_tokens = 0
        for text, n_tokens in zip(texts, token_counts):
            if current and (
                len(current) >= API_BATCH_SIZE or total_tokens + n_tokens > API_BATCH_TOKENS
            ):
                batches.append(current)
                current = []
                total_tokens = 0
            current.append(text)
            total_tokens += n_tokens
        if current:
            batches.append(current)
        return batches

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """编码一批（≤ API_BATCH_SIZE 条），带重试；失败语义见模块 docstring。"""
        body = self._post_with_retry(
            "/embeddings", {"model": EMBEDDING_MODEL, "input": texts, "options": {"flash_attention": False}}, "embeddings"
        )
        try:
            data = body["data"]
        except (KeyError, TypeError) as exc:
            raise SiliconFlowAPIError(
                f"embeddings 响应格式异常：{body}"
            ) from exc
        if len(data) != len(texts):
            raise SiliconFlowAPIError(f"返回向量数 {len(data)} != 输入数 {len(texts)}")
        data.sort(key=lambda item: item["index"])

        self.total_tokens += body.get("usage", {}).get("total_tokens", 0)
        vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
        # 客户端统一归一化，与原本地 encode(normalize_embeddings=True) 行为一致
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class SiliconFlowReranker(_SiliconFlowClient):
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """对 documents 打相关性分，返回与输入等长、按输入顺序对齐的分数列表。

        分批请求后按绝对分数合并（同一模型同一 query，分数跨批可比）。
        始终返回全部分数，排序与截断由调用方完成（与原本地 CrossEncoder 行为一致）。
        """
        if not documents:
            return []
        scores: list[float] = []
        for start in range(0, len(documents), RERANK_BATCH_SIZE):
            batch = documents[start:start + RERANK_BATCH_SIZE]
            scores.extend(self._rerank_batch(query, batch))
        return scores

    def _rerank_batch(self, query: str, documents: list[str]) -> list[float]:
        """重排一批（≤ RERANK_BATCH_SIZE 条），按 index 把分数对回输入顺序。"""
        payload = {
            "model": RERANKER_MODEL,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        body = self._post_with_retry("/rerank", payload, "rerank")
        try:
            results = body["results"]
            scores_by_index = {item["index"]: item["relevance_score"] for item in results}
        except (KeyError, TypeError) as exc:
            raise SiliconFlowAPIError(f"rerank 响应格式异常：{body}") from exc
        if len(scores_by_index) != len(documents):
            raise SiliconFlowAPIError(
                f"返回分数数 {len(scores_by_index)} != 输入文档数 {len(documents)}"
            )

        self.total_tokens += body.get("meta", {}).get("tokens", {}).get("input_tokens", 0)
        return [scores_by_index[i] for i in range(len(documents))]
