import json
import multiprocessing
import os
import shutil
import uuid
from collections import deque
from queue import Empty

# 必须在 torch 初始化 CUDA 分配器前设置：展开式 segment 消除碎片驻留，
# 否则 reserved 会推过物理显存并溢出，表现为越跑越慢（实测 13.5 GiB）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from loguru import logger
from pathlib import Path
from qdrant_client import QdrantClient, models

import numpy as np

from ..utils import PROJECT_DIR, QURANT_DATA_DIR, CHUNKS_PATH, get_model, format_duration

NAMESPACE = uuid.NAMESPACE_DNS

COLLECTION_NAME = "godot_docs"
VECTOR_DIM = 1024  # bge-m3 的输出维度

# 批的两道闸门（默认 "auto" = 由 GODOT_RAG_VRAM_BUDGET 换算；显式设置则覆盖）：
# ATTN_BUDGET：批 token 总量 T × 批内最长 token 数 L ≤ 预算（注意力项）
# TOKEN_BUDGET：批 token 总量 T ≤ 该值（线性激活项）
ATTN_BUDGET = os.environ.get("GODOT_RAG_ATTN_BUDGET", "auto").strip().lower()
TOKEN_BUDGET = os.environ.get("GODOT_RAG_TOKEN_BUDGET", "auto").strip().lower()
# 每 N 批 empty_cache 一次，压平缓存驻留抑制碎片；0 关闭
EMPTY_CACHE_EVERY = int(os.environ.get("GODOT_RAG_EMPTY_CACHE_EVERY", "256"))
# 写入子进程攒够这么多点才 upsert 一次
UPSERT_BATCH = int(os.environ.get("GODOT_RAG_UPSERT_BATCH", "512"))

# ---- 解析式预算（唯一模式，不跑探针；报错即退出，无重试/降批/动态调节）----
# 【系数仅适用于 bge-m3/fp32，换模型/精度后按实测回填】
# GODOT_RAG_VRAM_BUDGET：预计占用的显存，"80%"（相对总显存）/"9.5GiB"/纯字节
# 联合公式（两项同存于一个批，必须相加）：
#   批激活 ≈ ATTN_COST_COEF×(T×L) + TOKEN_COST_COEF×T；总占用 ≈ 权重(实测) + 批激活
#   注意力预算 = BUDGET_SHARE×激活余量÷260；token 上限 = (1−BUDGET_SHARE)×激活余量÷50k
#   两闸门分吃余量，联合最坏恰好 ≤ 余量（各自独占余量的写法实测会打满 12 GiB）
# 系数为本机探针实测拟合：注意力 28.3M token²→6.0GiB、33.5M→7.6GiB，取保守 260；
# 线性项 ≈4.3万 B/token，取 5 万。调大系数 = 更保守。
# 下限：注意力预算须 > 4096²≈16.8M（单条满长 chunk，~4.5 GiB 总占用），
# 即预算百分比别低于 ~57%，否则满长 chunk 批必然 OOM。跑挂了调低预算重跑。
VRAM_BUDGET = os.environ.get("GODOT_RAG_VRAM_BUDGET", "80%").strip()
ATTN_COST_COEF = float(os.environ.get("GODOT_RAG_ATTN_COST_COEF", "260"))
TOKEN_COST_COEF = float(os.environ.get("GODOT_RAG_TOKEN_COST_COEF", "50000"))
# 激活余量分给注意力项的比例；调大 = 长批更足、短批更保守
BUDGET_SHARE = float(os.environ.get("GODOT_RAG_BUDGET_SHARE", "0.7"))

CURRENT_PATH = Path(__file__)
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def _to_point(chunk: dict, vector) -> models.PointStruct:
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    return models.PointStruct(
        id=chunk_id_to_point_id(chunk["chunk_id"]),
        vector=vector,
        payload={"chunk_id": chunk["chunk_id"]},
    )


def _count_token_lens(chunks: list[dict], model) -> list[int]:
    """用 tokenizer 统计每条 chunk 的 token 数（仅计数不截断）。

    截断上限只由 get_model 的 model.max_seq_length（GODOT_RAG_MAX_SEQ_LENGTH）
    决定。闸门使用截断后的计费长度：超长序列编码时被截断，峰值显存由截断后
    长度决定，按原始长度计费会让预算换算失效。
    """
    from transformers.utils import logging as hf_logging

    prev_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        token_ids = model.tokenizer([c["content"] for c in chunks], add_special_tokens=False)["input_ids"]
    finally:
        hf_logging.set_verbosity(prev_verbosity)
    token_lens = [len(ids) for ids in token_ids]

    truncated = sum(1 for n in token_lens if n > model.max_seq_length)
    if truncated:
        logger.warning(
            f"{truncated} 个 chunk 超过 max_seq_length={model.max_seq_length}，"
            "编码时尾部内容将被截断（可调 GODOT_RAG_MAX_SEQ_LENGTH）"
        )
    max_seq = max(model.max_seq_length, 1)
    return [min(n, max_seq) for n in token_lens]


def _take_batch(items: deque, token_budget: int, attn_budget: int) -> list[tuple[dict, int]]:
    """从队头贪心取一批：T×L ≤ attn_budget 且 T ≤ token_budget；
    单条 chunk 自身超限也保留一条，保证循环推进（该批 OOM 会直接抛错）。"""
    batch: list[tuple[dict, int]] = []
    cur_tokens = 0
    cur_max_len = 0
    while items:
        chunk, n_tokens = items[0]
        if batch:
            new_max = max(cur_max_len, n_tokens)
            if cur_tokens + n_tokens > token_budget:
                break
            if (cur_tokens + n_tokens) * new_max > attn_budget:
                break
        batch.append(items.popleft())
        cur_tokens += n_tokens
        cur_max_len = max(cur_max_len, n_tokens)
    return batch


def make_batches(chunk_items: list[tuple[dict, int]],
                 token_budget: int, attn_budget: int) -> list[list[tuple[dict, int]]]:
    """贪心分批；调用前需按内容长度排序，保证批内长度均匀。"""
    items = deque(chunk_items)
    batches: list[list[tuple[dict, int]]] = []
    while items:
        batches.append(_take_batch(items, token_budget, attn_budget))
    return batches


def _parse_vram_budget(spec: str, total_bytes: int) -> int:
    """解析预算："80%" 相对总显存；"9.5GiB"/"9800MiB"/"512MB"/纯字节；
    auto/空 → 报错（探针标定已移除）。"""
    s = spec.strip().lower()
    if s in ("", "auto"):
        raise ValueError(
            "GODOT_RAG_VRAM_BUDGET 设为 auto/空 已不支持（探针标定已移除），"
            '请显式设置预算，如 GODOT_RAG_VRAM_BUDGET="80%" 或 "9.5GiB"'
        )
    if s.endswith("%"):
        frac = float(s[:-1]) / 100
        if not 0 < frac <= 1.5:
            raise ValueError(f"GODOT_RAG_VRAM_BUDGET 百分比非法：{spec!r}")
        return int(total_bytes * frac)
    if s.endswith("b"):
        s = s[:-1]  # gib→gi、mb→m
    for suf, mult in (("ki", 2**10), ("mi", 2**20), ("gi", 2**30), ("ti", 2**40),
                      ("k", 10**3), ("m", 10**6), ("g", 10**9), ("t", 10**12)):
        if s.endswith(suf):
            return int(float(s[:-len(suf)].strip()) * mult)
    return int(float(s))


def resolve_budgets(model, fixed_attn: int | None, fixed_tok: int | None) -> tuple[int, int]:
    """预算解析入口：显式 ATTN/TOKEN_BUDGET 项优先，其余由 VRAM_BUDGET
    经联合公式换算（不跑探针）。CPU/无 CUDA 用 floor 固定值。"""
    import torch

    if fixed_attn is not None and fixed_tok is not None:
        logger.warning(
            f"两个预算都显式固定（attn={fixed_attn}, token={fixed_tok}），跳过预算计算"
        )
        return fixed_attn, fixed_tok

    if str(model.device) == "cpu" or not torch.cuda.is_available():
        logger.warning("无可用 CUDA：两个预算使用 floor/固定值")
        return (fixed_attn if fixed_attn is not None else 2 * 1024 * 1024,
                fixed_tok if fixed_tok is not None else 4096)

    budget_bytes = _parse_vram_budget(
        VRAM_BUDGET, torch.cuda.get_device_properties(0).total_memory
    )
    weights = torch.cuda.memory_allocated()
    allowance = max(0, budget_bytes - weights)
    if not 0 < BUDGET_SHARE < 1:
        raise ValueError(f"GODOT_RAG_BUDGET_SHARE 必须在 (0,1) 区间：{BUDGET_SHARE}")
    # 拆分后联合最坏情形激活 = share×余量 + (1−share)×余量 = 余量，构造上不越线
    attn = max(1024, int(allowance * BUDGET_SHARE / ATTN_COST_COEF))
    token = max(1024, int(allowance * (1 - BUDGET_SHARE) / TOKEN_COST_COEF))
    if fixed_attn is not None:
        attn = fixed_attn
    if fixed_tok is not None:
        token = fixed_tok
    logger.info(
        f"解析式预算（未运行探针，公式仅适用于 bge-m3/fp32）：显存预算 "
        f"{budget_bytes / 2**30:.2f} GiB = 权重 {weights / 2**30:.2f} + 激活 "
        f"{allowance / 2**30:.2f} GiB（按 {BUDGET_SHARE:.0%}/{1 - BUDGET_SHARE:.0%} "
        f"分配给注意力/线性项）→ 注意力预算 {attn}（T×L），token 上限 {token}/批"
    )
    return attn, token


def _write_qdrant_worker(points_q: multiprocessing.Queue, result_q: multiprocessing.Queue):
    """写入子进程：重建 collection，消费 points 队列持续 upsert，哨兵后写完
    尾批并校验总数；异常带非零 exitcode 退出。"""
    try:
        logger.add(LOG_PATH, encoding="utf-8")
        # 物理清空存储目录再重建：delete_collection + create_collection
        # 实测不清空旧 point，旧点残留会让校验失败
        if QURANT_DATA_DIR.exists():
            shutil.rmtree(QURANT_DATA_DIR)
        client = QdrantClient(path=str(QURANT_DATA_DIR))
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        if client.count(COLLECTION_NAME, exact=True).count != 0:
            client.close()
            raise RuntimeError("新 collection 非空，存储目录清理失败")

        written = 0
        buffer: list[models.PointStruct] = []
        while True:
            batch = points_q.get()
            if batch is None:
                break

            # 攒够 UPSERT_BATCH 才写一次
            buffer.extend(batch)
            while len(buffer) >= UPSERT_BATCH:
                client.upsert(COLLECTION_NAME, buffer[:UPSERT_BATCH])
                written += UPSERT_BATCH
                logger.info(f"已写入 qdrant {written}")
                buffer = buffer[UPSERT_BATCH:]

        if buffer:
            client.upsert(COLLECTION_NAME, buffer)
            written += len(buffer)
            logger.info(f"已写入 qdrant {written}")

        # 校验：point 数必须恰好等于写入数，多/少都说明存储不可信
        total = client.count(COLLECTION_NAME, exact=True).count
        client.close()
        if total != written:
            raise RuntimeError(f"写入后 point 数 {total} != 本次写入 {written}，存储状态异常")
        result_q.put(written)
        logger.info(f"写入进程结束，总共写入 {written} 个 chunk")
    except BaseException as exc:
        logger.exception(f"qdrant 写入进程异常：{type(exc).__name__}: {exc}")
        raise


def build(chunks_path: Path = CHUNKS_PATH):
    import time

    started = time.perf_counter()
    logger.add(LOG_PATH, encoding="utf-8")
    logger.info("读取 chunks")
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    if not chunks:
        raise ValueError(f"chunks 文件为空：{chunks_path}")
    # 长度相近的排在一起，配合 make_batches 保证批内长度均匀
    chunks.sort(key=lambda c: len(c["content"]))
    logger.info(f"chunks: {len(chunks)}")

    model = get_model()
    logger.info(f"开始生成向量（device={model.device}）")

    token_lens = _count_token_lens(chunks, model)
    chunk_items = list(zip(chunks, token_lens))

    fixed_attn = None if ATTN_BUDGET in ("", "auto") else int(ATTN_BUDGET)
    fixed_tok = None if TOKEN_BUDGET in ("", "auto") else int(TOKEN_BUDGET)
    attn_budget, token_budget = resolve_budgets(model, fixed_attn, fixed_tok)

    logger.info(
        f"初始预估 {len(make_batches(chunk_items, token_budget, attn_budget))} 批"
        f"（注意力预算 {attn_budget}，token 上限 {token_budget}/批）"
    )

    # 写入放子进程，与编码重叠进行
    points_q: multiprocessing.Queue = multiprocessing.Queue()
    result_q: multiprocessing.Queue = multiprocessing.Queue()
    writer = multiprocessing.Process(target=_write_qdrant_worker, args=(points_q, result_q))
    writer.start()

    encoded = 0
    import torch

    pending: deque = deque(zip(chunks, token_lens))
    batch_no = 0
    try:
        while pending:
            # 写入进程已死就尽早中止；cancel_join_thread 防止 join
            # 写不出去的队列永久挂起（实测复现）
            if not writer.is_alive():
                points_q.cancel_join_thread()
                raise RuntimeError(
                    f"qdrant 写入进程提前退出（exitcode={writer.exitcode}），中止编码，详见日志"
                )
            batch_started = time.perf_counter()
            batch_items = _take_batch(pending, token_budget, attn_budget)
            batch_no += 1
            # 报错即退出：OOM/驱动异常直接抛出，调低 GODOT_RAG_VRAM_BUDGET 重跑
            vectors = model.encode(
                [c["content"] for c, _ in batch_items],
                batch_size=len(batch_items),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            encoded += len(vectors)
            points_batch = [_to_point(c, v) for (c, _), v in zip(batch_items, vectors)]
            if points_batch:
                points_q.put(points_batch)

            duration = time.perf_counter() - batch_started
            logger.info(f"批次 {batch_no} 耗时 {duration:.2f}s，已生成 {encoded}/{len(chunks)} 个向量")

            if EMPTY_CACHE_EVERY > 0 and batch_no % EMPTY_CACHE_EVERY == 0:
                torch.cuda.empty_cache()
    finally:
        # 无论成败都发哨兵并等写入进程收尾；put 失败兜底，
        # 避免次生异常掩盖编码阶段的原始错误
        try:
            points_q.put(None)
        except (ValueError, OSError):
            logger.warning("points_q put 失败（罕见），跳过哨兵发送")
        writer.join()
        if writer.exitcode not in (None, 0):
            points_q.cancel_join_thread()

    if writer.exitcode != 0:
        raise RuntimeError(f"qdrant 写入进程异常退出（exitcode={writer.exitcode}），详见日志")

    # 超时兜底：把"理论上不可能的挂死"变成明确报错
    try:
        written = result_q.get(timeout=120)
    except Empty as exc:
        raise RuntimeError(
            "写入进程退出后 120s 内未回传写入计数，result_q 状态异常"
        ) from exc
    logger.info(f"总共写入 {written} 个 chunk 到 Qdrant（共生成 {encoded} 个向量）")
    elapsed = time.perf_counter() - started
    logger.info(f"qdrant 索引构建完成，总耗时 {format_duration(elapsed)}")
    return elapsed


if __name__ == "__main__":
    build(PROJECT_DIR / "data/chunks.jsonl")
