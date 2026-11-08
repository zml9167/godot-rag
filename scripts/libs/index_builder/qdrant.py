import json
import multiprocessing
import os
import shutil
import uuid
from collections import deque
from queue import Empty

# 必须在 torch 首次初始化 CUDA 分配器之前设置（utils 里 torch 是函数内导入，
# 此处一定是它之前）。展开式 segment 让分配器按小块映射/卸载显存，
# 消除“多种批形状交替留下的 segment 在 empty_cache 后仍无法释放”的碎片驻留——
# 该驻留会把 reserved 水位推过物理显存，再由 WDDM 溢出到系统内存，
# 表现为越跑越慢（实测 reserved 可达 13.5 GiB > 物理显存）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from loguru import logger
from pathlib import Path
from qdrant_client import QdrantClient, models

import numpy as np

from ..utils import PROJECT_DIR, QURANT_DATA_DIR, CHUNKS_PATH, get_model, format_duration

NAMESPACE = uuid.NAMESPACE_DNS

COLLECTION_NAME = "godot_docs"
VECTOR_DIM = 1024  # bge-m3 的输出维度，请根据实际模型确认

# 批的两道闸门，对应显存的两个增长维度：
# 1) 注意力预算 ATTN_BUDGET（二次项）：批满足 批token总量 T × 批内最长 token 数 L ≤ 预算；
#    注意力显存与计算量 ∝ T×L，量纲 token²，"auto" 表示启动探针标定。
# 2) token 预算 TOKEN_BUDGET（一次项）：批 token 总量 ≤ 该值，封顶线性激活项（∝ T）。
#    两个维度相互独立，短文本批只受 2) 约束、长文本批只受 1) 约束时的极端情况
#    都会被各自的探针测到，标定缺一不可。
ATTN_BUDGET = os.environ.get("GODOT_RAG_ATTN_BUDGET", "auto").strip().lower()
TOKEN_BUDGET = os.environ.get("GODOT_RAG_TOKEN_BUDGET", "auto").strip().lower()
# 每隔这么多批调用一次 empty_cache，把缓存分配器的驻留水位压平，抑制碎片。
# 实测中碎片导致过 OOM（free=0 但申请大块失败），因此默认开启。
# 设 0 可关闭（牺牲显存曲线换取略少的 cudaMalloc 往返）。
EMPTY_CACHE_EVERY = int(os.environ.get("GODOT_RAG_EMPTY_CACHE_EVERY", "256"))
# 标定目标线：探针通过后仍应保留这么多比例的显存（给碎片和系统留余量）。
VRAM_MARGIN = float(os.environ.get("GODOT_RAG_VRAM_MARGIN", "0.20"))
# 越线后的二分精标轮数：翻倍梯子粒度是 x2，7 轮二分可把最终配置
# 解析到原跨度的 ~1.5%。每轮都是真实 encode 探针，轮数越多标定越慢。
CALIBRATE_REFINE_ROUNDS = int(os.environ.get("GODOT_RAG_REFINE_ROUNDS", "7"))
# 写入子进程内部攒批：攒够这么多点才执行一次 upsert，
# 减少 qdrant 本地存储的小段合并开销
UPSERT_BATCH = int(os.environ.get("GODOT_RAG_UPSERT_BATCH", "512"))

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


def encode_batch(model, texts: list[str]) -> tuple[np.ndarray, bool]:
    """编码一批文本；CUDA OOM 时释放缓存并把内部批减半重试。

    model.encode 的 batch_size 决定单次 forward 的条数，减半它即可
    线性降低显存峰值（批内长度相近时）。
    返回 (向量, 是否发生过降批)；发生过时调用方可自适应调低后续批预算。
    """
    import torch

    batch_size = len(texts)
    while True:
        try:
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            logger.warning(f"CUDA OOM，batch_size 降为 {batch_size} 后重试")
            continue
        break
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return vectors, batch_size < len(texts)


def _count_token_lens(chunks: list[dict], model) -> list[int]:
    """用模型自带 tokenizer 精确统计每条 chunk 的 token 数（不截断，仅计数）。

    序列截断的上限只由 utils.get_model 里的 model.max_seq_length
    （GODOT_RAG_MAX_SEQ_LENGTH）决定，这里只读取，不做第二处限制。
    """
    from transformers.utils import logging as hf_logging

    # 这里仅统计 token 数，故意不截断；超长的逐条警告改为汇总提示
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
    # 闸门与探针使用“截断后的实际计费长度”：编码时超长序列被截到 max_seq_length，
    # 真实峰值显存与耗时由截断后的长度决定（否则一个几十万 token 的巨型 chunk
    # 会以原始长度参与计量，霸占所有探针的 argmax，让标定梯子失效）。
    max_seq = max(model.max_seq_length, 1)
    return [min(n, max_seq) for n in token_lens]


def _take_batch(items: deque, token_budget: int, attn_budget: int) -> list[tuple[dict, int]]:
    """从 items 头部贪心取出“显存占用 ≤ 预算”的一批（元素为 [chunk, token数] 对）。

    两种超限都会截断当前批：
    - (批token总量 T × 批内最长 token 数 L) > attn_budget：注意力项（二次）闸门；
    - 批token总量 T > token_budget：线性激活项（一次）闸门。
    单条 chunk 自身就超限时至少保留一条（由 encode_batch 内部降批兜底），
    保证循环能推进、数据不遗漏。
    """
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
    """按显存占用做贪心分批，批内条数不设上限。

    chunk_items 为 [chunk, token数] 对，且调用前已按内容长度排序，
    因此相邻条目长度相近、每批的实际显存占用大致受闸门约束。
    """
    items = deque(chunk_items)
    batches: list[list[tuple[dict, int]]] = []
    while items:
        batches.append(_take_batch(items, token_budget, attn_budget))
    return batches


def _batch_attn_cost(batch: list[tuple[dict, int]]) -> int:
    """本批注意力开销代理：条数 n × 批内最长 token 数²（量纲 token²）。"""
    if not batch:
        return 0
    return len(batch) * max(n for _, n in batch) ** 2


def _batch_token_total(batch: list[tuple[dict, int]]) -> int:
    return sum(n for _, n in batch)


def _probe_budgets(model, chunk_items: list[tuple[dict, int]], attn: int, token: int):
    """在给定预算对下分批并实测两类探针。返回 (是否失败, 峰值占用列表)。

    度量不用驱动层 torch.cuda.mem_get_info：本机 ROCm/Windows 下该读数会
    失真（权重+探针编码后仍报 free==total≈12GiB，而驱动层实际已 free:0），
    据此判断会漏掉所有真实现象。改用 torch 分配器自身的记账：
    每个探针 encode 前重置峰值统计，encode 后取 max_memory_allocated 与
    memory_reserved 的较大值——前者是真实峰值分配量，后者含缓存驻留
    （碎片方向），两者取 max 对驱动层真实足迹是保守代理。
    """
    import torch

    batches = make_batches(chunk_items, token, attn)
    attn_probe = max(batches, key=_batch_attn_cost)
    tok_probe = max(batches, key=_batch_token_total)
    candidates = [attn_probe] if attn_probe is tok_probe else [attn_probe, tok_probe]

    peaks = []
    for probe in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.encode(
                [c["content"] for c, _ in probe],
                batch_size=len(probe),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return True, peaks
        peak = max(torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())
        peaks.append(peak)
    return False, peaks


def _shrink_between(lo: tuple[int, int], hi: tuple[int, int]) -> tuple[int, int]:
    return (max(1, (lo[0] + hi[0]) // 2), max(1, (lo[1] + hi[1]) // 2))


def calibrate_budgets(model, chunk_items: list[tuple[dict, int]],
                      fixed_attn: int | None = None, fixed_tok: int | None = None,
                      target_free_frac: float | None = None) -> tuple[int, int]:
    """联合标定 (注意力预算, token 预算) 两个闸门。

    梯子每档同时向上探索两个预算（翻倍），每档用真实数据按本档闸门分出
    全部批，从中挑两类探针独立实测：
    - 注意力探针：_batch_attn_cost 最大的批（二次维度，一般是长文本批）；
    - token 探针：_batch_token_total 最大的批（一次维度，一般是短文本批）。
    每个探针在干净缓存下单独 encode 并用 torch 分配器记账实测峰值占用
    （驱动层 mem_get_info 在本机读数失真，不可作依据），判决标准是
    峰值占用 ≤ 峰值上限 = (总显存 − 外部占用) × (1-VRAM_MARGIN)。

    翻倍粒度是 x2，最后一次越过峰顶上限的档位可能显著高于目标（如从 56% 跳到 100%），
    因此在“上一档 ↔ 本档”之间做 CALIBRATE_REFINE_ROUNDS 轮二分精标，
    逼近峰值上限对应的占用。

    停止条件：任一探针 OOM、峰值占用越过峰顶上限（二分后）、探针形状与上一档
    完全相同、或两个预算到达显式上限。返回 (attn_budget, token_budget)。
    """
    import torch

    # 目标线：默认取 VRAM_MARGIN（用户可调），标定后仍保留该比例显存给碎片与系统
    target_free_frac = VRAM_MARGIN if target_free_frac is None else target_free_frac

    def fallback() -> tuple[int, int]:
        return (fixed_attn if fixed_attn is not None else 2 * 1024 * 1024,
                fixed_tok if fixed_tok is not None else 4096)

    if str(model.device) == "cpu":
        logger.warning("device=cpu，无法做显存探针，两个预算使用 floor/固定值")
        return fallback()
    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，两个预算使用 floor/固定值")
        return fallback()

    # 总显存取设备属性（本机 mem_get_info 连总量都会报错值，见 _probe_budgets 注释）
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    # mem_get_info 只用于估计外部占用（桌面/其它进程），读数失真时按 0 处理
    try:
        driver_free, _ = torch.cuda.mem_get_info()
        external_used = max(0, total_bytes - driver_free - torch.cuda.memory_allocated())
        if external_used >= total_bytes // 2:
            external_used = 0
    except Exception:
        external_used = 0
    # 探针峰值上限 = (总显存 − 外部占用) × (1-VRAM_MARGIN)。
    # 峰值 gauge 含模型权重，用绝对值比较；mem_get_info 失真时外部占用按 0 处理，
    # 差额由 VRAM_MARGIN 余量吸收。这样探针永远不会再把驱动层打到 free:0，
    # 桌面卡顿和分配器濒死重试（HIPCachingAllocator 警告）都会消失。
    peak_cap = int((total_bytes - external_used) * (1 - target_free_frac))

    attn = fixed_attn if fixed_attn is not None else 2 * 1024 * 1024
    token = fixed_tok if fixed_tok is not None else 4096
    attn_max = fixed_attn if fixed_attn is not None else 2 ** 30
    token_max = fixed_tok if fixed_tok is not None else 2 ** 22

    def log_pass(attn_v: int, token_v: int, frees: list[int]):
        for free_after in frees:
            logger.info(
                f"探针通过（attn={attn_v}, token={token_v}）"
                f"，峰值占用 {free_after / 2**30:.2f} / 上限 {peak_cap / 2**30:.2f} GiB"
                f"（总 {total_bytes / 2**30:.2f} GiB，保留 {target_free_frac:.0%}）"
            )

    best: tuple[int, int] | None = None
    last_obs: list | None = None
    while True:
        failed, peaks = _probe_budgets(model, chunk_items, attn, token)
        if failed:
            logger.info(f"预算 (attn={attn}, token={token}) 探针 OOM，回退到 {best if best else '(floor 起步值)'}")
            break
        worst_peak = max(peaks)
        log_pass(attn, token, peaks)

        if worst_peak >= peak_cap:
            if best is None:
                # floor 起步档就已超上限：保留保守起步值，交给运行期降批兜底
                best = (attn, token)
                break
            # 越过峰值上限：与上一通过档之间二分精标，逼近上限的最大配置
            lo, hi = best, (attn, token)
            for _ in range(CALIBRATE_REFINE_ROUNDS):
                mid = _shrink_between(lo, hi)
                if mid == lo or mid == hi:
                    break
                failed, mid_peaks = _probe_budgets(model, chunk_items, *mid)
                if failed or max(mid_peaks) >= peak_cap:
                    hi = mid
                    if failed:
                        torch.cuda.empty_cache()
                else:
                    lo = mid
                    log_pass(*mid, mid_peaks)
            return lo

        best = (attn, token)
        obs = [(len(make_batches(chunk_items, token, attn)), peaks)]
        if obs == last_obs:
            break
        last_obs = obs
        if attn >= attn_max and token >= token_max:
            break
        attn = min(attn_max, attn * 2)
        token = min(token_max, token * 2)

    return best if best else fallback()


def _write_qdrant_worker(points_q: multiprocessing.Queue, result_q: multiprocessing.Queue):
    """子进程：初始化 collection，把队列里的 points 批次持续写入，收到哨兵后退出。

    写入放子进程可以让它和主进程的下一批编码重叠进行；
    异常时进程带非零 exitcode 退出，由主进程 join 后检查并抛错。
    """
    try:
        logger.add(LOG_PATH, encoding="utf-8")
        # 物理清空整个存储目录再重建：qdrant-client 1.20.0 本地模式下
        # delete_collection + create_collection 实测不清空旧 point（已隔离复现，
        # 旧点残留会让校验失败、检索时 CHUNK_MAP KeyError），不再信任它
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

            # 攒批在写入端做：不够 UPSERT_BATCH 就继续等下一批
            buffer.extend(batch)
            while len(buffer) >= UPSERT_BATCH:
                client.upsert(COLLECTION_NAME, buffer[:UPSERT_BATCH])
                written += UPSERT_BATCH
                logger.info(f"已写入 qdrant {written}")
                buffer = buffer[UPSERT_BATCH:]

        # 哨兵之后剩余的尾批也要写完
        if buffer:
            client.upsert(COLLECTION_NAME, buffer)
            written += len(buffer)
            logger.info(f"已写入 qdrant {written}")

        # 收尾校验：collection 里的 point 数必须恰好等于本次写入数，
        # 多了是残留、少了是丢点，都说明存储状态不可信
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
    if fixed_attn is None and fixed_tok is None:
        attn_budget, token_budget = calibrate_budgets(model, chunk_items)
        logger.info(f"标定完成：注意力预算 {attn_budget}，token 上限 {token_budget}/批")
    elif fixed_attn is not None and fixed_tok is not None:
        attn_budget, token_budget = fixed_attn, fixed_tok
        logger.warning(f"两个预算都显式固定（attn={attn_budget}, token={token_budget}），跳过标定")
    else:
        attn_budget, token_budget = calibrate_budgets(
            model, chunk_items, fixed_attn=fixed_attn, fixed_tok=fixed_tok
        )
        logger.info(f"标定完成（部分固定）：注意力预算 {attn_budget}，token 上限 {token_budget}/批")

    logger.info(
        f"初始预估 {len(make_batches(chunk_items, token_budget, attn_budget))} 批"
        f"（注意力预算 {attn_budget}，token 上限 {token_budget}/批）"
    )

    # 写入放到独立的子进程，与下一批的 GPU 编码重叠进行
    points_q: multiprocessing.Queue = multiprocessing.Queue()
    result_q: multiprocessing.Queue = multiprocessing.Queue()
    writer = multiprocessing.Process(target=_write_qdrant_worker, args=(points_q, result_q))
    writer.start()

    encoded = 0
    import torch

    # 待编码的 chunk 与 token 数一一对应，批是边取边组的：
    # 预算自适应调低后立即作用于剩余数据，无需整体重排
    pending: deque = deque(zip(chunks, token_lens))
    batch_no = 0
    try:
        while pending:
            # 写入进程已死就尽早中止：继续编码只会把数据喂给无人消费的队列，
            # 内存白涨、GPU 白烧，错误却要到全部编码完才浮出
            if not writer.is_alive():
                # 缓冲数据已永远无法送达，放弃 feeder 线程的退出冲洗：
                # 否则解释器退出时 join 一个写不出去的队列会永久挂起（实测复现）
                points_q.cancel_join_thread()
                raise RuntimeError(
                    f"qdrant 写入进程提前退出（exitcode={writer.exitcode}），中止编码，详见日志"
                )
            batch_started = time.perf_counter()
            batch_items = _take_batch(pending, token_budget, attn_budget)
            batch_no += 1
            vectors, had_oom = encode_batch(model, [c["content"] for c, _ in batch_items])
            encoded += len(vectors)
            # 主进程只管编码，攒批和 upsert 都交给写入进程
            points_batch = [_to_point(c, v) for (c, _), v in zip(batch_items, vectors)]
            if points_batch:
                points_q.put(points_batch)

            if had_oom:
                # 运行期 OOM：两个闸门是独立维度（见文件顶部常量注释），
                # 各自按比例收缩，而不是像 v2 那样由注意力预算推导 token 上限。
                shrunk_attn = max(1024 * 512, int(attn_budget * 0.75))
                shrunk_tok = max(1024, int(token_budget * 0.75))
                if shrunk_attn < attn_budget:
                    attn_budget = shrunk_attn
                    logger.warning(f"运行期遇到 OOM，注意力预算降至 {attn_budget}")
                if shrunk_tok < token_budget:
                    token_budget = shrunk_tok
                    logger.warning(f"运行期遇到 OOM，token 预算降至 {token_budget}/批")

            duration = time.perf_counter() - batch_started
            logger.info(f"批次 {batch_no} 耗时 {duration:.2f}s，已生成 {encoded}/{len(chunks)} 个向量")

            # 定期归还缓存显存，压平显存曲线（见 EMPTY_CACHE_EVERY 说明）
            if EMPTY_CACHE_EVERY > 0 and batch_no % EMPTY_CACHE_EVERY == 0:
                torch.cuda.empty_cache()
    finally:
        # 无论编码是否出错都要发哨兵并等待写入进程收尾。
        # （实测：写入进程死后 put 本身几乎不会抛错——数据只是被 feeder
        #   线程静默丢弃；这里兜底以防万一，避免次生异常掩盖编码阶段的原始错误）
        try:
            points_q.put(None)
        except (ValueError, OSError):
            logger.warning("points_q put 失败（罕见），跳过哨兵发送")
        writer.join()
        if writer.exitcode not in (None, 0):
            # writer 异常收场时同样放弃冲洗（死掉的消费者读不到任何数据）
            points_q.cancel_join_thread()

    if writer.exitcode != 0:
        raise RuntimeError(f"qdrant 写入进程异常退出（exitcode={writer.exitcode}），详见日志")

    # exitcode==0 已保证 worker 是在 result_q.put(written) 成功后才退出的，
    # 正常情况下 get() 立即返回；超时兜底只是把"理论上不可能的挂死"
    # 变成明确报错，而非永久阻塞
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


def validate_calibration(chunks_path: Path = CHUNKS_PATH):
    """两阶段验证标定，不启动完整 build：
    阶段一：纯逻辑不变量（不加载模型，token 数用字符数近似）。
    阶段二：真实 GPU 标定（载模型 + 真实 token 计数 + 探针 encode），
            输出最终 (attn, token) 并实测一批运行批编码。
    """
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    chunks.sort(key=lambda c: len(c["content"]))
    logger.info(f"chunks: {len(chunks)}")

    # ---- 阶段一：纯逻辑 ----
    approx_items = [(c, len(c["content"])) for c in chunks]
    combos = [
        (4096, 2 * 1024 * 1024),
        (16384, 32 * 1024 * 1024),
        (65536, 128 * 1024 * 1024),
        (512, 512 * 1024),
        (10 ** 9, 10 ** 12),  # 无约束：检验单条超长 chunk 也能成批推进
    ]
    all_ok = True
    for t_cap, a_budget in combos:
        batches = make_batches(approx_items, t_cap, a_budget)
        seen = sum(len(b) for b in batches)
        inv_ok = True
        for batch in batches:
            total = _batch_token_total(batch)
            longest = max(n for _, n in batch)
            if len(batch) > 1 and (total > t_cap or total * longest > a_budget):
                inv_ok = False
        if seen != len(approx_items):
            inv_ok = False
        if not inv_ok:
            all_ok = False
            logger.error(
                f"不变量失败：token_cap={t_cap}, attn={a_budget}, "
                f"覆盖 {seen}/{len(approx_items)}"
            )
            continue
        attn_probe = max(batches, key=_batch_attn_cost)
        tok_probe = max(batches, key=_batch_token_total)
        logger.info(
            f"组合通过：token_cap={t_cap}, attn={a_budget} → {len(batches)} 批 | "
            f"注意力探针 {len(attn_probe)} 条(max_token={max(n for _, n in attn_probe)}) | "
            f"token 探针 {len(tok_probe)} 条(T={_batch_token_total(tok_probe)})"
        )
    if not all_ok:
        raise RuntimeError("阶段一纯逻辑验证失败，请检查")

    # ---- 阶段二：真实 GPU 标定 ----
    import time
    import torch

    model = get_model()
    token_lens = _count_token_lens(chunks, model)
    chunk_items = list(zip(chunks, token_lens))
    fixed_attn = None if ATTN_BUDGET in ("", "auto") else int(ATTN_BUDGET)
    fixed_tok = None if TOKEN_BUDGET in ("", "auto") else int(TOKEN_BUDGET)
    attn_budget, token_budget = calibrate_budgets(
        model, chunk_items, fixed_attn=fixed_attn, fixed_tok=fixed_tok
    )

    # CPU 模式没有显存可探针：calibrate_budgets 返回的是 floor 兜底值，
    # 后面的 torch.cuda.* 统计调用在无 CUDA 环境会抛底层异常，提前收工
    if str(model.device) == "cpu" or not torch.cuda.is_available():
        logger.info(
            f"标定结果（CPU 模式，跳过显存探针）：attn={attn_budget}，token 上限={token_budget}/批"
        )
        return

    logger.info(f"标定结果：attn={attn_budget}，token 上限={token_budget}/批")

    # 用标定结果真实编码一批运行批，确认可跑
    first = _take_batch(deque(chunk_items), token_budget, attn_budget)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    vectors, had_oom = encode_batch(model, [c["content"] for c, _ in first])
    duration = time.perf_counter() - started
    peak = max(torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    logger.info(
        f"试编码通过：{len(first)} 条（max_token={max(n for _, n in first)}）"
        f"耗时 {duration:.2f}s，OOM 降批={had_oom}，"
        f"峰值占用 {peak / 2**30:.2f} / 总显存 {total_bytes / 2**30:.2f} GiB"
    )


if __name__ == "__main__":
    import sys

    chunks_path = PROJECT_DIR / "data/chunks.jsonl"
    if "--validate-calibration" in sys.argv:
        validate_calibration(chunks_path)
    else:
        build(chunks_path)
