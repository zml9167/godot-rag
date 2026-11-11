import json
import multiprocessing
import os
import shutil
import uuid
from queue import Empty

from loguru import logger
from pathlib import Path
from qdrant_client import QdrantClient, models

import numpy as np

from ..siliconflow import API_BATCH_SIZE, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from ..utils import PROJECT_DIR, QURANT_DATA_DIR, CHUNKS_PATH, format_duration, get_embedding_client

NAMESPACE = uuid.NAMESPACE_DNS

COLLECTION_NAME = "godot_docs"
VECTOR_DIM = EMBEDDING_DIMENSIONS  # bge-m3 固定 1024 维

# 写入子进程攒够这么多点才 upsert 一次
UPSERT_BATCH = int(os.environ.get("GODOT_RAG_UPSERT_BATCH", "512"))

CURRENT_PATH = Path(__file__)
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.parent.name}_{CURRENT_PATH.stem}.log"


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def _to_point(chunk: dict, vectors) -> models.PointStruct:
    return models.PointStruct(
        id=chunk_id_to_point_id(chunk["chunk_id"]),
        vector= {"content": vectors[1]} if vectors[0] is None else {"title": vectors[0], "content": vectors[1]},
        payload={"chunk_id": chunk["chunk_id"]},
    )


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
            vectors_config = {
                "title": models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
                "content": models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
            }
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
    logger.info(f"chunks: {len(chunks)}")

    client = get_embedding_client()
    logger.info(
        f"开始通过 SiliconFlow API 生成向量（model={EMBEDDING_MODEL}，"
        f"批大小上限 {API_BATCH_SIZE}）"
    )

    # 写入放子进程，与编码重叠进行
    points_q: multiprocessing.Queue = multiprocessing.Queue()
    result_q: multiprocessing.Queue = multiprocessing.Queue()
    writer = multiprocessing.Process(target=_write_qdrant_worker, args=(points_q, result_q))
    writer.start()

    encoded = 0
    batch_no = 0
    chunks.sort(key=lambda c: len(c["content"]), reverse=True)
    try:
        for start in range(0, len(chunks), API_BATCH_SIZE):
            # 写入进程已死就尽早中止；cancel_join_thread 防止 join
            # 写不出去的队列永久挂起（实测复现）
            if not writer.is_alive():
                points_q.cancel_join_thread()
                raise RuntimeError(
                    f"qdrant 写入进程提前退出（exitcode={writer.exitcode}），中止编码，详见日志"
                )
            batch_started = time.perf_counter()
            batch = chunks[start:start + API_BATCH_SIZE]
            batch_no += 1
            # 报错即退出：不可重试错误/重试耗尽由 siliconflow 客户端抛出，直接中止
            content_vectors = client.encode([c["content"] for c in batch])
            title_vectors = client.encode([" > ".join(c["header_path"]) for c in batch])
            for i, v in enumerate(batch):
                if not isinstance(v["header_path"], list):
                    title_vectors[i] = None
                    continue
                if len(v["header_path"]) == 0:
                    title_vectors[i] = None
            encoded += len(content_vectors)
            points_batch = [_to_point(c, v) for c, v in zip(batch, zip(title_vectors, content_vectors))]
            if points_batch:
                points_q.put(points_batch)

            duration = time.perf_counter() - batch_started
            logger.info(f"批次 {batch_no} 耗时 {duration:.2f}s，已生成 {encoded}/{len(chunks)} 个向量")
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
    logger.info(
        f"总共写入 {written} 个 chunk 到 Qdrant（共生成 {encoded} 个向量，"
        f"API 累计消耗 {client.total_tokens} tokens）"
    )
    elapsed = time.perf_counter() - started
    logger.info(f"qdrant 索引构建完成，总耗时 {format_duration(elapsed)}")
    return elapsed


if __name__ == "__main__":
    build(PROJECT_DIR / "data/chunks.jsonl")
