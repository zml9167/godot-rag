import re
import json
from pathlib import Path
from loguru import logger
from chunkana import ChunkConfig, chunk_markdown
from chunkana.adaptive_sizing import AdaptiveSizeConfig

from .libs.utils import PROJECT_DIR, CHUNKS_PATH, DOCS_DIRS

DOCS_DIR_NAME = "godot-docs-master"

EVAL_RST_FENCE = re.compile(r"(?m)^```\{eval-rst\}([ \t]*)$")
CHUNKANA_EVAL_RST_FENCE = "```eval_rst__"
CURRENT_PATH = Path(__file__).resolve()
DOCS_SOURCE = PROJECT_DIR / DOCS_DIR_NAME
LOG_PATH = PROJECT_DIR / f"logs/{CURRENT_PATH.stem}.log"


def write_chunk(line):
    with CHUNKS_PATH.open('a', encoding='utf-8') as fp:
        fp.write(json.dumps(line) + '\n')


def norm(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


if __name__ == "__main__":
    logger.add(LOG_PATH, encoding="utf-8")
    config = ChunkConfig(
        strategy_override="structural",
        use_adaptive_sizing = True,
        adaptive_config=AdaptiveSizeConfig(base_size=2048)
        )
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text('')
    chunk_count = 0
    doc_count = 0
    mismatch_docs = 0
    blank_only_docs = 0
    blank_chunks = 0
    for d in DOCS_DIRS:
        doc_dir = DOCS_SOURCE / d
        print(doc_dir)
        for f in doc_dir.glob("**/*.md"):
            content = f.read_text(encoding="utf-8")
            if CHUNKANA_EVAL_RST_FENCE in content:
                logger.info(f"Source already contains the reserved fence marker: {f}\n")
            # Chunkana's fence parser accepts word characters in info strings, but not
            # MyST's `{eval-rst}`. Keep the replacement the same length and restore it below.
            chunkable_content = EVAL_RST_FENCE.sub(r"```eval_rst__\1", content)
            chunks = chunk_markdown(chunkable_content, config)
            for chunk in chunks:
                chunk.content = re.sub(
                    r"(?m)^```eval_rst__([ \t]*)$",
                    r"```{eval-rst}\1",
                    chunk.content,
                )
            chunks_content = '\n'.join([c.content for c in chunks])
            orig_lines = norm(content)
            new_lines = norm(chunks_content)
            doc_count += 1
            if orig_lines != new_lines:
                mismatch_docs += 1
                # 空行差异已被 norm 过滤；用非空行拼接后的长度判断是否真丢了内容，
                # 否则原文里的空行在重建中增删会造成误报
                lost = len("\n".join(orig_lines)) - len("\n".join(new_lines))
                rel = f.relative_to(DOCS_SOURCE)
                if lost > 0:
                    logger.warning(
                        f"{rel}: 切分重建后内容丢失 {lost} 字符"
                        f"（非空行 {len(orig_lines)} -> {len(new_lines)}）"
                    )
                else:
                    logger.warning(
                        f"{rel}: 切分重建与原文不一致"
                        f"（非空行 {len(orig_lines)} -> {len(new_lines)}，长度差 {lost:+d} 字符）"
                    )
            elif content != chunks_content:
                # 行内容完全一致，仅空行/结尾差异——不影响 RAG 质量，只计数不刷屏
                blank_only_docs += 1
            for c in chunks:
                if not c.content.strip():
                    # 空白 chunk 无检索价值且下游 API 拒绝空串：跳过不写入。
                    # 内容丢失由上面的重建一致性校验兜底，空白 chunk 不影响它
                    blank_chunks += 1
                    logger.warning(
                        f"{f.relative_to(DOCS_SOURCE)}: 跳过空白 chunk"
                        f"（行 {c.start_line}-{c.end_line}）"
                    )
                    continue
                if isinstance(c.metadata['header_path'], list):
                    header_path = c.metadata['header_path']
                elif isinstance(c.metadata['header_path'], str):
                    header_path = c.metadata['header_path'].strip('/').split('/')
                else:
                    bad_header_path = c.metadata['header_path']
                    logger.error(
                        f"Unknown header_path type {type(bad_header_path).__name__}: "
                        f"{bad_header_path!r} (file: {f})"
                    )
                    continue
                source = f.relative_to(DOCS_SOURCE).as_posix()
                chunk_id = source + ':' + str(c.start_line) + '-' + str(c.end_line)
                line = {"chunk_id": chunk_id, "source": source, "header_path": header_path, "content": c.content, "start_line": c.start_line, "end_line": c.end_line, "metadata": c.metadata, "chunk_type": d}
                write_chunk(line)
                chunk_count += 1
    logger.info(
        f"已写入chunk {chunk_count}；校验 {doc_count} 个文档，"
        f"{mismatch_docs} 个重建不一致，{blank_only_docs} 个仅空行差异，"
        f"跳过 {blank_chunks} 个空白 chunk"
    )