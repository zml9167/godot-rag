import re
import json
from loguru import logger
from chunkana import ChunkConfig, chunk_markdown

from .libs.utils import PROJECT_DIR

DOCS_DIR_NAME = "godot-docs-master"

EVAL_RST_FENCE = re.compile(r"(?m)^```\{eval-rst\}([ \t]*)$")
CHUNKANA_EVAL_RST_FENCE = "```eval_rst__"
DOCS_SOURCE = PROJECT_DIR / DOCS_DIR_NAME
LOG_PATH = PROJECT_DIR / "logs/chunk.log"
CHUNKS_PATH = PROJECT_DIR / "data/chunks.jsonl"
MISMATCH_REPORTS = PROJECT_DIR / "chunk_mismatch_reports"

DOCS_DIRS = ["about", "classes", "community", "engine_details", "getting_started", "tutorials"]

MISMATCH_REPORTS.mkdir(parents=True, exist_ok=True)


def write_chunk(line):
    with CHUNKS_PATH.open('a', encoding='utf-8') as fp:
        fp.write(json.dumps(line) + '\n')


def norm(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


if __name__ == "__main__":
    logger.add(LOG_PATH)
    config = ChunkConfig(max_chunk_size = 8192,strategy_override="structural")
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text('')
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
            if norm(content) != norm(chunks_content):
                logger.warning(f"Mismatch {str(f)}")
                (MISMATCH_REPORTS / f.name).write_text(chunks_content, encoding='utf-8')
            for c in chunks:
                if isinstance(c.metadata['header_path'], list):
                    header_path = c.metadata['header_path']
                elif isinstance(c.metadata['header_path'], str):
                    header_path = c.metadata['header_path'].strip('/').split('/')
                else:
                    logger.error(f"Unknown header_path type {f}")
                    continue
                source = str(f.relative_to(DOCS_SOURCE))
                chunk_id = source + ':' + str(c.start_line) + '-' + str(c.end_line)
                line = {"chunk_id": chunk_id, "source": source, "header_path": header_path, "content": c.content, "start_line": c.start_line, "end_line": c.end_line, "metadata": c.metadata, "chunk_type": d}
                write_chunk(line)