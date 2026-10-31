import re
import json
import logging
from pathlib import Path
from chunkana import ChunkConfig, chunk_markdown

EVAL_RST_FENCE = re.compile(r"(?m)^```\{eval-rst\}([ \t]*)$")
CHUNKANA_EVAL_RST_FENCE = "```eval_rst__"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCS_SOURCE = PROJECT_DIR / "docs-source"
DOCS_DIRS = ["about", "classes", "community", "engin_details", "getting_started", "totorials"]
LOG_PATH = PROJECT_DIR / "logs" / "chunk.log"
CHUNK_PATH = PROJECT_DIR / "data" / "chunks" / "chunk.jsonl"
MISMATCH_REPORTS = PROJECT_DIR / "chunk_mismatch_reports"

CHUNK_PATH.parent.mkdir(parents=True, exist_ok=True)
MISMATCH_REPORTS.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.touch()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=str(LOG_PATH),       # 输出到文件（不写则输出到 stderr）
    filemode="a",             # 'a' 追加（默认），'w' 覆盖
    encoding="utf-8",         # 建议显式指定，避免中文乱码
)

config = ChunkConfig(max_chunk_size = 8192,strategy_override="structural")


def write_chunk(line):
    with CHUNK_PATH.open('a', encoding='utf-8') as fp:
        fp.write(json.dumps(line) + '\n')


def norm(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


for d in DOCS_DIRS:
    for f in (DOCS_SOURCE / d).glob('*.md'):
        print(f)
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
            header_path = c.metadata['header_path'].strip('/').split('/')
            source = str(f.relative_to(DOCS_SOURCE))
            chunk_id = source + ':' + str(c.start_line) + '-' + str(c.end_line)
            line = {"chunk_id": chunk_id, "source": source, "header_path": header_path, "content": c.content, "start_line": c.start_line, "end_line": c.end_line, "metadata": c.metadata, "chunk_type": d}
            write_chunk(line)