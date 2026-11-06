import re
from multiprocessing import Pool
from loguru import logger
from rst_to_myst import rst_to_myst
from pathlib import Path

from .libs.utils import PROJECT_DIR, DOCS_DIRS

DOCS_DIR_NAME = "godot-docs-master"
DOCS_SOURCE = PROJECT_DIR / DOCS_DIR_NAME
LOG_PATH = PROJECT_DIR / f"logs/{Path(__file__).stem}.log"


def strip_grid_borders(text: str) -> str:
    # 删除所有 +---+---+ 边框行（连同换行）
    return re.sub(r'(?m)^\s*\+[-=+]+\+\s*$\n?', '', text)


def run(f: Path):
    output = rst_to_myst(f.read_text(encoding='utf-8'))
    try:
        log_content = output.warning_stream.getvalue()
    except AttributeError:
        log_content = str(output.warning_stream)
    if log_content.strip():
       logger.error(log_content)
       
    output_path = f.with_name(f.stem + '.md')
    output_path.write_text(strip_grid_borders(output.text), encoding='utf-8')


if __name__ == "__main__":
    logger.add(LOG_PATH)
    with Pool() as pool:
        for d in DOCS_DIRS:
            doc_dir = DOCS_SOURCE / d
            print(doc_dir)
            pool.map(run, [f for f in doc_dir.rglob('**/*.rst')])
