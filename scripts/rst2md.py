from multiprocessing import Pool
from loguru import logger
from rst_to_myst import rst_to_myst
from pathlib import Path

from .paths import PROJECT_DIR

DOCS_DIR_NAME = "godot-docs-master"

DOCS_SOURCE = PROJECT_DIR / DOCS_DIR_NAME
LOG_PATH = PROJECT_DIR / "logs/convert.log"

DOCS_DIRS = ["about", "classes", "community", "engine_details", "getting_started", "tutorials"]


def run(f: Path):
    output = rst_to_myst(f.read_text(encoding='utf-8'))
    try:
        log_content = output.warning_stream.getvalue()
    except AttributeError:
        log_content = str(output.warning_stream)
    if log_content.strip():
       logger.error(log_content + '\n')
       
    output_path = f.with_name(f.stem + '.md')
    output_path.write_text(output.text, encoding='utf-8')


if __name__ == "__main__":
    logger.add(LOG_PATH)
    with Pool() as pool:
        for d in DOCS_DIRS:
            doc_dir = DOCS_SOURCE / d
            print(doc_dir)
            pool.map(run, [f for f in doc_dir.rglob('**/*.rst')])
