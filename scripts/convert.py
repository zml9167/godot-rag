import logging
from rst_to_myst import rst_to_myst
from pathlib import Path

DOCS_DIR_NAME = "godot-docs-master"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCS_SOURCE = PROJECT_DIR / DOCS_DIR_NAME
DOCS_DIRS = ["about", "classes", "community", "engine_details", "getting_started", "tutorials"]
LOG_PATH = PROJECT_DIR / "logs" / "convert.log"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.touch()
LOG_PATH.write_text('')

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=str(LOG_PATH),       # 输出到文件（不写则输出到 stderr）
    filemode="a",             # 'a' 追加（默认），'w' 覆盖
    encoding="utf-8",         # 建议显式指定，避免中文乱码
)

for d in DOCS_DIRS:
    doc_dir = DOCS_SOURCE / d
    print(doc_dir)
    for f in doc_dir.rglob('**/*.rst'):
        output = rst_to_myst(f.read_text(encoding='utf-8'))
        try:
            log_content = output.warning_stream.getvalue()
        except AttributeError:
            log_content = str(output.warning_stream)
        if log_content.strip():
           logger.error(log_content + '\n')

        output_path = f.with_name(f.stem + '.md')
        output_path.write_text(output.text, encoding='utf-8')