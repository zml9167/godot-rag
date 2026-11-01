from pathlib import Path

# 以本文件位置为锚点，与调用方脚本的深度无关
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent