import subprocess
import sys
from pathlib import Path

import portalocker

from .libs.utils import PROJECT_DIR

DOCS_REPOSITORY = "https://github.com/godotengine/godot-docs.git"
DOCS_DIR = PROJECT_DIR / "godot-docs-master"
LOCK_PATH = PROJECT_DIR / "data/.pipeline.lock"
PIPELINE_MODULES = (
    "scripts.rst2md",
    "scripts.chunk_documents",
    "scripts.build_all_index",
)


def acquire_pipeline_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = portalocker.Lock(str(LOCK_PATH), mode="w", timeout=0)
    try:
        lock.acquire()
    except portalocker.exceptions.LockException as error:
        raise RuntimeError(
            "Another pipeline run is already in progress "
            f"(lock file: {LOCK_PATH}). Wait for it to finish first."
        ) from error
    return lock


def run(command, cwd=PROJECT_DIR, *, capture_output=False):
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def validate_pipeline_scripts():
    missing = [
        str(PROJECT_DIR / Path(*module.split(".")).with_suffix(".py"))
        for module in PIPELINE_MODULES
        if not (PROJECT_DIR / Path(*module.split(".")).with_suffix(".py")).is_file()
        or (PROJECT_DIR / Path(*module.split(".")).with_suffix(".py")).stat().st_size
        == 0
    ]
    if missing:
        raise FileNotFoundError(
            "Required pipeline scripts are missing or empty:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def normalize_repository_url(url):
    url = url.strip()
    if url.startswith("git@github.com:"):
        url = url[len("git@github.com:") :]
    elif "github.com/" in url:
        url = url.split("github.com/", 1)[1]
    return url.removesuffix(".git").strip("/")


def prepare_docs_repository():
    if not DOCS_DIR.exists():
        print(f"Cloning Godot documentation into {DOCS_DIR}", flush=True)
        run(["git", "clone", "--depth", "1", DOCS_REPOSITORY, str(DOCS_DIR)])
        return

    if not DOCS_DIR.is_dir():
        raise NotADirectoryError(f"Documentation path is not a directory: {DOCS_DIR}")


def main():
    try:
        lock = acquire_pipeline_lock()
    except (OSError, RuntimeError) as error:
        print(f"Pipeline stopped: {error}", file=sys.stderr)
        return 1

    try:
        validate_pipeline_scripts()
        prepare_docs_repository()

        for module in PIPELINE_MODULES:
            command = [sys.executable, "-m", module]
            print(f"\nRunning: {module}", flush=True)
            run(command)
    except subprocess.CalledProcessError as error:
        print(
            f"Pipeline stopped: command exited with status {error.returncode}: "
            f"{error.cmd}",
            file=sys.stderr,
        )
        return error.returncode or 1
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as error:
        print(f"Pipeline stopped: {error}", file=sys.stderr)
        return 1
    finally:
        lock.release()

    print("\nPipeline completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
