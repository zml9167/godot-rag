import subprocess
import sys
from pathlib import Path
from paths import PROJECT_DIR

DOCS_REPOSITORY = "https://github.com/godotengine/godot-docs.git"
DOCS_DIR = PROJECT_DIR / "godot-docs-master"
PIPELINE_MODULES = (
    "scripts.rst2md",
    "scripts.chunk_documents",
    "scripts.build_index",
)


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

    result = run(
        ["git", "-C", str(DOCS_DIR), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
    )
    if result.stdout.strip() != "true":
        raise RuntimeError(
            f"{DOCS_DIR} already exists but is not a Git repository; "
            "move it aside or rename it before running the pipeline."
        )

    origin = run(
        ["git", "-C", str(DOCS_DIR), "remote", "get-url", "origin"],
        capture_output=True,
    ).stdout.strip()
    if normalize_repository_url(origin) != "godotengine/godot-docs":
        raise RuntimeError(
            f"Unexpected origin for {DOCS_DIR}: {origin}\n"
            f"Expected {DOCS_REPOSITORY}"
        )

    changes = run(
        [
            "git",
            "-C",
            str(DOCS_DIR),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
    ).stdout.strip()
    if changes:
        raise RuntimeError(
            f"{DOCS_DIR} has tracked local changes. Commit or stash them "
            "before updating the documentation repository."
        )

    print("Updating Godot documentation repository", flush=True)
    run(["git", "-C", str(DOCS_DIR), "pull", "--ff-only"])


def main():
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

    print("\nPipeline completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
