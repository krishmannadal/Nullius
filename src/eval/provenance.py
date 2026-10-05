"""Content identity of the actual checkout, including uncommitted research code."""

import platform
from pathlib import Path

from src.core.config import git_is_dirty, git_sha
from src.eval.s1_annotations import canonical_sha256


def software_identity():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(
        [
            *root.joinpath("src").rglob("*.py"),
            *root.joinpath("scripts").glob("*.py"),
            root / "requirements.lock.txt",
            root / "pyproject.toml",
        ]
    )
    return {
        "git_revision": git_sha(),
        "git_dirty": git_is_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "canonical_file_sha256": {
            path.relative_to(root).as_posix(): canonical_sha256(path.read_bytes()) for path in paths
        },
        "note": "Content hashes identify the working source; preserve this checkout and model artifacts to rerun.",
    }
