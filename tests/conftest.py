"""Research artifacts are read-only to every test, even accidentally."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = [ROOT / "data/eval/s1", ROOT / "data/annotations", ROOT / "results"]


def _protected(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    try:
        path = Path(os.fsdecode(value)).resolve()
        return any(path == root or path.is_relative_to(root) for root in PROTECTED)
    except (OSError, ValueError):
        return False


def _audit(event, args):
    if event == "open":
        path, _mode, flags = args
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and _protected(path):
            raise PermissionError(f"Tests must not write research artifacts: {path}")
    elif event in {"os.remove", "os.rmdir", "os.rename", "os.mkdir"}:
        if any(_protected(p) for p in args[:2]):
            raise PermissionError(f"Tests must isolate filesystem mutation: {args[0]}")


sys.addaudithook(_audit)
