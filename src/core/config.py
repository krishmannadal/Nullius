"""Config loading, hashing, seeding, provenance.

Small on purpose.  Its only job is to turn a YAML file plus command-line overrides
into (a) a plain dict the registry can build from, and (b) the three provenance
strings a Trace must carry: ``config_hash``, ``git_sha``, ``run_id``.

The rule this module exists to enforce: **a result that cannot be regenerated from
its config does not exist.**  So the hash covers the *resolved* config (after
overrides), not the file on disk, and it is computed over canonical JSON so that
key order and YAML formatting cannot change it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config.  No includes, no interpolation, no magic."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"config root must be a mapping, got {type(cfg).__name__}")
    cfg["_config_path"] = str(p.resolve())
    return cfg


def apply_overrides(cfg: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` strings (from CLI or the Streamlit sidebar).

    Values are parsed as YAML scalars, so ``k=5`` gives an int and ``name=bm25``
    gives a string -- the same rules as the file itself, which means a value typed
    on the command line and the same value written in the YAML resolve identically
    and therefore hash identically.

    Overrides may only *replace* existing leaves or add new ones; they never merge
    dicts, because a half-merged component spec is a debugging nightmare.
    """
    out = json.loads(json.dumps({k: v for k, v in cfg.items()}))  # deep copy, JSON-safe
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like path.to.key=value")
        dotted, _, raw = item.partition("=")
        keys = dotted.strip().split(".")
        node: Any = out
        for key in keys[:-1]:
            nxt = node.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                node[key] = nxt
            node = nxt
        node[keys[-1]] = yaml.safe_load(raw)
    return out


def canonical_json(cfg: Mapping[str, Any]) -> str:
    """Deterministic serialisation used for hashing.

    Keys beginning with ``_`` are excluded: ``_config_path`` differs between
    machines and would make the same experiment hash differently on your laptop and
    on a lab box, which defeats the purpose of the hash.
    """
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    return json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(cfg: Mapping[str, Any], length: int = 12) -> str:
    return hashlib.blake2b(canonical_json(cfg).encode("utf-8"), digest_size=16).hexdigest()[:length]


def git_sha(repo_root: Optional[Path] = None) -> Optional[str]:
    """Current commit sha, or None if this is not a git repo / git is absent.

    Returns None rather than raising or faking a value: an honest ``null`` in a
    trace is recoverable, a fabricated sha is not.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def git_is_dirty(repo_root: Optional[Path] = None) -> Optional[bool]:
    """True if there are uncommitted changes; None if not a git repo."""
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and torch if present.

    Note what this does NOT buy you: cuDNN kernel selection and some fused ops are
    still nondeterministic, so two runs with the same seed can differ in the last
    decimal of an NLI probability.  Full determinism costs throughput and is not
    worth it for an inspection harness; if a *result* ever depends on that last
    decimal, the result is the problem.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def project_root() -> Path:
    """Repo root, resolved from this file.  Never hardcode a drive letter."""
    return Path(__file__).resolve().parents[2]


def provenance(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The block that goes into every Trace and every results/<run_id>/ dir."""
    return {
        "config_hash": config_hash(cfg),
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "config_path": cfg.get("_config_path"),
    }


__all__ = [
    "load_config",
    "apply_overrides",
    "canonical_json",
    "config_hash",
    "git_sha",
    "git_is_dirty",
    "set_global_seed",
    "project_root",
    "provenance",
]
