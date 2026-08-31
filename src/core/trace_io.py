"""Writing and reading traces, and the layout of ``results/<run_id>/``.

The rule this file enforces: **a result that cannot be regenerated from its own
directory does not exist.** So a run directory is self-describing — it carries the
resolved config, the git sha, the environment, and the metrics alongside the traces,
and nothing in it points at state that lives only in this process.

    results/<run_id>/
        traces.jsonl        one Trace per line, retrieved mode
        oracle.jsonl        one Trace per line, oracle mode (absent if none ran)
        config.json         the RESOLVED config, after overrides
        provenance.json     git sha, dirty flag, python/torch/cuda versions, timestamp
        metrics.json        counts and timings, labelled as debug-harness output

Failure cases go somewhere else on purpose:

    results/failure_cases/<timestamp>-<claim_id>.json

one file per saved case, holding the full trace plus a free-text note. One file rather
than an appended log because these get read, edited, and moved around by hand over
weeks; a corrupted append would take the corpus with it.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from src.core.config import git_is_dirty, git_sha, project_root
from src.core.types import SCHEMA_VERSION, Trace


class TraceWriter:
    """Append traces to a JSONL file, one per line.

    Opened in append mode and flushed per write, so an interrupted run keeps every
    trace it had already produced. A long run that dies at example 180 of 200 should
    not cost you the first 179.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self.count = 0

    def write(self, trace: Trace) -> None:
        self._fh.write(trace.to_json_line() + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False


def read_traces(path: str | Path) -> Iterator[Trace]:
    """Stream traces back. Raises on the first malformed line, naming it.

    Deliberately not lenient: a trace file with a bad line is a file you cannot trust,
    and skipping the bad line silently changes every count computed from it.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Trace.from_json_line(line)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{p}:{lineno}: not a valid trace: {exc}") from None


def environment() -> dict[str, Any]:
    """What the config hash cannot cover: code, weights, and driver versions.

    Two runs with the same ``config_hash`` and different library versions are different
    experiments, and nothing in the config would say so. This is the other half of
    reproducibility, and it is cheap to record.
    """
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "schema_version": SCHEMA_VERSION,
    }
    lock_path = project_root() / "requirements.lock.txt"
    if lock_path.is_file():
        import hashlib
        env["lock_hash"] = hashlib.sha256(lock_path.read_bytes()).hexdigest()[:16]
    else:
        env["lock_hash"] = None
    for name in ("torch", "transformers", "sentence_transformers", "faiss", "spacy", "numpy"):
        try:
            module = __import__(name)
            env[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            env[name] = None
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["torch_cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    return env


def write_run(
    run_dir: str | Path,
    *,
    resolved_config: dict[str, Any],
    metrics: dict[str, Any],
    harness_kind: str = "debug",
) -> Path:
    """Write the non-trace half of a run directory."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)

    (d / "config.json").write_text(
        json.dumps(resolved_config, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    (d / "provenance.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "git_dirty": git_is_dirty(),
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "environment": environment(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payload = dict(metrics)
    # The warning travels INSIDE the artifact, so a metrics file read six weeks from
    # now cannot be mistaken for an evaluation result.
    payload["_warning"] = (
        f"harness_kind={harness_kind}. Every corpus in this repo contains the gold "
        "evidence for its own examples by construction and its distractor pool is not "
        "adversarial. These numbers describe the harness, not a method. Nothing here "
        "is tuned. Do not report them."
    )
    (d / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return d


def run_dir_for(run_id: str, results_dir: str | Path = "results") -> Path:
    base = Path(results_dir)
    if not base.is_absolute():
        base = project_root() / base
    return base / run_id


# --------------------------------------------------------------------------- #
# failure cases — the error-analysis corpus
# --------------------------------------------------------------------------- #

def save_failure_case(
    trace: Trace,
    note: str,
    *,
    claim_id: str | None = None,
    oracle_trace: Trace | None = None,
    extra: dict[str, Any] | None = None,
    out_dir: str | Path = "results/failure_cases",
) -> Path:
    """Write one saved failure case. The frontend's save button calls this.

    The record embeds the WHOLE trace, not a summary, and the oracle trace beside it
    when there is one. Six weeks from now the question will be "why did it decide
    that", and a summary cannot answer it — ``aggregation_trace`` can.

    Same format as the browser extension's annotation record will use, so one
    ``annotation_stats.py`` reads both (ADR-012).
    """
    d = Path(out_dir)
    if not d.is_absolute():
        d = project_root() / out_dir
    d.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = (claim_id or trace.run_id).replace(":", "_").replace("/", "_")
    path = d / f"{stamp}-{suffix}.json"

    record = {
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "note": note,
        "claim_id": claim_id,
        "source": "harness",
        "trace": trace.to_dict(),
        "oracle_trace": oracle_trace.to_dict() if oracle_trace else None,
        "git_sha": trace.git_sha,
        "config_hash": trace.config_hash,
        **(extra or {}),
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_failure_cases(out_dir: str | Path = "results/failure_cases") -> list[dict[str, Any]]:
    d = Path(out_dir)
    if not d.is_absolute():
        d = project_root() / out_dir
    if not d.is_dir():
        return []
    cases = []
    for p in sorted(d.glob("*.json")):
        cases.append(json.loads(p.read_text(encoding="utf-8")))
    return cases


__all__ = [
    "TraceWriter",
    "environment",
    "load_failure_cases",
    "read_traces",
    "run_dir_for",
    "save_failure_case",
    "write_run",
]
