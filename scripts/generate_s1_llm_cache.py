"""Reserved entry point for real, provenance-recorded LLM extraction.

No provider is implemented. Fail without writing placeholder model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def cache_key(response_text: str) -> str:
    return hashlib.blake2b(response_text.encode("utf-8"), digest_size=16).hexdigest()[:16]


def generate_llm_cache(s1_dir: Path, model_name: str) -> None:
    raise NotImplementedError(
        "No LLM provider client is configured. No cache was written. "
        "Complete independent annotation, human adjudication and gold freezing first. "
        "Then integrate a real provider with model, prompt, response IDs, raw outputs "
        "and timestamp provenance. Never substitute empty or simulated claims."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        generate_llm_cache(Path("data/eval/s1"), args.model)
    except NotImplementedError as exc:
        parser.exit(1, f"Blocked: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
