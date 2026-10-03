"""Start the local Chrome-extension backend with warm spaCy/BM25 retrieval.

Run from the repository root: python -m scripts.run_extension_backend
Use --warm-full to download/load the real NLI model before opening the extension.
"""

from __future__ import annotations

import argparse

import uvicorn

from src.api.app import create_app
from src.service import NulliusService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warm-full", action="store_true", help="Load the real NLI model at startup"
    )
    args = parser.parse_args()
    service = NulliusService(default_config_path="configs/extension.yaml")
    print("Preparing Quick Check (spaCy + BM25; debug corpus)…", flush=True)
    service._ensure_quick_loaded()
    if args.warm_full:
        print("Preparing Full Inspection (NLI model; may download files)…", flush=True)
        service._ensure_loaded()
    print("Starting local Nullius server on 127.0.0.1:8000", flush=True)
    uvicorn.run(create_app(service=service), host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
