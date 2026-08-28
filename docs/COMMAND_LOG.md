# Command log

Non-obvious commands run against this repo, in order. Trivial `ls`/`cat` omitted.

## Step 1 — core contracts (2026-08-28)

```bash
# environment probe
python --version                        # 3.11.0
nvidia-smi --query-gpu=name,memory.total --format=csv   # RTX 4050 Laptop, 6141 MiB
python -c "import yaml, pytest"         # pyyaml 6.0.1, pytest 8.1.0 already present

# scaffold (all inside C:\Users\krish\Nullius)
mkdir -p src/core src/components src/data configs tests docs results/failure_cases data/debug

# verification
python -m pytest tests -q                       # 55 passed
python -c "from src.core import registry; print(registry.available_all())"
#   -> all five kinds empty; correct, _BUILTIN_MODULES is empty until step 2
python -c "from src.core.config import load_config, config_hash; print(config_hash(load_config('configs/debug.yaml')))"
#   -> 6b92647c0633
```

Nothing was installed, downloaded, or deleted. No files written outside the project
directory. No `git init` — see OQ-005.

## Step A — corpus + retrievers (2026-08-28)

```bash
# environment (venv, so the system interpreter stays clean)
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
#   -> torch 2.4.1+cu121, cuda True, NVIDIA GeForce RTX 4050 Laptop GPU
# `pip install -r requirements.txt` BACKTRACKED FOR >15 MIN WITHOUT CONVERGING.
# Killed it and installed in stages, spacy excluded (only needed in step B):
./.venv/Scripts/python.exe -m pip install --no-cache-dir numpy==1.26.4 rank-bm25==0.2.2     faiss-cpu==1.8.0.post1 transformers==4.44.2 sentence-transformers==3.0.1
./.venv/Scripts/python.exe -m pip install --no-cache-dir pyyaml==6.0.1 pytest==8.3.3     pandas==2.2.2 fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.2 requests==2.32.3     streamlit==1.38.0
#   -> pytest 8.1.0 is YANKED (pytest-dev/pytest#12069); pin corrected to 8.3.3
#   -> scikit-learn resolved to 1.9.0, not the proposed 1.5.2; requirements.txt corrected
./.venv/Scripts/python.exe -m pip freeze > requirements.lock.txt

# corpus
python -m scripts.build_mini_corpus
#   -> 40 docs, 115 sentences, fingerprint aa7c5b1202fff125, 14 examples, 16 gold keys

# verification
./.venv/Scripts/python.exe -m pytest tests    # 92 passed, 0 skipped (16.9 s, incl. FAISS)

# live check: all three retrievers on the same claim, gold marked, Recall@5 shown
./.venv/Scripts/python.exe -c "from src.core.registry import build; ..."   # see step report
```

Disk note: `C:` is at 92% (40 GB free) and the pip cache alone is ~20 GB.
`pip cache purge` would reclaim most of that. Not run — it is your cache, and other
projects may be relying on it.

Downloaded this step: torch cu121 (~2.4 GB) plus the requirements set. No dataset
downloads; the FEVER wiki dump (~1.7 GB) is gated on OQ-016.
