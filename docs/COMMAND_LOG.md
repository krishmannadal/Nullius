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

## Step A.2 — git init + FEVER corpus (2026-08-28)

```bash
git init && git add -A && git commit    # -> 2d5c846; config.git_sha() now returns a real sha
#   .gitattributes added: * text=auto eol=lf, so corpus/trace JSONL bytes stay stable

# FEVER. The old S3 URLs (s3-eu-west-1.amazonaws.com/fever.public/*) return 403.
curl -sIL https://fever.ai/download/fever/wiki-pages.zip      # 200, 1,713,485,474 bytes
curl -L -o data/raw/shared_task_dev.jsonl https://fever.ai/download/fever/shared_task_dev.jsonl
curl -L -o data/raw/wiki-pages.zip        https://fever.ai/download/fever/wiki-pages.zip
#   -> 1,713,485,474 bytes, matches Content-Length exactly

# dev-set structure (measured, not assumed)
#   19,998 rows, exactly 6,666 per class; 2,892 distinct gold pages
#   11.3% of evidence groups span >1 page (genuine multi-hop)

# zip structure: 218 members = 109 data files + 109 macOS AppleDouble forks
#   (__MACOSX/wiki-pages/._wiki-NNN.jsonl -- these END IN .jsonl and sort FIRST,
#    so a naive endswith(".jsonl") filter dies on JSONDecodeError at char 0)

./.venv/Scripts/python.exe -m scripts.build_debug_corpus --n-claims 200 --n-docs 4000 --seed 1337
```

Measured on the first 3,000 real records: **6,200 of 16,205 sentences are empty
(38%)**. This is the concrete justification for ADR-019 — filtering empties would have
shifted the majority of gold sentence indices, silently.

Downloaded this step: 1.72 GB (FEVER). The zip is never extracted; the builder streams
members out of it, so peak disk stays at the zip.

### Two things the dump taught us (both found by running, not by reading)

**1. `__MACOSX` resource forks.** `wiki-pages.zip` has 218 members: 109 data files and
109 AppleDouble forks (`__MACOSX/wiki-pages/._wiki-NNN.jsonl`). They end in `.jsonl`,
they are small binary blobs, and `._` sorts *before* `w` — so `endswith(".jsonl")`
picks one first and dies with `JSONDecodeError: Expecting value: line 1 column 1`.
Fixed by `is_real_wiki_member()`, tested.

**2. Windows cp1252 console encoding.** The first build scanned all 5.4 M pages
successfully and then died inside a `print()`:

```
UnicodeEncodeError: 'charmap' codec can't encode character '́' in position 56
```

A Wikipedia page title contains a combining acute accent, which cp1252 cannot encode.
Four minutes of work lost in a status message. Fixed at the top of the script with
`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`. This will recur in the
CLI and in the FastAPI logs; it belongs in a shared helper.

Also noted: the first record of `wiki-001.jsonl` has an empty `id` and no lines —
`iter_wiki_pages` already skips falsy ids.

### The third thing the dump taught us: FEVER's own files disagree on Unicode

The build's "gold pages absent from the dump" warning fired for exactly one page,
`Cléopâtre`. Chasing it found a real bug in FEVER itself:

```
shared_task_dev.jsonl:  b'Cléopâtre'   NFD, decomposed
wiki-pages.zip:         b'Cl\xe9op\xe2tre'          NFC, precomposed
exact equal (==)          : False
equal after NFC normalise : True
```

Measured over all 2,892 dev gold pages:

```
not-NFC = 32/2892 = 1.11% of all gold pages
        = 32/44   = 72.7% of NON-ASCII gold pages
```

Björk, Curaçao, Café Society, Cléopâtre, Bañuela, 1974 Cypriot coup d'état…
The 1% aggregate is what makes this dangerous — the loss is not random, it removes
three quarters of an identifiable slice while looking like rounding error.

Fixed by `normalize_page_id()` at both boundaries (ADR-023) and rebuilt. Note the
same character caused the cp1252 crash above: the warning that found the bug is the
warning that could not print it.

## Step B — extractors, verifiers, aggregators (2026-08-28)

```bash
# spacy on its own pass -- installs in seconds; the earlier >15 min backtrack was from
# resolving it together with everything else, not a real conflict
./.venv/Scripts/python.exe -m pip install --no-cache-dir spacy==3.7.5
./.venv/Scripts/python.exe -m spacy download en_core_web_sm      # 3.7.1

# VRAM + label mapping measured BEFORE committing to the checkpoint
#   MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, fp16, RTX 4050:
#     184.4M params | weights 359 MiB | peak 467 MiB @ batch16x512 of 6140 MiB
#     forward 50-95 ms
#   id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}   <- REVERSE of the
#     common MNLI convention; hardcoding index 0 as contradiction would swap
#     Supported/Contradicted project-wide with no structural symptom
#   tokenizer.model_max_length = 1000000000000000019884624838656
#     -> truncation=True ALONE IS A NO-OP; explicit max_length required everywhere

./.venv/Scripts/python.exe -m pytest tests    # 229 passed, 0 skipped
```

Known-answer probe confirming the mapping behaves as its names claim:

```
expect ENTAILMENT     -> entailment     {'entailment': 0.9942, ...}
expect CONTRADICTION  -> contradiction  {'contradiction': 0.9995, ...}
expect NEUTRAL        -> neutral        {'neutral': 0.9988, ...}
```

No downloads beyond the two checkpoints (~370 MB DeBERTa, ~90 MB cross-encoder) and
en_core_web_sm (12 MB).
