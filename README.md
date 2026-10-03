# Nullius — a claim-level hallucination-detection *inspection harness*

> *Nullius in verba* — take nobody's word for it. Including this pipeline's.

This repository is a **research instrument**, not a system. Its job is to let you see
what every stage of a claim-verification pipeline does to a given input, so you can
find where it fails. Every component is a placeholder behind a clean interface. The
value is the interfaces and the visibility, not the quality of the defaults.

**Nothing in this repo is tuned, and nothing in it is evaluated.** Every corpus here is
built to contain the gold evidence for its own examples, so retrieval recall over it is
meaningless as a number — and so is anything downstream of it. Two exist:

* `data/debug/mini/` — 40 hand-written documents, 115 sentences, 14 examples. Checked
  in, offline, milliseconds. This is what the tests run against.
* `data/debug/` — FEVER-derived: 200 dev claims (stratified) plus ~4,000 Wikipedia
  pages, built by `scripts/build_debug_corpus.py` from FEVER's 2017 dump. The gold page
  for every shipped claim is present **by construction** and the distractor pool is not
  adversarial, so recall over it is an artifact of its assembly. `data/debug/manifest.json`
  carries that warning inside the artifact. See `docs/data-fever.md`.

Both are labelled `harness.kind: debug` in the config, and the UI will show a banner.
Full-corpus indexing is a later problem; see `docs/EXPERIMENT_BACKLOG.md`.

## The pipeline

```
response text
  → ClaimExtractor       → list[Claim]
  → Retriever            → list[Evidence]      (per claim)
  → Reranker (optional)  → list[Evidence]
  → Verifier             → EvidenceVerdict     (per claim-evidence PAIR)
  → Aggregator           → ClaimVerdict        (Supported / Contradicted / Insufficient / Abstain)
  → Trace                → JSONL
```

The load-bearing design decision: the **Verifier scores one pair** and the
**Aggregator is a separate object**. A verifier cannot pool over *k* because its
signature does not let it see *k*. Aggregation is the slot most likely to become the
research question, so it is isolated and instrumented from day one — every aggregator
must write `rule`, `explanation`, and `decisive_evidence_ids` into
`aggregation_trace`, and the type system enforces it.

Because of that split, **all five aggregators run on identical per-pair scores** — the
NLI model runs once and comparing four aggregation rules costs zero extra forward
passes.

Two null baselines are wired in from the start, not bolted on later:
`claim_only` (verifies with the evidence blanked) and `majority` (votes, discarding
probability magnitudes). If the real pipeline cannot beat them, that is the finding.

## Build status

The original seven-step plan and `docs/CLAUDE_CODE_EXTENSION_PROMPT.md` have been
merged into one order (ADR-012) — the extension's endpoint set is a superset of the
harness's, so there is one backend, not two.

| Step | What | State |
|---|---|---|
| 1 | `src/core/` — types, interfaces, registry, config | **done** |
| A | corpus layer + mini & FEVER corpora + BM25 / dense / hybrid retrievers | **done** |
| B | extractors, verifiers, 4 aggregators, 2 null baselines | **done** |
| C | trace writer + CLI end-to-end on 20 examples | **done** |
| D | one FastAPI app: `/analyze`, `/analyze/oracle`, `/verify/quick`, `/verify/full`, `/annotate`, `/health` | **done** |
| E | Streamlit inspection harness | **implemented** |
| F | Chrome MV3 extension for ChatGPT: Quick Check, Full Inspection, evidence traceability | **implemented** |
| — | `docs/EXPERIMENT_BACKLOG.md` | not started |

FastAPI backend, Streamlit harness, and Chrome MV3 extension: **implemented**. The extension requires a local backend and uses a limited debug corpus; scientific evaluation remains separate.

E and F are independent once D exists.

## Setup (Windows 11, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# torch first, from the CUDA 12.1 index (the PyPI wheel is CPU-only):
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip freeze > requirements.lock.txt
```

The core contracts and the corpus layer need only `pyyaml` and `pytest`. BM25 adds
`rank_bm25`; the dense retriever adds `torch`, `sentence-transformers` and `faiss-cpu`.

```powershell
python -m scripts.build_mini_corpus   # regenerate the offline corpus (already checked in)
python -m pytest                      # the test suite currently contains >260 tests, 0 skipped with the full stack

# the FEVER corpus (optional; 1.72 GB download, one streaming pass, several minutes)
curl -L -o data/raw/shared_task_dev.jsonl https://fever.ai/download/fever/shared_task_dev.jsonl
curl -L -o data/raw/wiki-pages.zip        https://fever.ai/download/fever/wiki-pages.zip
python -m scripts.build_debug_corpus --n-claims 200 --n-docs 4000 --seed 1337
```

A **skip is not a pass** — `python -m pytest -rs` prints why. All skips mean a missing
dependency, never a disabled check.

## Layout

```
src/core/types.py            frozen dataclasses: Claim, Evidence, EvidenceVerdict, ClaimVerdict, Trace
src/core/interfaces.py       the five ABCs + runtime contract checks
src/core/registry.py         YAML name -> component class; what the UI dropdowns read
src/core/config.py           config load / override / hash, seeding, git provenance
src/data/corpus.py           canonical sentence order + fingerprint (index-alignment guard)
src/data/examples.py         labelled examples with gold evidence keys
src/data/metrics.py          per-example Recall@k, gold ranks (None when gold is empty)
src/components/extractors.py  spacy sentence split; LLM decomposition from a cache
src/components/retrievers.py  BM25, dense (bge-small + FAISS flat), hybrid (RRF)
src/components/rerankers.py   noop, cross-encoder (ms-marco-MiniLM)
src/components/verifiers.py   NLI (DeBERTa-v3-base), cosine similarity, claim-only NULL
src/components/aggregators.py max-entailment, noisy-OR, rank-weighted, threshold+abstain, majority NULL
src/service.py               shared application service layer (CLI + FastAPI execution)
src/api/                     FastAPI application, routes, and boundary Pydantic schemas
scripts/build_mini_corpus.py  regenerates the checked-in 40-doc offline corpus
scripts/build_debug_corpus.py FEVER -> data/debug/ (streams the 1.7 GB dump, never extracts)
configs/debug.yaml           FEVER debug harness (read its header before believing a number)
configs/mini.yaml            the 40-doc corpus; what you point at while changing code
data/debug/mini/             40 docs, 115 sentences, 14 examples — for tests, measures nothing
tests/                       contract tests for the silent-bug surfaces
docs/                        one doc per module + DECISIONS + OPEN_QUESTIONS + EXPERIMENT_BACKLOG
results/failure_cases/       where the UI's "save as failure case" button writes
```

## FastAPI backend

Start the local backend server:

```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is live at `http://127.0.0.1:8000/docs` and schema at `/openapi.json`.

Endpoints:
* `GET /health` — runtime diagnostics, device, CUDA availability, VRAM allocation, queue depth, loaded components.
* `POST /analyze` — standard retrieved pipeline run over response text.
* `POST /analyze/oracle` — oracle evaluation with gold evidence substitution (strictly isolated from normal inference).
* `POST /verify/quick` — Tier 1 availability check (extracts claims and queries retrieval; never asserts that a claim is false).
* `POST /verify/full` — Tier 2 full pipeline execution (supports JSON or Server-Sent Events streaming via `stream=true`).
* `POST /annotate` — writes a structured annotation record to `data/annotations/annotations.jsonl`.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Marie Curie was born in Warsaw."}'
```

*Note: Nullius is a research inspection harness, not a production detector.*

## Chrome extension for ChatGPT

The Manifest V3 extension lives in `src/extension`. It captures the latest
completed ChatGPT answer and offers Quick Check and Full Inspection in its popup.
Start its local backend with `python -m scripts.run_extension_backend --warm-full`,
then load `src/extension` through Chrome's **Load unpacked** action.

See [Chrome installation and troubleshooting](src/extension/README.md) for the
complete steps. The extension uses a limited debug corpus; missing evidence is
not proof of a false claim.

## Documentation contract

Every module gets `docs/<module>.md` with, in order: problem; formal I/O with shapes
and dtypes; algorithm in pseudocode; a line-by-line walkthrough of only the lines that
matter; data flow; a runnable test with expected output and failure signature;
limitations; rejected alternatives. `docs/DECISIONS.md` is an append-only ADR log.
`docs/OPEN_QUESTIONS.md` lists everything resolved by assumption rather than by
evidence — read it before trusting any behaviour it names.
