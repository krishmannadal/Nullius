# Open questions — things resolved by assumption, not by evidence

Everything here is a decision I made without asking, plus what it would take to
settle it. Read this before trusting any behaviour it names.

Status key: **[assumed]** = a guess is in the code · **[deferred]** = decided not to
decide yet · **[needs you]** = I cannot settle it without your input.

---

## Step 1 (core types / interfaces / registry / config)

### OQ-001 — `sent_id` is an `int` [assumed]
`Evidence.sent_id` is typed `int ≥ 0`, which matches FEVER's sentence indexing and
the planned debug corpus. If a corpus turns out to be addressed by string sentence
ids or by character offsets, `evidence_id()` and one dataclass field change.
**Settle by:** looking at the actual corpus format in step 2.

### OQ-002 — Similarity is stored as raw cosine in [−1, 1] [assumed]
`SimilarityVerifier` will return a raw cosine, unthresholded and un-rescaled, and
`EvidenceVerdict` validates that range. If the chosen bi-encoder produces
non-negative similarities in practice, the range is merely loose, not wrong. But a
model whose similarities are not cosines at all (dot product on unnormalised
embeddings) would fail validation.
**Settle by:** step 3, when the embedding model is pinned — assert normalisation.

### OQ-003 — `latency_ms` is measured per pair with `perf_counter` [assumed]
Not yet implemented. The intent is wall-clock around the forward pass only, excluding
tokenisation, so numbers are comparable between verifiers. Excluding tokenisation is
arguable: for a cross-encoder over long evidence it is not negligible.
**Settle by:** deciding in step 3 and stating it in `docs/components-verifiers.md`.
Tentative: record `latency_ms` as **tokenise + forward**, since that is what a user
of the pipeline actually waits for.

### OQ-004 — Two claims, one response, identical text [assumed]
Content-addressed claim ids include the source span, so two identical sentences at
different offsets get different ids. Two claims with the *same* text and the *same*
span cannot occur from a splitter, but could from an LLM extractor that emits a
duplicate. In that case `check_claims` raises "duplicate claim ids".
**Settle by:** step 3 — decide whether the LLM extractor de-duplicates silently or
whether a duplicate is an error worth seeing. Tentative: de-duplicate, and record the
count in `extractor_meta`.

### OQ-005 — ~~No git repo~~ RESOLVED 2026-08-28 [closed]
`C:\Users\krish\Nullius` is not a git repository, so provenance currently rests on
`config_hash` alone. I did not run `git init` because initialising version control is
your call, not a side effect of a build step.
**Resolved.** `git init` + first commit `2d5c846`. `config.git_sha()` returns a real
sha with no code change, and `git_is_dirty()` reports alongside it, exactly as ADR-009
predicted. Provenance is live.

### OQ-006 — Dependency pins are proposed, not resolved [assumed]
`requirements.txt` is written from knowledge of what these packages target, and
**nothing in it has been installed on this machine**. Specific risks: the `faiss-cpu`
Windows wheel for Python 3.11, the `numpy<2` constraint against the torch/faiss ABI,
and whether `spacy==3.7.5` and `sentence-transformers==3.0.1` co-resolve.
**Settle by:** step 2 begins with creating `.venv`, installing, and freezing
`requirements.lock.txt`. Treat the lock file as fact and the pins as intent.

### OQ-007 — `PROB_SUM_TOL = 1e-3` [assumed]
Chosen to survive fp16 softmax rounding over three terms while still catching stored
logits. It cannot catch a 2-class head padded as `(p, 0.0, 1−p)` — that sums to 1.
**Settle by:** step 3's verifier test, which asserts the checkpoint's index↔name
mapping against a known-answer pair rather than trusting the sum.

### OQ-008 — Full determinism is not enforced [deferred]
`set_global_seed` seeds python/numpy/torch but does not set
`torch.use_deterministic_algorithms(True)` or `cudnn.deterministic`. Two same-seed
runs can therefore differ in the last decimal of an NLI probability.
**Settle by:** leaving it, unless a bug appears that is nondeterminism-shaped.

### OQ-009 — Component instances are rebuilt on every config change [deferred]
`build_pipeline` constructs eagerly, so flipping a sidebar dropdown reloads the model
behind it. Fine for a CLI; visibly bad in Streamlit, where a `k` slider change would
reload DeBERTa.
**Settle by:** step 5/6 — an instance cache keyed on `(kind, name, frozen params)`.
Noted now so it is not discovered as a UI bug.

### OQ-010 — What counts as "the same experiment" [assumed]
`config_hash` covers the resolved config only — not the code, the model weights, or
the corpus. Two runs with equal hashes and different `transformers` versions are
different experiments and will not look it.
**Settle by:** step 4 writes `results/<run_id>/` containing the resolved config,
`git_sha`, and `requirements.lock.txt` hash. The run directory, not the config hash,
is the unit of reproducibility.

### OQ-011 — Debug corpus size and construction [needs you, deferred to step 2]
"A few thousand documents" is the instruction. Two sub-decisions are not settled:
(a) whether the corpus is gold documents + sampled distractors (recall becomes
meaningless but positive) or a random Wikipedia slice (recall becomes near-zero and
the verifier is never exercised); (b) whether examples come from FEVER dev.
**Tentative plan for step 2, say if you disagree:** gold documents for ~200 FEVER dev
claims plus ~3–5k random distractor pages, explicitly labelled as making recall
uninformative, because the harness's job is to exercise every stage rather than to
measure any of them.

### OQ-012 — Which spaCy model, and whether spaCy at all [deferred]
`SpacySentenceExtractor` implies `en_core_web_sm` (~12 MB, CPU). A `blank("en")` pipe
with only the `sentencizer` is faster and has no model download, at the cost of worse
sentence boundaries on abbreviations ("Dr.", "U.S.").
**Settle by:** step 3. Tentative: `en_core_web_sm`, since boundary errors would show
up as decomposition errors and contaminate exactly the thing you want to measure.

### OQ-013 — RRF constant `k = 60` [assumed, placeholder]
`configs/debug.yaml` names `rrf_k: 60`, the value from the original RRF paper
(Cormack et al., 2009). It is a placeholder, not a tuned value, and the config says
so. Its sensitivity is a backlog item, not a step-2 decision.

### OQ-014 — The BRIEF's stage plan vs this build's step order [assumed]
`docs/BRIEF.md` defines a nine-stage research plan with gates; the current
instruction defines a seven-step build with different scope and an explicit "do not
optimise" boundary. I have treated the current instruction as governing and the
brief's *standards* sections (documentation contract, engineering standards, hardware
envelope) as still binding. See ADR-001.
**Settle by:** telling me if the brief's stage plan should resume after step 7.

---

## Step A (corpus + retrievers)

### OQ-015 — The mini corpus is 40 documents, not "a few thousand" [assumed]
You approved "gold docs for ~200 FEVER dev claims + ~3–5k distractors". What is built
and checked in is much smaller: 40 hand-written documents, 115 sentences, 14 examples.
This is deliberate and additive, not a substitution — it exists so the test suite runs
offline in milliseconds and so every stage can be watched today without a 1.7 GB
download. The FEVER-derived debug corpus is the next artifact and needs a decision
(OQ-016).
**Settle by:** nothing; both corpora coexist. `data/debug/mini/` for tests,
`data/debug/` for the FEVER-derived one.

### OQ-016 — ~~FEVER ingestion~~ RESOLVED 2026-08-28 [closed]
Gold evidence in FEVER is `(page, sentence_id)` into FEVER's **June 2017** Wikipedia
snapshot. Fetching current Wikipedia by title instead (~30 MB, no dump) would be much
cheaper, but current article text has different sentence boundaries, so gold indices
would have to be re-derived by string matching — which is lossy and would silently
corrupt `is_gold`, the one signal the oracle comparison depends on. I will not do that.
So the honest options are the dump, or staying on hand-written corpora.
**Cost:** `wiki-pages.zip` ≈ 1.7 GB download; we extract only the ~3–5k pages we need,
so the on-disk debug corpus stays small. Disk is currently at 92% (40 GB free), and
the pip cache alone is ~20 GB — `pip cache purge` would free most of that.
**Resolved: go.** Dump downloaded from `fever.ai` (the old S3 URLs return 403).
`scripts/build_debug_corpus.py` streams it without extracting. See ADR-018 and
`docs/data-fever.md`.

### OQ-017 — `bge-small-en-v1.5` picked arbitrarily [assumed]
`e5-small-v2` and `all-MiniLM-L6-v2` all fit 6 GB comfortably and are all plausible.
BGE was picked with no comparison. The `query_prefix` constructor param exists because
these families differ in exactly this convention (BGE puts an instruction on the query
side only; E5 uses `query:`/`passage:` on both; MiniLM uses neither).
**Settle by:** a backlog experiment, not a build decision.

### OQ-018 — The index manifest does not record the encoder library version [assumed]
It records corpus fingerprint, model name, dimension, and sentence count. A
`sentence-transformers` or tokenizer upgrade can change embeddings without changing
any of those, so a stale index would load clean. Low probability, silent if it happens.
**Settle by:** adding `sentence_transformers.__version__` and `transformers.__version__`
to the manifest when `requirements.lock.txt` is frozen. One line; deferred only because
the version is not pinned yet.

### OQ-019 — `load_corpus_cached` is keyed on path, not contents [assumed]
Editing a corpus file inside a running process (a Streamlit session, notably) serves
the stale corpus from cache. `lru_cache(maxsize=8)` also evicts silently on the ninth
distinct path.
**Settle by:** step E, when the UI can trigger a reload. Tentative: key on
`(path, st_mtime, st_size)`.

### OQ-020 — `n_arms_hit` looks for literal `bm25_rank` / `dense_rank` keys [assumed]
A third fusion arm with different `retriever_meta` keys will not be counted. It is a
panel affordance rather than a general mechanism.
**Settle by:** if a third arm ever exists, count distinct `*_rank` suffixes instead.

### OQ-021 — `latency_ms` is not yet recorded by retrievers [deferred]
`Evidence` has no latency field; only `EvidenceVerdict` does. Per-stage retrieval
timing therefore lives in `Trace.timings`, which is written by the pipeline runner
(step C), not by the retrievers themselves.
**Settle by:** step C. Tentative keys: `retrieve_ms`, `rerank_ms`, `verify_ms`,
`aggregate_ms`, all per claim and summed.

### OQ-022 — `--n-docs` counts pages, but sentences are what cost [assumed]
The debug corpus is specified as ~4,000 *pages*; FEVER pages average tens of sentences,
so the sentence count (which governs index size, encode time, and BM25 memory) is only
known after the build. If it lands far above ~150 k sentences, `--n-docs` should come
down rather than the index design changing.
**Settle by:** reading `manifest.json → corpus.n_sentences` after the build.

### OQ-023 — Nothing computes the official FEVER score [deferred]
`meta.evidence_groups` preserves what is needed (label correct AND at least one
complete evidence group recovered), but no code reads it yet. Recall@k over flattened
gold is a *different, easier* measure and must not be reported as the FEVER score.
**Settle by:** whenever evaluation happens — explicitly out of scope for this build.

### OQ-024 — NEI examples have no annotated evidence, so the oracle is empty for them [assumed]
For a third of the sample, "run with gold evidence substituted" means "run with no
evidence". That is faithful to FEVER's annotation, and it is also exactly the
retrieval-failure-vs-genuine-insufficiency boundary: whether the corpus contains
sentences that *would* settle an NEI claim is unknown and unannotated.
**Settle by:** it is a research question, not a build decision. Flagged for the backlog.

### OQ-025 — `spacy` still not installed [deferred, blocks step B]
Adding it to the resolve made pip backtrack >15 min without converging. It is needed
for `SpacySentenceExtractor`. Step B opens with its own install pass; if it still will
not co-resolve, the fallback is `spacy.blank("en")` with only the `sentencizer`, which
needs no model download at the cost of worse boundaries on abbreviations (see OQ-012).
