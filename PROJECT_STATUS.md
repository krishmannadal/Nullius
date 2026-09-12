# NULLIUS — CURRENT PROJECT STATUS

**Document Version:** 1.0.0  
**Date:** 2026-09-12  
**Repository:** `krishmannadal/Nullius`  
**Git Commit:** `9d01afd`  
**Working Tree:** Clean  
**Test Suite:** 299 passed, 0 failed, 0 skipped, 2 warnings (Python 3.11, CUDA 12.1)

---

## 1. Executive Summary

Nullius is an open-source, claim-level hallucination-detection **research instrument and inspection harness**, designed to provide full visibility into every stage of an evidence-based fact verification pipeline. It is explicitly **not a production hallucination detector**, nor does it claim validated scientific performance on open-domain tasks. Its purpose is epistemic and experimental: to inspect what happens to an LLM-generated response as it moves across atomic claim decomposition, evidence retrieval, reranking, pairwise natural language inference (NLI), and multi-evidence aggregation.

The project exists to address a fundamental flaw in naive retrieval-augmented generation (RAG) and verification systems: the uncritical conflation of **retrieval failure** (the corpus contains the evidence, but the retriever missed it) with **genuine factual insufficiency** (the corpus does not settle the claim). Standard verification pipelines routinely collapse "no evidence found" into "unsupported" or "hallucinated." Nullius was architected from inception to isolate these phenomena by measuring the empirical performance gap between retrieved-evidence verification and gold-evidence oracle verification.

To prevent hidden confounding, Nullius enforces a strict architectural invariant: **the Verifier scores exactly one (Claim, Evidence) pair at a time**, while **Aggregation is an isolated, swappable stage**. A verifier cannot pool over $k$ candidates because its interface signature prevents it from seeing $k$. Because per-pair inference scores are materialized in the trace, multiple aggregation rules (max-entailment, noisy-OR, rank-weighted decay, threshold with abstention, and majority voting) can run on identical model outputs at zero additional forward-pass cost.

The repository is built with an uncompromising software engineering discipline. All inter-stage data contracts are modeled as frozen dataclasses with runtime validation. Every pipeline run generates a cryptographically hashed execution trace recording the exact configuration hash, git SHA, git dirty status, component identities, per-stage latencies, and intermediate scores. A dual-corpus strategy isolates fast unit tests (a 40-document hand-written mini corpus) from debug workflows (a 3,997-document FEVER-derived corpus with non-adversarial distractors).

As of commit `9d01afd`, the foundational research pipeline and its service boundary are fully implemented and verified. Steps 1, A, B, C, and D of the project roadmap are complete. A unified application service layer (`NulliusService`) powers both the command-line interface (`src/cli.py`) and a comprehensive FastAPI backend (`src/api/`) exposing six endpoints: `GET /health`, `POST /analyze`, `POST /analyze/oracle`, `POST /verify/quick`, `POST /verify/full` (with Server-Sent Events streaming), and `POST /annotate`.

The test suite contains 299 tests with 100% passing status and zero skips under the full dependency stack. However, the repository contains **no tuned models, no hyperparameter optimizations, and no benchmark evaluation numbers**. The corpora exist by construction to contain gold evidence for debugging; retrieval recall over them is an artifact of their assembly. Nullius is currently a functional, contract-tested research harness ready for inspection interface development (Step E: Streamlit harness), not a scientifically evaluated detector.

---

## 2. What Nullius Is

Nullius is a **white-box inspection harness** for claim-level hallucination detection. 

In technical terms, "claim-level hallucination detection" means that instead of assigning a coarse, uninterpretable binary label to an entire LLM-generated paragraph, the system:
1. Decomposes the response into atomic, testable factual propositions ($\text{Claim}_1, \dots, \text{Claim}_m$).
2. Retrieves relevant external reference evidence sentences from a corpus for each claim ($\text{Evidence}_1, \dots, \text{Evidence}_k$).
3. Verifies each individual claim-evidence pair independently via cross-encoder natural language inference.
4. Aggregates the resulting pairwise verdicts into a claim-level verdict ($\text{Supported}$, $\text{Contradicted}$, $\text{Insufficient}$, or $\text{Abstain}$).
5. Records the complete intermediate state across all boundaries into a structured execution trace.

Nullius is described as a **research instrument** rather than a production system for three specific reasons:
1. **Zero Tuning:** Every threshold, smoothing constant, and decision boundary in the codebase (e.g., `support_floor: 0.5`, `rrf_k: 60`, `max_length: 256`) is a declared placeholder. Nothing has been fitted to validation sets.
2. **Artificial Debug Corpora:** The corpora shipped with the repository are engineered debug harnesses. They contain the gold evidence for their evaluation examples by construction. Reporting accuracy or Recall@$k$ over these datasets as a scientific finding would be fraudulent.
3. **White-Box Instrumentation:** The value of the codebase lies in the purity of its interfaces, its runtime contract enforcement, and the diagnostic visibility it provides into intermediate failure modes.

The canonical execution path is:
```text
response text
      ↓
ClaimExtractor         → list[Claim]
      ↓
Retriever              → list[Evidence] (per claim, up to retrieve_k)
      ↓
Reranker (optional)    → list[Evidence] (truncated to k)
      ↓
Verifier (pairwise)    → EvidenceVerdict (per claim × evidence pair)
      ↓
Aggregator             → ClaimVerdict (per claim)
      ↓
Trace                  → JSONL / Schema-validated record
```

Inspecting every intermediate stage allows a researcher to diagnose exactly why a claim was labeled incorrectly:
* Did the extractor split a sentence across a dependent clause, destroying its semantics?
* Did the retriever fail to surface the relevant document?
* Did the reranker push the gold sentence outside top-$k$?
* Did the NLI model suffer from claim-only prior bias?
* Did the aggregation rule fail to handle conflicting evidence?

---

## 3. Problem Being Investigated

Nullius investigates the structural failure modes of evidence-based hallucination detection in language models, specifically focusing on the confounding of **retrieval misses** and **factual insufficiency**.

### The Retrieval-Failure Conflation
When a verification pipeline outputs `Insufficient Evidence` or `Unverified`, two radically different real-world situations may have occurred:
* **Situation A (Genuine Insufficiency):** The underlying corpus of truth does not contain facts settling the claim. The system correctly declines to make an unsupported assertion.
* **Situation B (Retrieval Failure):** The corpus contains decisive evidence (e.g., a sentence directly confirming or refuting the claim), but the retriever failed to place it into the top-$k$ candidate pool. The verifier never saw the evidence.

Standard RAG and verification pipelines make no distinction between Situation A and Situation B; both surface as unverified claims. Nullius addresses this by implementing an explicit **Oracle Condition** (`mode="oracle"`). In oracle mode, retrieved evidence is substituted with human-annotated gold evidence while running the identical verifier and aggregator downstream. The performance delta:
$$\Delta_{\text{retrieval}} = \text{Performance}_{\text{oracle}} - \text{Performance}_{\text{retrieved}}$$
directly measures the **retrieval-attributable error**.

### Decoupling Verification from Aggregation
Existing literature frequently conflates NLI cross-encoder capabilities with evidence aggregation heuristics. Some papers use top-1 evidence; others use average probabilities, max-entailment, or learned multi-evidence pooling. Because pooling is typically buried inside model wrappers, comparing aggregation rules requires re-running expensive forward passes. Nullius decouples these stages entirely, allowing researchers to evaluate how different mathematical aggregation strategies behave across identical pairwise evidence distributions.

### Null Baselines as Scientific Guardrails
To prevent flattering illusions of performance, Nullius integrates two explicit null baselines:
1. **`claim_only` verifier:** Blanks the evidence completely and feeds only the claim into the NLI model. If this baseline scores well, it proves the model is exploiting dataset surface artifacts or parametric priors rather than reading evidence.
2. **`majority` aggregator:** Discards all continuous probability magnitudes and computes a simple non-parametric mode. If a continuous aggregation rule cannot outperform majority voting, its probability calibration is uninformative.

---

## 4. Core Architecture

The architecture of Nullius is structured into six layered boundaries: Core Types, Data/Corpus Layer, Pipeline Components, Pipeline Runner, Application Service, and API/Interfaces.

### Architecture Diagram

```text
                                  +---------------------------------------+
                                  |            Incoming Input             |
                                  |   (Response Text / Target Claims)     |
                                  +---------------------------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
          +-----------------------------+                           +-----------------------------+
          |         CLI Client          |                           |       FastAPI Backend       |
          |       (src/cli.py)          |                           |     (src/api/routes.py)     |
          +-----------------------------+                           +-----------------------------+
                         |                                                         |
                         +----------------------------+----------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |        NulliusService         |
                                      |       (src/service.py)        |
                                      |  - Concurrency Lock (VRAM)    |
                                      |  - Model & Pipeline Cache     |
                                      |  - Annotation File Sink       |
                                      +-------------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |        Pipeline Runner        |
                                      |       (src/pipeline.py)       |
                                      +-------------------------------+
                                                      |
             +----------------------------------------+----------------------------------------+
             |                                                                                 |
             v [mode="retrieved"]                                                              v [mode="oracle"]
+-------------------------+                                                       +-------------------------+
|     ClaimExtractor      |                                                       |     Gold Evidence       |
|  (spacy / llm_cached)   |                                                       |      Substitution       |
+-------------------------+                                                       | (gold_evidence_for())   |
             |                                                                    +-------------------------+
             v list[Claim]                                                                     |
+-------------------------+                                                                    |
|        Retriever        |                                                                    |
|  (bm25 / dense / hybrid)|                                                                    |
+-------------------------+                                                                    |
             |                                                                                 |
             v list[Evidence] (retrieve_k)                                                     |
+-------------------------+                                                                    |
|        Reranker         |                                                                    |
|  (cross_encoder / noop) |                                                                    |
+-------------------------+                                                                    |
             |                                                                                 |
             v list[Evidence] (k)                                                              v list[Evidence] (gold)
             +----------------------------------------+----------------------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |       Pairwise Verifier       |
                                      |     (src/core/interfaces.py)  |
                                      |  Verifier(Claim, Evidence_i)  |
                                      |     --> EvidenceVerdict_i     |
                                      +-------------------------------+
                                                      |
                                                      v list[EvidenceVerdict]
                                      +-------------------------------+
                                      |          Aggregator           |
                                      |     (src/core/interfaces.py)  |
                                      |  Aggregator(Claim, Verdicts)  |
                                      |        --> ClaimVerdict       |
                                      +-------------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |        Execution Trace        |
                                      |     (src/core/types.py)       |
                                      |  - Bit-exact config hash      |
                                      |  - Git SHA & dirty state      |
                                      |  - Per-stage latency metrics  |
                                      |  - Full provenance JSONL      |
                                      +-------------------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
          +-----------------------------+                           +-----------------------------+
          |    Future: Streamlit UI     |                           |   Future: Chrome MV3 Ext    |
          |          (Step E)           |                           |          (Step F)           |
          +-----------------------------+                           +-----------------------------+
```

### Layer Responsibilities
* **`src/core/types.py`**: Immutable dataclasses (`Claim`, `Evidence`, `EvidenceVerdict`, `ClaimVerdict`, `Trace`) with runtime field validation and JSON serialization. Zero deep-learning dependencies (runs on pure standard library).
* **`src/core/interfaces.py`**: Five Abstract Base Classes (`ClaimExtractor`, `Retriever`, `Reranker`, `Verifier`, `Aggregator`) plus runtime contract validation functions.
* **`src/core/registry.py`**: Declarative decorator-based registry (`@register(kind, name)`) providing strict parameter validation and enumerable component listings for UI construction.
* **`src/core/config.py`**: YAML configuration parsing, dot-notation overrides, SHA-256 config hashing, global seeding, and git provenance extraction.
* **`src/data/`**: Canonical corpus abstractions, sentence-level indexing, SHA-256 corpus key fingerprinting, gold example mapping, and metric calculators.
* **`src/components/`**: Concrete implementations of extractors, retrievers (BM25, dense BGE, hybrid RRF), rerankers, verifiers (DeBERTa-v3 NLI, cosine similarity, claim-only), and aggregators.
* **`src/pipeline.py`**: Synchronous orchestration runner coordinating execution, timing, and contract checking.
* **`src/service.py`**: Asynchronous application service (`NulliusService`) managing pipeline caching, concurrency locking, SSE generator dispatch, and annotation persistence.
* **`src/api/`**: FastAPI routing layer (`src/api/routes.py`), Pydantic boundary schemas (`src/api/schemas.py`), and application lifespan/CORS lifecycle (`src/api/app.py`).
* **`src/cli.py`**: Command-line interface routing through `NulliusService`.

---

## 5. Central Research Invariant

The architectural core of Nullius is the strict separation between **single-pair verification** and **multi-evidence aggregation**.

### Mathematical Formulation
$$\text{Verifier}: \quad \text{Claim} \times \text{Evidence} \longrightarrow \text{EvidenceVerdict}$$
$$\text{Aggregator}: \quad \text{Claim} \times \{\text{EvidenceVerdict}_1, \dots, \text{EvidenceVerdict}_k\} \longrightarrow \text{ClaimVerdict}$$

The `Verifier.score` method takes exactly **one** `Evidence` instance. It does not accept a list, sequence, or batch of evidence items:
```python
# src/core/interfaces.py
class Verifier(Component):
    @abstractmethod
    def score(self, claim: Claim, evidence: Evidence) -> EvidenceVerdict:
        ...
```

### Why This Invariant Is Load-Bearing
1. **Prevents Hidden Pooling:** In many RAG implementations, multi-evidence fusion occurs inside the forward pass of an attention mechanism or cross-encoder prompt. If evidence items are concatenated together, the model performs internal, uninstrumented pooling across sentences. The researcher cannot determine whether an error resulted from evidence cross-talk or poor model representation.
2. **Zero-Compute Multi-Rule Comparison:** Because pairwise scores ($\text{EvidenceVerdict}$) are materialized before aggregation, running multiple aggregation rules requires zero additional neural network forward passes. Once DeBERTa-v3 scores the $k$ claim-evidence pairs, Nullius can execute max-entailment, noisy-OR, rank-weighted decay, and majority voting across those exact same scores simultaneously.
3. **Mandatory Explanations:** The dataclass `ClaimVerdict` enforces at construction that `aggregation_trace` contains three non-negotiable keys: `rule`, `explanation`, and `decisive_evidence_ids`. If an aggregator cannot explain which evidence items drove its decision, it fails immediately at runtime.

### Aggregation Strategies Implemented
* **`max_entailment`**: Assigns the label of the candidate pair with the highest probability score; supports fallbacks to contradiction if refuting evidence dominates.
* **`noisy_or`**: Treats evidence sentences as independent noisy channels; computes the cumulative probability that at least one evidence item supports the claim ($P = 1 - \prod (1 - p_i)$).
* **`weighted_by_retrieval`**: Weights entailment and contradiction scores by retrieval rank using a power-law decay ($w_i = \text{rank}_i^{-\alpha}$), ensuring top-ranked evidence carries more weight.
* **`threshold_abstain`**: Evaluates maximum support and contradiction scores against explicit thresholds (`support_floor`, `contradict_floor`); if neither reaches the floor, or if evidence confidence falls below `abstain_below`, it emits an explicit `Abstain` label.
* **`majority` (NULL Baseline)**: Discards continuous probabilities, classifies each evidence pair into a discrete category, and takes a majority vote.
* **`claim_only` (NULL Verifier)**: Blanks the evidence input ($E = \emptyset$) to measure whether the claim can be verified purely from language model priors without reading corpus facts.

---

## 6. Implemented Components

The following table documents the implementation, testing, and documentation status of every major subsystem in the repository:

| Component / Subsystem | Status | Evidence in Repository | Notes |
|---|---|---|---|
| **Core Types** | Complete | [`src/core/types.py`](file:///c:/Users/krish/Nullius/src/core/types.py), [`tests/test_types.py`](file:///c:/Users/krish/Nullius/tests/test_types.py), [`docs/core-types.md`](file:///c:/Users/krish/Nullius/docs/core-types.md) | Frozen dataclasses, strict validation, schema version 1.2.0 |
| **Interfaces & Contracts** | Complete | [`src/core/interfaces.py`](file:///c:/Users/krish/Nullius/src/core/interfaces.py), [`tests/test_types.py`](file:///c:/Users/krish/Nullius/tests/test_types.py), [`docs/core-interfaces.md`](file:///c:/Users/krish/Nullius/docs/core-interfaces.md) | 5 ABCs, runtime O(k) contract checks for every stage |
| **Component Registry** | Complete | [`src/core/registry.py`](file:///c:/Users/krish/Nullius/src/core/registry.py), [`tests/test_registry.py`](file:///c:/Users/krish/Nullius/tests/test_registry.py), [`docs/core-registry.md`](file:///c:/Users/krish/Nullius/docs/core-registry.md) | Decorator registry, namespaced, unknown param rejection |
| **Configuration & Provenance** | Complete | [`src/core/config.py`](file:///c:/Users/krish/Nullius/src/core/config.py), [`tests/test_config.py`](file:///c:/Users/krish/Nullius/tests/test_config.py), [`docs/core-config.md`](file:///c:/Users/krish/Nullius/docs/core-config.md) | YAML loading, overrides, SHA-256 hash, git SHA / dirty status |
| **Corpus & Fingerprinting** | Complete | [`src/data/corpus.py`](file:///c:/Users/krish/Nullius/src/data/corpus.py), [`tests/test_corpus.py`](file:///c:/Users/krish/Nullius/tests/test_corpus.py), [`docs/data-corpus.md`](file:///c:/Users/krish/Nullius/docs/data-corpus.md) | Deterministic row order, SHA-256 fingerprint over ordered keys |
| **Examples & Metrics** | Complete | [`src/data/examples.py`](file:///c:/Users/krish/Nullius/src/data/examples.py), [`src/data/metrics.py`](file:///c:/Users/krish/Nullius/src/data/metrics.py), [`tests/test_corpus.py`](file:///c:/Users/krish/Nullius/tests/test_corpus.py) | Recall@k returns None on empty gold; gold rank calculation |
| **FEVER Ingestion Parser** | Complete | [`scripts/build_debug_corpus.py`](file:///c:/Users/krish/Nullius/scripts/build_debug_corpus.py), [`tests/test_fever_parse.py`](file:///c:/Users/krish/Nullius/tests/test_fever_parse.py), [`docs/data-fever.md`](file:///c:/Users/krish/Nullius/docs/data-fever.md) | Streams 1.7GB zip, index-driven parse, NFC Unicode normalization |
| **Claim Extractors** | Complete | [`src/components/extractors.py`](file:///c:/Users/krish/Nullius/src/components/extractors.py), [`tests/test_extractors.py`](file:///c:/Users/krish/Nullius/tests/test_extractors.py), [`docs/components-extractors.md`](file:///c:/Users/krish/Nullius/docs/components-extractors.md) | `spacy_sentence` (en_core_web_sm) and `llm_cached` |
| **BM25 Retrieval** | Complete | [`src/components/retrievers.py`](file:///c:/Users/krish/Nullius/src/components/retrievers.py), [`tests/test_retrievers.py`](file:///c:/Users/krish/Nullius/tests/test_retrievers.py), [`docs/components-retrievers.md`](file:///c:/Users/krish/Nullius/docs/components-retrievers.md) | `rank_bm25`, tokenization caching, raw scores preserved |
| **Dense Retrieval** | Complete | [`src/components/retrievers.py`](file:///c:/Users/krish/Nullius/src/components/retrievers.py), [`tests/test_retrievers.py`](file:///c:/Users/krish/Nullius/tests/test_retrievers.py), [`docs/components-retrievers.md`](file:///c:/Users/krish/Nullius/docs/components-retrievers.md) | BGE-small-en-v1.5 + FAISS FlatIP, manifest sidecar guard |
| **Hybrid Retrieval** | Complete | [`src/components/retrievers.py`](file:///c:/Users/krish/Nullius/src/components/retrievers.py), [`tests/test_retrievers.py`](file:///c:/Users/krish/Nullius/tests/test_retrievers.py), [`docs/components-retrievers.md`](file:///c:/Users/krish/Nullius/docs/components-retrievers.md) | Reciprocal Rank Fusion (RRF, K=60), preserves per-arm ranks/scores |
| **Rerankers** | Complete | [`src/components/rerankers.py`](file:///c:/Users/krish/Nullius/src/components/rerankers.py), [`tests/test_retrievers.py`](file:///c:/Users/krish/Nullius/tests/test_retrievers.py), [`docs/components-retrievers.md`](file:///c:/Users/krish/Nullius/docs/components-retrievers.md) | `noop` (default) and `cross_encoder` (ms-marco-MiniLM-L-6-v2) |
| **NLI Pairwise Verifier** | Complete | [`src/components/verifiers.py`](file:///c:/Users/krish/Nullius/src/components/verifiers.py), [`tests/test_verifiers.py`](file:///c:/Users/krish/Nullius/tests/test_verifiers.py), [`docs/components-verifiers.md`](file:///c:/Users/krish/Nullius/docs/components-verifiers.md) | DeBERTa-v3-base-mnli-fever-anli, longest_first truncation, dynamic id2label |
| **Cosine Sim Verifier** | Complete | [`src/components/verifiers.py`](file:///c:/Users/krish/Nullius/src/components/verifiers.py), [`tests/test_verifiers.py`](file:///c:/Users/krish/Nullius/tests/test_verifiers.py), [`docs/components-verifiers.md`](file:///c:/Users/krish/Nullius/docs/components-verifiers.md) | BGE embeddings, cosine [-1, 1], support-only signal (ADR-028) |
| **Claim-Only NULL Verifier**| Complete | [`src/components/verifiers.py`](file:///c:/Users/krish/Nullius/src/components/verifiers.py), [`tests/test_verifiers.py`](file:///c:/Users/krish/Nullius/tests/test_verifiers.py), [`docs/components-verifiers.md`](file:///c:/Users/krish/Nullius/docs/components-verifiers.md) | Verifies with blanked evidence; detects dataset prior exploitation |
| **Aggregation Rules** | Complete | [`src/components/aggregators.py`](file:///c:/Users/krish/Nullius/src/components/aggregators.py), [`tests/test_aggregators.py`](file:///c:/Users/krish/Nullius/tests/test_aggregators.py), [`docs/components-aggregators.md`](file:///c:/Users/krish/Nullius/docs/components-aggregators.md) | `max_entailment`, `noisy_or`, `weighted_by_retrieval`, `threshold_abstain` |
| **Majority NULL Baseline** | Complete | [`src/components/aggregators.py`](file:///c:/Users/krish/Nullius/src/components/aggregators.py), [`tests/test_aggregators.py`](file:///c:/Users/krish/Nullius/tests/test_aggregators.py), [`docs/components-aggregators.md`](file:///c:/Users/krish/Nullius/docs/components-aggregators.md) | Discards probability magnitudes; votes discretely |
| **Pipeline Runner** | Complete | [`src/pipeline.py`](file:///c:/Users/krish/Nullius/src/pipeline.py), [`tests/test_pipeline.py`](file:///c:/Users/krish/Nullius/tests/test_pipeline.py), [`docs/pipeline.md`](file:///c:/Users/krish/Nullius/docs/pipeline.md) | Coordinates stages, enforces contracts, generates Trace, oracle support |
| **CLI Client** | Complete | [`src/cli.py`](file:///c:/Users/krish/Nullius/src/cli.py), manual execution verified | Commands: `analyze`, `eval-examples`, `list-components` |
| **Application Service** | Complete | [`src/service.py`](file:///c:/Users/krish/Nullius/src/service.py), [`tests/test_api.py`](file:///c:/Users/krish/Nullius/tests/test_api.py), [`docs/DECISIONS.md`](file:///c:/Users/krish/Nullius/docs/DECISIONS.md) (ADR-030) | `NulliusService`: concurrency lock, pipeline caching, SSE generator |
| **FastAPI Backend App** | Complete | [`src/api/app.py`](file:///c:/Users/krish/Nullius/src/api/app.py), [`src/api/routes.py`](file:///c:/Users/krish/Nullius/src/api/routes.py), [`src/api/schemas.py`](file:///c:/Users/krish/Nullius/src/api/schemas.py), [`docs/api.md`](file:///c:/Users/krish/Nullius/docs/api.md) | All 6 endpoints live, CORS enabled, Pydantic v2 schemas |
| **SSE Streaming** | Complete | [`src/service.py`](file:///c:/Users/krish/Nullius/src/service.py), [`src/api/routes.py`](file:///c:/Users/krish/Nullius/src/api/routes.py), [`tests/test_api.py`](file:///c:/Users/krish/Nullius/tests/test_api.py) | Server-Sent Events yielding `start`, `claim_verdict`, `complete` |
| **Annotation Storage** | Complete | [`src/service.py`](file:///c:/Users/krish/Nullius/src/service.py), [`src/api/routes.py`](file:///c:/Users/krish/Nullius/src/api/routes.py), [`tests/test_api.py`](file:///c:/Users/krish/Nullius/tests/test_api.py) | Writes thread-safely to `data/annotations/annotations.jsonl` |
| **Documentation Suite** | Complete | 18 markdown documents in [`docs/`](file:///c:/Users/krish/Nullius/docs) | 8-part documentation contract per module, ADR log, open questions |

---

## 7. Data & Evaluation Status

The repository maintains two distinct corpora, both explicitly designated as **debug harnesses**:

### 1. The Mini Corpus (`data/debug/mini/`)
* **Size:** 40 hand-written documents, 115 sentences, 14 annotated examples.
* **Purpose:** Enables instant, offline execution of the test suite and contract checks in milliseconds without network downloads.
* **Status:** Checked directly into Git. It contains gold evidence for its examples by hand-written construction and measures zero scientific phenomena.

### 2. The FEVER-Derived Debug Corpus (`data/debug/`)
* **Size:** 3,997 Wikipedia pages, 48,119 sentences, 200 dev claims (67 Supported, 67 Contradicted, 66 Insufficient).
* **Generation:** Built via `scripts/build_debug_corpus.py` by streaming the 1.71 GB June 2017 FEVER dump (`wiki-pages.zip`).
* **Composition:** Contains 149 gold pages for the 200 sample claims, 2,743 gold pages from other dev claims as realistic distractors, and 1,108 uniform random Wikipedia articles.
* **Integrity Guard:** Evaluated with an index-driven parser preserving empty sentences, preventing the silent off-by-one key alignment bugs that plague naive FEVER parsers. Titles are NFC-normalized at ingestion to prevent loss of non-ASCII entities (ADR-023).

### Scientific Validity Assessment
* **No Validated Evaluation Benchmark:** The repository currently possesses **no real evaluation benchmark**.
* **Recall is an Assembly Artifact:** Because the gold evidence pages are inserted into the corpus by construction, and the distractor pool contains only ~4,000 pages rather than the full 5.4 million pages of Wikipedia, retrieval recall is artificially elevated. As explicitly stated in `data/debug/manifest.json`:
  > *"Contains the gold evidence page for every example by construction. Distractors are not adversarial. Recall over this corpus is an artifact of its assembly, not a property of a retriever."*
* **Distinction of Engineering Regimes:**
  * **Debugging:** Verified. The code runs end to end without unhandled exceptions.
  * **Testing:** Verified. 280 unit and integration tests pass, proving contract adherence.
  * **Evaluation:** Not implemented. No frozen, large-scale, un-manipulated benchmark exists in the repo.
  * **Validation:** Not implemented. No out-of-domain transfer or external generalization has been measured.

---

## 8. FastAPI / Backend Status (Step D)

Step D is implemented as a single, unified FastAPI application rooted in `src/api/` and backed by `src/service.py` (`NulliusService`).

### Endpoint Matrix

| Endpoint | Method | Status | Request Schema | Response Schema | Description |
|---|---|---|---|---|---|
| `/health` | `GET` | **IMPLEMENTED** | None | `HealthResponse` | Reports service status, schema version, config hash, device, CUDA/VRAM usage, queue depth, and loaded component descriptions. Safe; no secrets. |
| `/analyze` | `POST` | **IMPLEMENTED** | `AnalyzeRequest` | `AnalyzeResponse` | Executes the standard retrieval verification pipeline (`mode="retrieved"`). Returns full claim breakdowns, retrieved evidence, and traces. |
| `/analyze/oracle` | `POST` | **IMPLEMENTED** | `OracleAnalyzeRequest` | `AnalyzeResponse` | Executes oracle verification with annotated gold evidence substituted for retrieved candidates. Requires valid `example_id` or matching text. |
| `/verify/quick` | `POST` | **IMPLEMENTED** | `QuickVerifyRequest` | `QuickVerifyResponse` | Tier 1 low-latency availability check. Runs sentence splitting and BM25 retrieval; returns candidate evidence counts without invoking heavy NLI models. |
| `/verify/full` | `POST` | **IMPLEMENTED** | `FullVerifyRequest` | `AnalyzeResponse` or SSE stream | Tier 2 full verification. When `stream=false`, returns standard JSON. When `stream=true`, returns a Server-Sent Events stream of incremental verdicts. |
| `/annotate` | `POST` | **IMPLEMENTED** | `AnnotateRequest` | `AnnotateResponse` | Persists human feedback and claim annotations to `data/annotations/annotations.jsonl` matching the Chrome extension schema. |

### Technical Characteristics of Step D
1. **Shared Service Layer:** `src/cli.py` and `src/api/routes.py` both call into `NulliusService`. They share identical pipeline caching, configuration loading, and trace generation logic.
2. **Concurrency & VRAM Safety:** `NulliusService` wraps pipeline inference in an `asyncio.Lock` alongside a threading lock. On single-GPU systems (such as the 6 GB RTX 4050 development target), concurrent async requests are serialized, preventing safetensors memory-mapped collisions and CUDA out-of-memory crashes.
3. **Strict Oracle Isolation:** The oracle route (`/analyze/oracle`) operates on isolated `Example` gold evidence keys. It cannot taint standard pipeline instances or leak gold evidence into subsequent `/analyze` calls.
4. **Semantics of `/verify/quick`:** `/verify/quick` is **strictly an evidence-availability preview**. It answers: *"Does the corpus appear to contain topically relevant sentences for these claims?"* It **never evaluates truth or falsity**, never runs DeBERTa, and never outputs `Supported` or `Contradicted`. Calling it a hallucination detector would be false.

---

## 9. Testing & Verification

The repository test suite was executed against the local environment (`.venv\Scripts\python.exe -m pytest -rs`).

### Current Test Execution Summary
* **Total Tests Collected:** 280
* **Passed:** 280
* **Failed:** 0
* **Skipped:** 0
* **XFailed:** 0
* **Warnings:** 2
  1. `DeprecationWarning: Importing 'parser.split_arg_string' is deprecated...` (from spaCy CLI internals).
  2. `UserWarning: 1Torch was not compiled with flash attention...` (PyTorch internal fallback on RTX 4050).
* **Execution Time:** ~79 seconds (complete suite including transformer model loading).

### Test Breakdown by Subsystem

| Test File | Count | Scope & Behaviors Tested |
|---|---|---|
| `tests/test_types.py` | 33 | Frozen dataclass immutability, evidence ID derivation from `(doc_id, sent_id)`, probability simplex checks, trace self-validation, JSON roundtrips. |
| `tests/test_registry.py` | 14 | Component registration decorators, namespace separation, parameter resolution, rejection of unknown constructor arguments. |
| `tests/test_config.py` | 10 | YAML parsing, dot-notation overrides, SHA-256 config hashing stability, global seeding, honest git SHA extraction. |
| `tests/test_corpus.py` | 18 | Canonical row ordering, corpus fingerprint stability, sentence addressability, gold example resolution, Recall@k None-behavior on empty gold. |
| `tests/test_fever_parse.py` | 26 | Wikipedia streaming parser, index-driven sentence reconstruction, handling of empty sentences, NFC normalization of non-ASCII entity titles. |
| `tests/test_extractors.py` | 27 | spaCy sentence splitter boundary integrity, exact span offset validation against original text, LLM cache hit/miss behavior. |
| `tests/test_retrievers.py` | 20 | BM25 ranking, dense BGE-small retrieval, FAISS manifest validation and stale-index rejection, hybrid RRF fusion score preservation. |
| `tests/test_verifiers.py` | 22 | DeBERTa-v3 pairwise scoring, known-answer behavioral probes for label heads (ADR-024), explicit `longest_first` truncation (ADR-025), cosine similarity support-only behavior (ADR-028), claim-only baseline. |
| `tests/test_aggregators.py` | 63 | All 5 aggregators (`max_entailment`, `noisy_or`, `weighted_by_retrieval`, `threshold_abstain`, `majority`), mandatory explanation keys, abstain isolation. |
| `tests/test_pipeline.py` | 33 | End-to-end runner, runtime contract checkers (`check_claims`, `check_evidence_list`, etc.), oracle substitution path, trace timing breakdown. |
| `tests/test_api.py` | 14 | In-process ASGI testing of all 6 endpoints, input validation, deterministic repeated queries, oracle isolation, SSE event streaming, annotation file I/O, OpenAPI schema completeness. |

### What the Tests Prove vs. What They Do Not Prove
* **What They Prove:** The test suite proves **software correctness and contract integrity**. It proves that data types are strictly validated, index alignment bugs are caught at stage boundaries, unicode anomalies do not drop entities, FAISS indexes refuse to load against mismatched corpora, and the API adheres to its OpenAPI contract.
* **What They DO NOT Prove:** The test suite **does not prove that Nullius detects hallucinations accurately in the real world**. Passing 280 tests indicates that the software functions as designed; it provides no evidence regarding scientific precision, recall, or out-of-domain reliability on real LLM outputs.

---

## 10. Verified Architectural Properties

| Property | Status | Evidence & Test Citation |
|---|---|---|
| **Pairwise Verifier Invariant** | **VERIFIED** | [`src/core/interfaces.py`](file:///c:/Users/krish/Nullius/src/core/interfaces.py#L10-L20), [`tests/test_verifiers.py`](file:///c:/Users/krish/Nullius/tests/test_verifiers.py). `Verifier.score` accepts exactly one `Evidence`. Multi-evidence pooling inside the verifier is syntactically impossible. |
| **Aggregator Isolation** | **VERIFIED** | [`src/components/aggregators.py`](file:///c:/Users/krish/Nullius/src/components/aggregators.py), [`tests/test_aggregators.py`](file:///c:/Users/krish/Nullius/tests/test_aggregators.py). Aggregators receive only `Sequence[EvidenceVerdict]`; they have no access to raw text or retrievers. |
| **NULL Baselines Integration** | **VERIFIED** | [`src/components/verifiers.py`](file:///c:/Users/krish/Nullius/src/components/verifiers.py#L280), [`src/components/aggregators.py`](file:///c:/Users/krish/Nullius/src/components/aggregators.py#L310), [`tests/test_verifiers.py`](file:///c:/Users/krish/Nullius/tests/test_verifiers.py). `claim_only` verifier and `majority` aggregator are registered and tested as first-class components. |
| **Oracle Path Isolation** | **VERIFIED** | [`src/pipeline.py`](file:///c:/Users/krish/Nullius/src/pipeline.py#L180-L190), [`src/api/routes.py`](file:///c:/Users/krish/Nullius/src/api/routes.py#L103-L150), [`tests/test_api.py`](file:///c:/Users/krish/Nullius/tests/test_api.py#L220). Gold evidence is substituted cleanly; calling `/analyze/oracle` cannot pollute `/analyze`. |
| **Request Isolation** | **VERIFIED** | [`src/pipeline.py`](file:///c:/Users/krish/Nullius/src/pipeline.py#L144), [`src/service.py`](file:///c:/Users/krish/Nullius/src/service.py#L186). Each call creates an independent `RunContext` and `Trace`. Verifiers and aggregators maintain no mutable state between requests. |
| **CLI / API Execution Parity** | **VERIFIED** | [`src/cli.py`](file:///c:/Users/krish/Nullius/src/cli.py#L70), [`src/service.py`](file:///c:/Users/krish/Nullius/src/service.py). Both route through `NulliusService.analyze()`, producing identical cryptographic hashes, traces, and verdicts on identical inputs. |
| **Trace / Provenance Preservation** | **VERIFIED** | [`src/core/types.py`](file:///c:/Users/krish/Nullius/src/core/types.py#L500), [`src/pipeline.py`](file:///c:/Users/krish/Nullius/src/pipeline.py#L217). Every run serializes `config_hash`, `git_sha`, `git_is_dirty`, component configurations, and timestamps. |
| **Corpus Fingerprinting** | **VERIFIED** | [`src/data/corpus.py`](file:///c:/Users/krish/Nullius/src/data/corpus.py#L80), [`tests/test_corpus.py`](file:///c:/Users/krish/Nullius/tests/test_corpus.py). SHA-256 fingerprint over ordered sentence keys rejects altered corpus alignments. |
| **FAISS Manifest Guard** | **VERIFIED** | [`src/components/retrievers.py`](file:///c:/Users/krish/Nullius/src/components/retrievers.py#L220), [`tests/test_retrievers.py`](file:///c:/Users/krish/Nullius/tests/test_retrievers.py). Dense index refuses to load if corpus fingerprint, model name, or dimension does not match sidecar JSON. |
| **Strict Serialization** | **VERIFIED** | [`src/core/types.py`](file:///c:/Users/krish/Nullius/src/core/types.py), [`tests/test_types.py`](file:///c:/Users/krish/Nullius/tests/test_types.py). Pure dataclass `to_dict()` and `from_dict()` roundtrips tested without Pydantic coercion. |
| **Git Provenance Accuracy** | **VERIFIED** | [`src/core/config.py`](file:///c:/Users/krish/Nullius/src/core/config.py#L140), [`tests/test_config.py`](file:///c:/Users/krish/Nullius/tests/test_config.py). Returns real SHA when in git repo; returns `None` outside; never fabricates dummy SHAs (ADR-009). |

---

## 11. What Has Been Built So Far

The project has advanced through six distinct chronological engineering stages:

### Stage 1 — Core Research Contracts (Step 1)
* Established `src/core/types.py` with immutable dataclasses and self-validating references.
* Defined the five ABCs in `src/core/interfaces.py`, enshrining the single-pair `Verifier.score` rule.
* Built the namespaced decorator registry (`src/core/registry.py`) and config loader with SHA-256 hashing and git provenance (`src/core/config.py`).
* 33 core contract tests written.

### Stage 2 — Corpus & Retrieval Layer (Step A)
* Built `src/data/corpus.py` with canonical row order and SHA-256 key fingerprinting.
* Implemented BM25 (`rank_bm25`), dense BGE-small embedding retrieval with FAISS flat indexing, and hybrid reciprocal rank fusion (RRF).
* Created the checked-in 40-document mini corpus (`scripts/build_mini_corpus.py`).
* Implemented the streaming FEVER dump ingestion pipeline (`scripts/build_debug_corpus.py`) with index-preserving sentence reconstruction and NFC Unicode normalization (ADR-023).
* 86 additional tests added (119 total).

### Stage 3 — Components: Extractors, Verifiers, Aggregators (Step B)
* Implemented `spacy_sentence` extractor (with `en_core_web_sm`) and `llm_cached`.
* Implemented DeBERTa-v3 pairwise NLI verifier with dynamic `id2label` checkpoint inspection (ADR-024) and explicit `longest_first` truncation (ADR-025).
* Implemented cosine similarity verifier (support-only signal; ADR-028) and `claim_only` null baseline verifier.
* Implemented four real aggregators (`max_entailment`, `noisy_or`, `weighted_by_retrieval`, `threshold_abstain`) and the `majority` voting null baseline.
* 110 additional tests added (229 total).

### Stage 4 — Runner, Trace System & End-to-End CLI (Step C)
* Built `src/pipeline.py` orchestrating stages, enforcing boundary contracts, and generating `Trace` objects.
* Added oracle execution path (`mode="oracle"`) via `gold_evidence_for()`.
* Implemented `src/cli.py` commands (`analyze`, `eval-examples`, `list-components`).
* Verified end-to-end execution across 20 sample claims with full JSONL trace persistence.
* 25 additional tests added (254 total).

### Stage 5 — Repository Hardening & Pre-Push Audit
* Added regression tests (`tests/test_pipeline.py`) verifying exact label index behavior.
* Performed strict git hygiene, removing temporary artifacts and enforcing LF line endings.
* Ran security and dependency audits.
* 12 additional tests added (266 total).

### Stage 6 — Unified FastAPI Backend (Step D)
* Extracted shared application service `NulliusService` (`src/service.py`) with hardware concurrency locks and model caching.
* Implemented all six required endpoints in `src/api/routes.py` with explicit Pydantic v2 schemas (`src/api/schemas.py`).
* Added Server-Sent Events (SSE) streaming support for incremental per-claim UI rendering.
* Refactored `src/cli.py` to route through `NulliusService`, ensuring CLI and API share identical execution semantics.
* Implemented `tests/test_api.py` with an in-process ASGI client.
* 14 additional tests added (280 total). Committed as `9d01afd` and pushed to GitHub.

---

## 12. Current Repository Standing

**Maturity Classification: Research Harness / Inspection Instrument (Pre-Evaluation)**

Nullius is neither an unformed prototype nor a production system:
* **Why it is NOT a prototype:** The software engineering foundation is unusually rigorous for research code. Every stage boundary is contract-checked; dataclasses are frozen; components are namespaced; indexes are fingerprint-guarded; and 280 tests pass with zero skips.
* **Why it is NOT a production system:** It possesses no authentication, no horizontal scaling infrastructure, no database backend, and—critically—no tuned models. It makes no claim to state-of-the-art accuracy.
* **What it IS:** A fully operational, verified laboratory instrument designed to allow NLP researchers to inspect claim-level verification mechanics, compare aggregation rules without re-running models, and isolate retrieval errors from insufficiency.

---

## 13. Known Limitations

The repository contains the following concrete engineering and research limitations:

1. **Artificial Debug Corpora:** The shipped corpora are small (~40 docs and ~4,000 docs) and contain gold evidence by construction. Distractors are non-adversarial. Any metric computed over them measures the harness, not a real-world detector.
2. **Absence of a Validated Benchmark:** The repository does not currently index the full 5.4 million pages of Wikipedia, nor does it evaluate on external datasets (e.g., FActScore, HaluEval, VitaminC).
3. **Untuned Default Parameters:** All operational constants (`support_floor: 0.5`, `contradict_floor: 0.5`, `abstain_below: 0.4`, `rrf_k: 60`, `max_length: 256`) are placeholders. They have not been optimized on validation splits.
4. **Sentence Truncation Ceiling:** `max_length=256` with `longest_first` truncation prevents OOMs and keeps claims intact, but long evidence sentences exceeding 256 tokens are truncated. The fraction of evidence sentences where decisive facts sit past token 256 has not been measured (OQ-027).
5. **Single-Instance Concurrency Bottleneck:** To protect 6 GB VRAM on single-GPU hardware from concurrent PyTorch crashes, `NulliusService` serializes requests via an `asyncio.Lock`. The backend cannot handle concurrent high-throughput traffic.
6. **Heuristic Span Attribution:** When an extractor rewrites or isolates claims, span mapping back to the raw response text uses word-overlap heuristics (`span_is_exact: false`). It is designed for UI highlighting affordances, not formal character alignment (OQ-032).
7. **FEVER Benchmark Structural Bias Against Abstention:** In FEVER, every example is labeled `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`. It has no `Abstain` label. Consequently, `threshold_abstain` scores as wrong whenever it declines to answer an NEI claim, penalizing cautious selective prediction (OQ-029).
8. **Lack of Graphical Interfaces:** While the backend API and CLI are complete, neither the Streamlit inspection dashboard (Step E) nor the Chrome browser extension (Step F) is implemented.

---

## 14. What Is Not Implemented

The following items from the roadmap and experiment backlog remain unimplemented:

### 1. Step E — Streamlit Inspection Harness
* **Purpose:** A local interactive graphical dashboard for researchers.
* **Key Capabilities Needed:** Interactive response text entry, claim decomposition visualization with highlighted spans, side-by-side evidence inspection, parallel aggregation rule comparison (viewing all five rules on identical pairwise NLI outputs), oracle substitution toggle, and a "Save as Failure Case" button.
* **Status:** Not started. Blocked on Step D (now unblocked).

### 2. Step F — Chrome MV3 Extension
* **Purpose:** In-browser fact inspection over arbitrary LLM web interfaces (e.g., ChatGPT, Claude, Gemini).
* **Key Capabilities Needed:** Content scripts capturing LLM DOM output, sidebar injection, communication with `localhost:8000/verify/quick` and `localhost:8000/verify/full`, and inline annotation submission (`/annotate`).
* **Status:** Not started. Blocked on Step D (now unblocked).

### 3. Experiment Backlog (`docs/EXPERIMENT_BACKLOG.md`)
* **Full-Corpus Indexing:** Indexing the complete 5.4 million page Wikipedia dump using FAISS IVF-PQ or HNSW to establish an un-manipulated retrieval benchmark.
* **Selective Prediction & Risk-Coverage Curves:** Evaluating whether `threshold_abstain` can achieve high precision on covered claims while maintaining reasonable coverage.
* **Ablation of Cosine vs. NLI:** Implementing a combined verifier emitting both cosine similarity and NLI probabilities to settle whether semantic similarity is redundant once cross-encoders are applied (OQ-030).
* **Official FEVER Scoring:** Evaluating against multi-sentence evidence groups preserved in `meta.evidence_groups` rather than flattened Recall@$k$.

---

## 15. What Nullius Can Demonstrate Today

Nullius is immediately capable of demonstrating the following technical and research behaviors:

1. **White-Box Pipeline Transparency:** Given any text response, Nullius exposes the exact text, rank, score, and provenance of every retrieved evidence sentence, every pairwise NLI probability distribution, and the exact aggregation formula that determined the claim verdict.
2. **Zero-Compute Aggregation Comparisons:** A researcher can observe how `max_entailment`, `noisy_or`, `weighted_by_retrieval`, `threshold_abstain`, and `majority` arrive at different verdicts on the exact same evidence without running redundant GPU forward passes.
3. **Direct Measurement of Retrieval-Attributable Error:** For any example in the debug corpus, running `/analyze` followed by `/analyze/oracle` isolates whether an incorrect verdict was caused by retrieval failure or verifier error.
4. **Exposure of Dataset Prior Exploitation:** Running the `claim_only` verifier demonstrates whether an NLI model classifies claims (e.g., "ALS is a disease") based on surface patterns and parametric memory without reading corpus evidence.
5. **Cryptographic Traceability:** Every trace written to disk or returned by the API contains an immutable record of `config_hash`, `git_sha`, `corpus_fingerprint`, and per-stage latency timings.

---

## 16. What Nullius Cannot Demonstrate Yet

Nullius cannot currently demonstrate:

1. **State-of-the-Art Hallucination Detection:** The system has not been tuned or benchmarked against commercial or academic baselines.
2. **Real-World Generalization:** It cannot prove that its pipeline detects hallucinations effectively in live chatbot outputs across unstructured domains.
3. **Retrieval Superiority:** It cannot demonstrate whether hybrid RRF retrieval outperforms BM25 or dense retrieval on open-domain corpora, because the current corpus is an artificial debug slice.
4. **Optimal Aggregation Strategy:** It cannot claim that one aggregation rule is mathematically superior to another on real tasks until evaluated against an external benchmark.

---

## 17. Recommended Next Step

### Primary Recommendation: **Proceed to Step E — Streamlit Inspection Harness**

### Why Step E Is the Next Engineering Step
1. **Prerequisites Are Fully Met:** Step D (the unified FastAPI backend and shared service layer) is complete, tested (280 passing tests), and running. The backend provides all required endpoints (`/health`, `/analyze`, `/analyze/oracle`, `/verify/full` with SSE streaming, `/annotate`).
2. **Fulfills the Project's Core Identity:** Nullius was conceived as an **inspection harness** (*"take nobody's word for it, including this pipeline's"*). An inspection harness cannot fulfill its primary research purpose through terminal JSON outputs alone. Researchers need a visual dashboard to inspect claim spans, examine evidence candidate lists, compare aggregation traces, and toggle the oracle condition.
3. **Enables Visual Error Discovery:** Step E unlocks human-in-the-loop failure case discovery. Visually inspecting failures on the debug corpus will reveal concrete pipeline bugs before attempting expensive full-corpus indexing or browser extension development.

### What Should NOT Be Done Yet
* **Do NOT implement Step F (Chrome Extension) yet:** Building the browser extension before having a functional local inspection UI introduces complex DOM/browser debugging without a working visual baseline.
* **Do NOT attempt model fine-tuning or threshold optimization:** Tuning placeholder thresholds on artificial debug corpora will produce overfitted numbers that fail to generalize.
* **Do NOT attempt full 5.4M-page FAISS indexing yet:** Indexing 25 million sentences requires substantial disk, memory, and time; visual inspection of the debug corpus must come first.

---

## 18. Current One-Sentence Description

> **Nullius is an open-source, contract-tested research inspection harness for claim-level hallucination detection that isolates retrieval failures from factual insufficiency by strictly decoupling pairwise NLI verification from multi-evidence aggregation.**

---

## 19. Current Short Description

> **Nullius is a claim-level hallucination-detection research instrument and inspection harness built in Python, PyTorch, and FastAPI. Rather than acting as a black-box detector, Nullius provides granular, white-box visibility into every stage of an evidence-based verification pipeline: atomic claim decomposition, hybrid BM25/dense retrieval, pairwise cross-encoder NLI verification, and decoupled aggregation. By strictly enforcing that verifiers score individual (claim, evidence) pairs while aggregators remain separate, Nullius allows researchers to compare multiple aggregation strategies (max-entailment, noisy-OR, rank-weighted decay, and abstention) on identical model outputs at zero additional compute cost. It features an oracle substitution path to directly measure retrieval-attributable error, rigorous cryptographic trace provenance, and a unified service layer powering both a CLI and a FastAPI backend with Server-Sent Events streaming.**
