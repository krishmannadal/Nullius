# `src/service.py`, `src/api/` — The Nullius FastAPI Backend

One unified FastAPI service for both the inspection harness (Step E) and the browser extension (Step F).

---

## 1. Problem

Three requirements govern this module:

**1. One backend, not two.** Per ADR-012, the extension endpoint set is a superset of the harness backend. Building separate services or duplicating pipeline orchestration would create divergence where the CLI, UI, and extension see different behaviors. Both the CLI (`src/cli.py`) and the API (`src/api/`) route through `src.service.NulliusService`.

**2. Preserve the central research invariant.**
```
Claim -> Retriever -> Evidence[] -> Pairwise Verifier -> EvidenceVerdict[] -> Aggregator -> ClaimVerdict
```
The verifier scores strictly single `(Claim, Evidence)` pairs (`Verifier.score(claim, ev) -> EvidenceVerdict`). Pooling over *k* remains strictly isolated in the swappable `Aggregator`. The API layer introduces no shortcuts around this contract.

**3. Two latency tiers.** Full verification (extract -> retrieve -> rerank -> NLI cross-encoder -> aggregate) takes hundreds of milliseconds to seconds. The API provides:
- **Tier 1 (`/verify/quick`)**: Decomposes text and evaluates an *evidence-availability signal* using lightweight retrieval. Never asserts that a claim is false.
- **Tier 2 (`/verify/full`)**: Full verification with support for Server-Sent Events (SSE) streaming per-claim verdicts.

---

## 2. Formal I/O

| Endpoint | Method | Request | Response | Semantics |
|---|---|---|---|---|
| `/health` | `GET` | *(none)* | `HealthResponse` | Diagnostics: status, schema version, config hash, device, CUDA/VRAM, queue depth, components. |
| `/analyze` | `POST` | `AnalyzeRequest` | `AnalyzeResponse` | Standard retrieved pipeline run (`mode="retrieved"`). Full trace. |
| `/analyze/oracle` | `POST` | `AnalyzeOracleRequest` | `AnalyzeResponse` | Oracle evaluation (`mode="oracle"`). Substitutes annotated gold evidence for retrieved evidence. |
| `/verify/quick` | `POST` | `VerifyQuickRequest` | `VerifyQuickResponse` | Tier 1 fast availability check. Statuses: `likely-checkable`, `no-evidence-found`. Badge counts. |
| `/verify/full` | `POST` | `VerifyFullRequest` | `AnalyzeResponse` \| SSE Stream | Tier 2 full verification. Returns JSON or streams `start`, `claim_verdict`, `complete` SSE events. |
| `/annotate` | `POST` | `AnnotateRequest` | `AnnotateResponse` | Writes human annotation record to `data/annotations/annotations.jsonl`. |

---

## 3. Algorithm in Pseudocode

```
analyze_endpoint(req):
    acquire service concurrency lock (protect 6 GB VRAM)
    pipe <- get_or_build_pipeline(req.config_path, req.overrides)
    ctx  <- RunContext.create(cfg, pipe)
    claims <- pipe.extractor.extract(req.text)
    for claim in claims:
        candidates <- pipe.retriever.retrieve(claim, pool)
        reranked   <- pipe.reranker.rerank(claim, candidates, k)
        verdicts   <- [pipe.verifier.score(claim, ev) for ev in reranked]
        decision   <- pipe.aggregator.aggregate(claim, verdicts)
    trace <- Trace(...)
    release lock
    return trace.to_dict()
```

---

## 4. Code Walkthrough — The Lines That Matter

**Single Service Execution Boundary (`src/service.py:122`):**
```python
async with self._lock:
    cfg, pipe = self._ensure_loaded(config_path, overrides)
    ctx = RunContext.create(cfg, pipe)
    return analyze(pipe, response_text, ctx, retrieve_k=retrieve_k, check_contracts=check_contracts)
```
CLI and API call the exact same `service.analyze()` method. Models remain loaded in memory across requests rather than re-instantiating on every HTTP call.

**Strict Oracle Isolation (`src/service.py:175`):**
```python
if not target_example.has_gold_evidence:
    raise ValueError(f"Example {target_example.id!r} has no gold evidence annotated")
return analyze(pipe, target_example.text, ctx, example=target_example, corpus=corpus, mode="oracle")
```
Oracle execution is explicitly gated on having valid gold evidence, sets `mode="oracle"`, and cannot contaminate the normal retrieved pipeline state.

**Tier 1 Evidence Availability Signal (`src/service.py:228`):**
```python
valid_hits = [e for e in candidates if e.score >= evidence_floor_score]
status = "likely-checkable" if valid_hits else "no-evidence-found"
```
Tier 1 runs only extractor and retriever. It never invokes the NLI verifier, never emits a truth label, and never asserts that a claim is false.

**Server-Sent Events Streaming (`src/api/routes.py:173`):**
```python
async for item in service.verify_full_stream(...):
    yield _format_sse_event(item["event"], item["data"])
```
Emits `start`, progressive `claim_verdict` events as each claim finishes scoring, and a final `complete` event containing the full trace.

---

## 5. Data Flow

```
HTTP Client (Extension / Streamlit / curl)
       │
       ▼
 FastAPI Router (src/api/routes.py)
       │  Pydantic validation (schemas.py)
       ▼
 NulliusService (src/service.py)
       │  Concurrency lock (asyncio.Lock)
       ▼
 Pipeline Runner (src/pipeline.py)
       │
       ├──> ClaimExtractor       ──> list[Claim]
       ├──> Retriever            ──> list[Evidence]  (or gold in oracle mode)
       ├──> Reranker             ──> list[Evidence]
       ├──> Pairwise Verifier    ──> EvidenceVerdict (Claim x Evidence)
       └──> Aggregator           ──> ClaimVerdict
       │
       ▼
 Trace / QuickResponse / SSE Stream
```

---

## 6. Runnable Verification

### Starting the server:
```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

### Health check:
```powershell
curl -s http://127.0.0.1:8000/health
```
Expected output:
```json
{
  "status": "ok",
  "schema_version": "1.2.0",
  "harness_kind": "debug",
  "device": "cuda:0",
  "queue_depth": 0
}
```

### Failure signature (swapped oracle or contract violation):
If an invalid example or empty input is supplied:
```powershell
curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d '{"text": ""}'
```
Returns HTTP 422:
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "text"]
    }
  ]
}
```

---

## 7. Limitations

- **Single-worker inference queue:** Concurrent requests acquire an `asyncio.Lock` to avoid out-of-memory errors on 6 GB GPU hardware. Queue depth is reported in `/health`.
- **In-memory cache:** The default pipeline is cached in memory; overriding pipeline components in a request builds a temporary pipeline for that request.
- **Localhost binding:** The server is intended for local execution (`127.0.0.1`), not public multi-tenant deployment.

---

## 8. Rejected Alternatives

- **Separate backends for Extension and Harness:** Rejected per ADR-012. Both require the same pipeline and verification types; maintaining two backends guarantees drift.
- **Pydantic dataclasses inside core:** Rejected per ADR-006. Stdlib frozen dataclasses govern `src/core/types.py` for headless stdlib-only trace inspection; Pydantic is restricted to the FastAPI boundary (`src/api/schemas.py`).
- **Batched verification in Verifier:** Rejected per ADR-002. Pairwise scoring `Verifier.score(claim, ev)` is load-bearing for aggregator isolation and comparability.
