# `src/core/interfaces.py` — the five ABCs and the contract checks

## 1. Problem

Swappability is easy to claim and hard to keep. Two things erode it:

* **Signature creep.** A verifier that "just needs a bit of context" grows a
  `list[Evidence]` parameter, and six weeks later pooling lives in three places and
  you can no longer compare aggregators without re-running the NLI model.
* **Silent contract violations.** A reranker returns evidence objects it built
  itself rather than the ones it was given; a retriever returns `k+2` items; ranks
  come back 0-based from one component and 1-based from another. None of these
  crash. All of them corrupt the Retrieval panel and Recall@k.

This module fixes the signatures and makes the invariants checkable in O(k).

## 2. Formal I/O

| ABC | Method | Signature | Post-conditions |
|---|---|---|---|
| `ClaimExtractor` | `extract` | `str → list[Claim]` | ids unique; `span.text_from(response)` is a substring of `response` |
| `Retriever` | `retrieve` | `(Claim, int k) → list[Evidence]` | `len ≤ k`; ranks `1..len` contiguous ascending; ids unique; `score` in retriever-native units |
| `Reranker` | `rerank` | `(Claim, list[Evidence], int k) → list[Evidence]` | result ids ⊆ input ids; ranks renumbered `1..len`; `retriever_name` updated |
| `Verifier` | `score` | `(Claim, Evidence) → EvidenceVerdict` | `verdict.claim_id == claim.id ∧ verdict.evidence_id == ev.id`; `latency_ms` measured |
| `Aggregator` | `aggregate` | `(Claim, list[EvidenceVerdict]) → ClaimVerdict` | pure; handles `[]`; writes `rule`, `explanation`, `decisive_evidence_ids` |

`Component` supplies `name` (the registry string) and `describe() → {name, class,
params}`, which is what lands in `Trace.resolved_config`.

## 3. Algorithm

There is no algorithm — this is a specification file. The checks are:

```
check_claims(claims, response):
    ids unique
    ∀c with span: c.span.text_from(response) ⊆ response

check_evidence_list(ev, k, stage):
    len(ev) ≤ k
    ids unique
    [e.rank for e in ev] == [1..len(ev)]

check_rerank_is_subset(before, after):
    {e.id for e in after} ⊆ {e.id for e in before}

check_pair_verdict(v, claim, ev):
    (v.claim_id, v.evidence_id) == (claim.id, ev.id)
```

Cost: O(k) set construction over lists of ≤ ~100 elements, i.e. microseconds against
a ~30 ms NLI forward pass. Cheap enough to run on every call in the pipeline, not
only in tests — which is the point, since these bugs appear under configurations the
tests do not enumerate.

## 4. Code walkthrough — the lines that matter

**The single-pair verifier signature (`interfaces.py:135`).**
```python
def score(self, claim: Claim, ev: Evidence) -> EvidenceVerdict:
```
This is the load-bearing line in the whole repository. Because `ev` is one object,
a verifier *cannot* pool, so aggregation can only happen in an `Aggregator`, which
is a separate registry entry with a separate name in the trace and a mandatory
`aggregation_trace`. The cost is real and worth stating: `k` forward passes per
claim rather than one batched call over `k`. On a 4050 with DeBERTa-v3-base at
fp16 and `k=5`, that is roughly 5×(20–40 ms) ≈ 0.1–0.2 s per claim — fine for an
inspection harness, and recoverable later with an internal micro-batch that still
returns per-pair verdicts.

**The rank contract (`interfaces.py:183`).**
```python
expected_ranks = list(range(1, len(evidence) + 1))
if [e.rank for e in evidence] != expected_ranks:
    raise ContractError(...)
```
Rank must equal list position, always. Two things depend on it: the Verification
panel sorts by rank rather than by score (so retrieval order stays visible even when
the NLI model disagrees with it), and RRF's `1/(rrf_k + rank)` is nonsense if one arm
is 0-based.

**The subset check (`interfaces.py:192`).**
```python
unknown = {e.id for e in after} - {e.id for e in before}
```
A reranker may reorder and drop; it may not create. Since `Evidence.id` is corpus
identity, this also catches the subtler bug of a reranker that rebuilds evidence
objects from a *different* corpus copy — the ids would not match.

**Lenient `Component.__init__` (`interfaces.py:51`).** `describe()` reads
`resolved_params` via `getattr(..., {})`, so a subclass that forgets
`super().__init__()` degrades to an empty params dict instead of raising
`AttributeError` mid-run. Deliberate leniency at a boundary where the failure would
otherwise appear far from its cause.

## 5. Data flow

```
registry.build_pipeline(cfg) ──► Pipeline{extractor, retriever, reranker, verifier, aggregator, k}
                                   │
run_pipeline (step 4):             ▼
  claims  = extractor.extract(response)          → check_claims
  for claim in claims:
      cands = retriever.retrieve(claim, k_pool)  → check_evidence_list
      ev    = reranker.rerank(claim, cands, k)   → check_rerank_is_subset + check_evidence_list
      pairs = [verifier.score(claim, e) for e in ev]   → check_pair_verdict per call
      verdict = aggregator.aggregate(claim, pairs)
```

## 6. How to verify it

The ABCs are exercised through `tests/test_registry.py`, which defines a minimal
legal implementation of each and builds a pipeline from them:

```powershell
python -m pytest tests/test_registry.py -v
```
Expected: **14 passed**.

Failure signature to watch for once real components exist: `ContractError:
retrieve returned ranks [0, 1, 2, 3, 4]` means a retriever enumerated from zero —
fix the component, never the check, because RRF and the UI both assume 1-based.

A direct check that the ABCs are actually abstract:

```powershell
python -c "from src.core.interfaces import Verifier; Verifier()"
```
Expected: `TypeError: Can't instantiate abstract class Verifier with abstract method score`.

## 7. Limitations and failure modes

* **No batching anywhere.** `retrieve` takes one claim and `score` takes one pair.
  For an inspection harness that is right (it keeps per-pair latency honest), but a
  full-corpus evaluation over thousands of claims will want batched variants. The
  intended shape is an optional `score_batch` default-implemented as a loop, added
  when profiling says so — not now.
* **The contract checks are advisory, not enforced by the type system.** A component
  that is never run through the pipeline (e.g. called directly in a notebook) can
  violate them freely.
* **`check_claims` only verifies that the span text appears *somewhere* in the
  response**, not that it appears *at that offset*. A stricter check
  (`response[start:end] == claim.text`) would be wrong for extractors that legitimately
  rewrite a claim (LLM decomposition produces text that is not a substring). The
  looser check is what both cases can satisfy; the strict version belongs in the
  sentence-splitter's own test.
* **`Component.name` comes from the class, not the instance.** Two differently
  configured instances of the same class report the same name; they are told apart by
  `describe()["params"]`. If you need distinct names in one trace, register the
  variant separately.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **`typing.Protocol` instead of ABCs** | Structural typing means a class with a typo'd method name is simply "not a Retriever" and fails at call time. The registry needs a runtime `issubclass` check to reject registering a Retriever as a Verifier, and Protocols make that unreliable. |
| **`Verifier.score(claim, list[Evidence])`** | The design flaw this repo exists to avoid. Would let pooling hide inside a model wrapper, making aggregators incomparable without re-running the model. |
| **One `Stage.run(input) → output` interface for all five** | Uniform and useless: every stage needs a different signature, so the payload would become `dict`, and every type error would move to runtime. |
| **Returning `(evidence, scores)` tuples instead of `Evidence` objects** | Requires parallel-array alignment, which is exactly the silent-bug class being designed out. |
| **Contract checks as `assert`** | Disabled under `python -O`. `ContractError(AssertionError)` reads like an assert, keeps `pytest.raises(AssertionError)` working, and cannot be optimised away. |
| **Validating contracts inside the ABCs (template method)** | Would force every implementation through a fixed wrapper and make ad-hoc use in a notebook awkward. Keeping checks as free functions lets the pipeline enforce and the notebook not. |
