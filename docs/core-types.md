# `src/core/types.py` — the data contracts

## 1. Problem

Five components written at five different times must exchange objects without a
shared mental model, and the resulting records must still be readable weeks later
by a frontend that has no models installed. Without a fixed contract, three things
go wrong silently:

1. **Evidence identity drifts.** BM25 returns "sentence 3 of *Marie_Curie*" and the
   dense retriever returns the same sentence, but they are different Python objects
   with different scores. Any fusion that de-duplicates on the wrong key either
   double-counts or drops evidence, and nothing crashes.
2. **Aggregation hides inside a verifier.** If a verifier is allowed to take a list,
   pooling becomes an implementation detail of the NLI wrapper — untraceable and
   unswappable.
3. **Verdicts reference evidence the claim never saw.** An off-by-one in a rerank or
   a stale cache produces a verdict "explained" by a sentence that was never in the
   candidate list. Aggregate metrics look fine; the error analysis is fiction.

This module makes (1) impossible by construction, (2) impossible via the interface
signatures in `interfaces.py`, and (3) an exception at `Trace` construction.

## 2. Formal I/O

Nothing here computes; it constrains. Types and admissible ranges:

| Type | Field | dtype / domain | Invariant |
|---|---|---|---|
| `SourceSpan` | `start, end` | `int ≥ 0` | `0 ≤ start ≤ end`; indexes the **original** response string |
| `Claim` | `id` | `str` | `blake2b(response_id ⊕ span ⊕ text)[:12]`, prefix `clm_` |
| | `text` | `str` | non-empty after strip |
| | `extractor_meta` | `dict` | JSON-serialisable |
| `Evidence` | `id` | `str` | `≡ f"ev::{doc_id}::{sent_id}"` — **checked, not computed, in `__post_init__`** |
| | `sent_id` | `int ≥ 0` | |
| | `rank` | `int ≥ 1` | 1-based |
| | `score` | `float`, finite | retriever-native scale, **not** normalised |
| | `is_gold` | `Optional[bool]` | `None` = unannotated ≠ `False` = annotated-not-gold |
| `EvidenceVerdict` | `p_entail, p_contra, p_neutral` | `Optional[float] ∈ [0,1]` | all-None **or** all-present with `\|Σp − 1\| ≤ 10⁻³` |
| | `similarity` | `Optional[float] ∈ [−1,1]` | raw cosine, threshold-free |
| | `latency_ms` | `float ≥ 0` | measured, not constant |
| `ClaimVerdict` | `confidence` | `float ∈ [0,1]` | |
| | `per_evidence` | `tuple[EvidenceVerdict,…]` | every element has `claim_id == self.claim_id` |
| | `aggregation_trace` | `dict` | must contain `rule`, `explanation`, `decisive_evidence_ids` |
| `Trace` | `evidence_by_claim` | `dict[str, tuple[Evidence,…]]` | keys ⊆ claim ids |
| | `verdicts` | `tuple[ClaimVerdict,…]` | `{ev.evidence_id} ⊆ {e.id for e in evidence_by_claim[claim_id]}` |
| | `mode` | `"retrieved" \| "oracle"` | |

The only equation in the file is the probability-simplex check:

$$\left|\,p_{\text{entail}} + p_{\text{contra}} + p_{\text{neutral}} - 1\,\right| \le \tau,\qquad \tau = 10^{-3}$$

τ = 10⁻³ is chosen to survive fp16 softmax rounding (worst case ≈10⁻³ for three
terms near 0.5) while still catching the two realistic bugs: logits stored without a
softmax (sum is arbitrary) and a two-class model's output padded with a zero (sum
still 1 — *not* caught; see §7).

## 3. Algorithm

```
stable_id(prefix, *parts):
    h ← blake2b(digest_size=16)
    for p in parts:
        h.update(utf8(str(p)))
        h.update(0x1f)              # unit separator: ("ab","c") ≠ ("a","bc")
    return prefix + "_" + hex(h)[:12]

evidence_id(doc_id, sent_id) := "ev::" + doc_id + "::" + str(sent_id)

Trace.__post_init__:
    assert claim ids unique
    known ← {c.id}
    for cid in evidence_by_claim:  assert cid ∈ known
    for v in verdicts:
        assert v.claim_id ∈ known
        available ← {e.id for e in evidence_by_claim[v.claim_id]}
        scored    ← {ev.evidence_id for ev in v.per_evidence}
        assert scored ⊆ available          # ← the index-alignment guard
```

## 4. Code walkthrough — the lines that matter

**`stable_id`, the separator (`types.py:108`).**
```python
h.update(str(p).encode("utf-8"))
h.update(_ID_FIELD_SEP)          # b"\x1f"
```
Without the separator, `("ab", "c")` and `("a", "bc")` hash the same bytes. That is
the classic concatenation-id collision, and here it would silently merge two claims
in a cache. `\x1f` is ASCII *unit separator*, which cannot appear in text extracted
from a corpus.

**`Evidence.__post_init__`, identity is checked not computed (`types.py:276`).**
```python
expected = evidence_id(self.doc_id, self.sent_id)
if self.id != expected:
    raise ValueError(...)
```
This *checks* rather than *assigns* because the dataclass is frozen — assignment
would need `object.__setattr__`, which hides the constraint. The consequence you
care about: two hits on the same corpus sentence from BM25 and from the dense arm
have equal `id`, so RRF can fuse on `id` and oracle substitution can ask "was gold
retrieved?" by set membership. Rank and score deliberately do **not** participate.

**`EvidenceVerdict.__post_init__`, the all-or-nothing probability rule
(`types.py:366`).**
```python
n_present = sum(p is not None for p in probs)
if n_present not in (0, 3):
    raise ValueError("must be all-None or all-present")
```
`SimilarityVerifier` produces no NLI distribution. The alternative — writing `0.0`
into the three probability fields — would make a similarity verdict indistinguishable
from a confidently-neutral NLI verdict in every downstream aggregate. `None` is the
honest encoding, and forcing all-or-nothing stops a half-populated verdict from a
partially-migrated verifier.

**`ClaimVerdict.__post_init__`, the mandatory trace keys (`types.py:452`).**
```python
missing = [k for k in REQUIRED_AGGREGATION_TRACE_KEYS if k not in self.aggregation_trace]
if missing: raise ValueError(...)
```
This is the type system enforcing a UI requirement, which is unusual and deliberate:
the Aggregation panel renders *why*, and a new aggregator that forgets to explain
itself must fail at construction rather than render an empty panel that looks like a
model that had nothing to say.

**`Trace.__post_init__`, the dangling-evidence check (`types.py:528`).**
```python
dangling = scored - available
if dangling:
    raise ValueError(f"... verdict scores evidence never given to it: {sorted(dangling)}")
```
The whole reason this class validates itself. Catches: reranker returning
substituted objects, cached verdicts surviving a claim edit, oracle evidence
injected into `per_evidence` without being put in `evidence_by_claim`.

**`Trace.to_json_line` (`types.py:593`).** `ensure_ascii=False` so that non-ASCII
corpus text stays readable in the JSONL by eye; `sort_keys=True` so a trace file
diffs cleanly between runs.

## 5. Data flow

```
ClaimExtractor  ── Claim ────────────────►  Retriever
Retriever       ── list[Evidence] ───────►  Reranker ──► Verifier
Verifier        ── EvidenceVerdict ──────►  Aggregator
Aggregator      ── ClaimVerdict ─────────►  Trace writer ──► results/<run_id>/traces.jsonl
                                                          └─► results/failure_cases/*.json
Trace.from_json_line  ◄── the Streamlit renderer, offline, no models
```

Boundary shapes: for one response with `n` claims at `k` evidence each, a `Trace`
holds `n` claims, `n×k` `Evidence`, and `n×k` `EvidenceVerdict` inside `n`
`ClaimVerdict`s. At `n=8, k=5` that is ~40 pair-verdicts, i.e. a trace is a few tens
of KB — cheap to keep one JSONL per run.

## 6. How to verify it

```powershell
python -m pytest tests/test_types.py -v
```
Expected: **32 passed**. Key cases and their failure signatures:

| Test | Passing means | Failure signature |
|---|---|---|
| `test_evidence_id_depends_only_on_corpus_position` | RRF/oracle de-dup is sound | two ids for one sentence → fusion double-counts, "gold retrieved?" says no when it was |
| `test_label_parse_refuses_nli_class_names` | NLI-neutral is not silently NEI | `Label.parse("neutral")` returning `Insufficient` — the conflation would be invisible in every table |
| `test_trace_rejects_a_verdict_scoring_unseen_evidence` | the alignment guard is live | trace constructs happily → error analysis cites evidence the model never saw |
| `test_trace_round_trips_through_json` | saved traces are renderable offline | `Label` comes back as `str`, or `per_evidence` as `list` ≠ `tuple`, so `==` fails |

Ten-second manual check that the guard is real:

```powershell
python -c "from tests.test_types import build_trace; t=build_trace(); print(t.to_json_line()[:160])"
```

## 7. Limitations and failure modes

* **The simplex check cannot catch a 2-class model padded to 3.** If a NLI head with
  {entail, non-entail} is wrapped as `(p, 0.0, 1-p)`, the sum is 1 and validation
  passes. Mitigation is at the verifier: its doc must state the checkpoint's label
  order, and `tests/test_verifiers.py` (step 3) will assert index↔name mapping
  against a known-answer pair.
* **`aggregation_trace` is a plain `dict`, so a frozen `ClaimVerdict` is only
  shallowly immutable.** Someone can mutate the dict in place after construction.
  Accepted deliberately: `MappingProxyType` is not JSON-serialisable by `json.dumps`,
  and the serialisation path matters more than defensive immutability here.
* **`per_evidence` is a `tuple`, not the `list` in the original spec.** Frozen
  dataclasses holding mutable containers are frozen in name only. `to_dict()` emits a
  JSON list, so the wire format is unaffected.
* **Content-addressed claim ids collide for two identical sentences with identical
  spans in one response** — impossible in practice, since identical spans means the
  same span. Two identical sentences at *different* offsets get different ids, which
  is correct.
* **Span validity is checked only when `check_claims` is called** (in
  `interfaces.py`), not inside `Claim`, because `Claim` does not hold the response
  text. An extractor that normalises whitespace before splitting and reports spans
  into the normalised copy will produce highlights that drift. That check runs at the
  pipeline boundary.
* **No unit is attached to `score`.** BM25 scores and cosine scores live in the same
  field with different meanings; only `retriever_name` disambiguates. Accepted: the
  UI shows them in separate columns, and normalising at this layer would destroy the
  information the Retrieval panel exists to show.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Pydantic models** | Would bring validation "for free", but pulls pydantic into the trace-reading path (the frontend must load traces with only stdlib), and its coercion is a liability here: `p_entail="0.7"` silently becoming a float is exactly the class of silent fix this project is trying to prevent. |
| **Plain dicts + JSON Schema** | Schema lives away from the code, so nothing fails at the moment of the bug. Frozen dataclasses fail at construction, with a stack trace pointing at the component that did it. |
| **Auto-generated UUID ids** | Non-reproducible: the same input produces different ids on every run, so traces from two runs cannot be diffed and caches cannot be warm. Content-addressing gives free memoisation and a meaningful diff. |
| **`Evidence.id` including retriever + rank** | Makes the same sentence two objects under fusion. This is precisely the bug the identity rule exists to prevent. |
| **A single `probs: dict[str,float]` field** | Loses static typing and invites key-name drift (`"entail"` vs `"entailment"` vs `"ENTAILMENT"`) across verifiers. |
| **`Label` as plain `str`** | No exhaustiveness, and typos become new classes. The `str`-mixin `Enum` gives both JSON-friendliness and a closed set. |
| **Storing the label as an int index** | Index↔name mapping is the single most common silent bug in NLI code. Storing names means a mis-mapping becomes a visibly wrong word, not a plausible wrong number. |
