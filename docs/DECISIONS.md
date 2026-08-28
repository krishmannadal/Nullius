# Decision log (ADR)

Append-only. When a later step contradicts an earlier decision, append the reversal
rather than editing history — the reversals are part of the record.

Format: `ADR-nnn — date — decision`, then alternatives, rationale, and what would
make us revisit.

---

## ADR-001 — 2026-08-28 — This build is an inspection harness, not an evaluation

**Decision.** Every component shipped is a deliberately simple placeholder behind a
clean interface. No tuning, no threshold selection, no benchmark numbers, no claimed
contribution. The debug corpus is labelled `harness.kind: debug` in the config, in
the README, and (from step 6) in the UI.

**Alternatives.** Build the "real" pipeline first and instrument later — rejected,
because instrumentation retrofitted onto tuned code tends to be shaped around what the
code already does, which is how you end up unable to see the failure you are looking
for.

**Rationale.** The stated goal is visibility into where the pipeline fails. Defaults
that look good hide that; defaults that are obviously placeholders do not.

**Revisit when.** Never during this build. Choosing components is a separate exercise
that starts from `docs/EXPERIMENT_BACKLOG.md`.

**Supersedes.** `docs/BRIEF.md` stage plan (stages 0–9) is on hold for this build; the
seven-step order in the current instruction governs. The brief's documentation and
engineering standards still apply verbatim.

---

## ADR-002 — 2026-08-28 — Verifier scores one pair; Aggregator is a separate object

**Decision.** `Verifier.score(claim, ev) -> EvidenceVerdict`, singular. Pooling over
*k* happens only in an `Aggregator`, which is a separate registry entry.

**Alternatives.** (a) `score(claim, list[Evidence])` with internal pooling — one
batched forward pass, faster. (b) A combined `verify_and_aggregate` stage.

**Rationale.** Aggregation is the slot most likely to become the research question.
If pooling can hide inside a model wrapper it becomes untraceable, unswappable, and
confounded with the choice of NLI checkpoint. With per-pair scores materialised, four
aggregators can be compared on *identical* inputs without re-running the model.

**Cost, accepted.** *k* forward passes per claim instead of one batched call
(≈5×20–40 ms at k=5 on the 4050). Recoverable later by micro-batching *inside* a
verifier that still returns per-pair verdicts.

**Revisit when.** Verification latency dominates a full-corpus run. Even then, the
return type stays per-pair.

---

## ADR-003 — 2026-08-28 — `Evidence.id` is derived from `(doc_id, sent_id)` only

**Decision.** `id ≡ f"ev::{doc_id}::{sent_id}"`, validated (not assigned) in
`__post_init__`. Retriever, score, and rank do not participate.

**Alternatives.** UUID per hit; hash including retriever and rank; parallel arrays of
(id, score).

**Rationale.** Three operations depend on "same corpus sentence" being decidable:
RRF de-duplication across arms, "was gold evidence retrieved, and at what rank?", and
oracle substitution. All three are wrong if the same sentence has two identities.

**Revisit when.** The corpus stops being sentence-addressable (e.g. passage-level
chunking with overlap). Then the identity key changes, and it changes in one place.

---

## ADR-004 — 2026-08-28 — `aggregation_trace` must contain `rule`, `explanation`, `decisive_evidence_ids`

**Decision.** `ClaimVerdict.__post_init__` raises if any of the three keys is absent.

**Alternatives.** Convention plus a UI fallback ("no explanation available").

**Rationale.** The Aggregation panel exists to render *why*, not *what*. A new
aggregator that forgets to explain itself should fail at construction, not render an
empty panel that reads like a model with nothing to say. Making the type system
enforce a UI requirement is unusual; it is justified here because the explanation is
the deliverable.

**Revisit when.** An aggregator appears for which "which evidence drove this" is
genuinely undefined. (A learned set-transformer would still owe attention weights.)

---

## ADR-005 — 2026-08-28 — Decorator registry, namespaced per interface

**Decision.** `@register(kind, name)` with one namespace per ABC; explicit
`_BUILTIN_MODULES` import list; unknown constructor params rejected.

**Alternatives.** Hydra `_target_` import paths; `entry_points` plugins;
`pkgutil` auto-discovery; a dict literal of imports.

**Rationale.** `available(kind)` must return the exact list the five sidebar dropdowns
render — that is the whole reason the registry exists. `_target_` paths are
unenumerable and execute arbitrary imports named in a config file. Auto-discovery is
import-order-dependent and makes "what is registered?" unanswerable without running
it. Full reasoning in `docs/core-registry.md` §8.

**Revisit when.** Components need to live in a separate distributable package.

---

## ADR-006 — 2026-08-28 — Frozen dataclasses, not pydantic

**Decision.** Plain frozen dataclasses with hand-written `to_dict`/`from_dict`.

**Alternatives.** Pydantic v2 models; dicts + JSON Schema; attrs.

**Rationale.** The trace-reading path must work with stdlib only, so the Streamlit
renderer can open a saved trace on a machine with no model stack. Pydantic's coercion
is also actively unwanted: `p_entail="0.7"` silently becoming a float is the class of
silent fix this project is trying to eliminate.

**Revisit when.** The FastAPI layer needs request/response models — those can be
pydantic *at the API boundary only*, converting to these dataclasses immediately.

---

## ADR-007 — 2026-08-28 — NLI class names are not dataset labels

**Decision.** `Label.parse` accepts dataset vocabularies (`SUPPORTS`, `REFUTES`,
`NOT ENOUGH INFO`, `NEI`) and **rejects** `entailment` / `contradiction` / `neutral`.

**Alternatives.** Accept both, mapping neutral → Insufficient.

**Rationale.** "The NLI model said neutral" and "the corpus does not settle this" are
different propositions, and conflating them at parse time makes the conflation
invisible in every downstream table. The mapping is a modelling decision, so it belongs
in an aggregator where it is recorded in `aggregation_trace` and visible in the UI.

**Revisit when.** Never silently. If an aggregator maps neutral → Insufficient, its
`rule` string must say so.

---

## ADR-008 — 2026-08-28 — Reranker defaults to `noop` rather than `None`

**Decision.** A config with no `reranker` key gets `noop`; there is no
`if reranker is not None` branch in the pipeline.

**Alternatives.** `Optional[Reranker]` and a conditional call.

**Rationale.** Same code path on and off means the trace, the timing breakdown, and
the panel layout have identical shape either way, so "reranker off" is a measurement
rather than a different program.

**Revisit when.** The no-op's overhead becomes measurable (it will not).

---

## ADR-009 — 2026-08-28 — `git_sha` is `None` outside a repo, never a placeholder

**Decision.** `git_sha()` returns `Optional[str]`; traces record `null` today because
this directory is not a git repo. `git_is_dirty` is recorded alongside.

**Alternatives.** `"unknown"`, `"nogit"`, or running `git init` unprompted.

**Rationale.** A fabricated sha is indistinguishable from a real one in a trace read
six weeks later. `null` is recoverable; a plausible lie is not.

**Revisit when.** You run `git init` — no code change needed, provenance starts
working immediately.

---

## ADR-010 — 2026-08-28 — `Trace` validates its own cross-references

**Decision.** `Trace.__post_init__` raises if a verdict scores evidence that was
never in `evidence_by_claim[claim_id]`, or if evidence/verdicts reference an unknown
claim.

**Alternatives.** Validate in the pipeline only; validate in tests only.

**Rationale.** Off-by-one and stale-cache bugs of this shape do not crash and do not
change aggregate metrics much — they corrupt the error analysis, which is the
deliverable. Validating at the record boundary catches them regardless of which code
path produced the record, including the frontend's oracle substitution.

**Revisit when.** Validation shows up in a profile (it will not: it is set arithmetic
over ~40 items).

---

## ADR-011 — 2026-08-28 — Additions beyond the specified field list

**Decision.** `Trace` also carries `schema_version`, `mode` (`retrieved`/`oracle`),
and `resolved_config`. `ClaimVerdict.per_evidence` is a `tuple`, not a `list`.

**Rationale.** `schema_version` — saved traces are an error-analysis corpus that will
outlive refactors, and the renderer must be able to refuse rather than mis-render.
`mode` — the oracle comparison is the core diagnostic, and the two runs must be
distinguishable from the file alone. `resolved_config` — failure-case files get copied
out of `results/` and must stay self-describing. `tuple` — a frozen dataclass holding
a `list` is frozen in name only; the JSON wire format is still a list.

**Revisit when.** Any of these turns out to be unused after step 6.

---

## ADR-012 — 2026-08-28 — The two build plans are merged into one backend

**Decision.** `docs/CLAUDE_CODE_EXTENSION_PROMPT.md` assumes a pipeline in `src/`
that does not exist (step 1 built contracts only). Rather than write a stub backend
now and rewrite it later, the original step 5 (`/analyze`, `/analyze/oracle`) and the
extension's §4 (`/verify/quick`, `/verify/full`, `/annotate`, `/health`) become one
FastAPI app, built after the components land.

Merged order: **A** corpus + retrievers · **B** verifiers + aggregators · **C** trace
writer + CLI · **D** one backend · **E** Streamlit harness · **F** extension steps
2–6. E and F are independent after D.

**Rationale.** The extension's endpoint set is a superset of the harness's, and the
harness's "save as failure case" corpus and the extension's annotation corpus are the
same instrument in two skins — they should share one JSONL record format so
`annotation_stats.py` reads both.

**Revisit when.** Never; they are one service.

---

## ADR-013 — 2026-08-28 — `Corpus` has a canonical row order and a fingerprint

**Decision.** Row order is *load order* (documents in file order, sentences in
document order), never sorted. `Corpus.fingerprint()` hashes the ordered
`(doc_id, sent_id)` keys, and every derived artifact records it.

**Alternatives.** Sort by `doc_id` for a "canonical" order; hash the file bytes; no
fingerprint at all and rely on rebuilding by hand.

**Rationale.** Sorting makes row order depend on collation, which differs by locale
and Python version, and hides invalidating edits behind a stable-looking sort. Hashing
file bytes over-triggers: a typo fix would invalidate a valid index. Hashing the
*keys* has exactly the right sensitivity — it changes on insert/delete/reorder (which
break row→key alignment) and not on rewording (which does not).

**Revisit when.** The corpus stops fitting in memory and `Corpus` gets an on-disk
implementation. The fingerprint contract survives that; the storage does not.

---

## ADR-014 — 2026-08-28 — The FAISS index carries a manifest and refuses on mismatch

**Decision.** Every dense index is written with a sidecar manifest recording the
corpus fingerprint, model name, dimension, and sentence count. Loading compares all
four and raises `stale FAISS index` on any mismatch. The fingerprint is also in the
index filename, so two corpora coexist instead of overwriting each other.

**Alternatives.** Rebuild every time (slow at scale); trust the user to delete stale
indexes; store the index inside the corpus file.

**Rationale.** A FAISS index is an anonymous matrix. A corpus/index mismatch produces
confidently wrong evidence *text* at plausible scores — it reads as a bad retriever,
not as a bug, and it corrupts the error analysis rather than the run.

**Revisit when.** Never. If anything, extend it: the manifest should also record the
`sentence-transformers` version, since a tokenizer change alters embeddings without
changing any of the four fields checked today. Logged as OQ-018.

---

## ADR-015 — 2026-08-28 — Fusion is RRF, and per-arm scores survive it

**Decision.** `HybridRetriever` fuses by reciprocal rank, `K = 60` (Cormack, Clarke &
Buettcher 2009), and writes both arms' raw scores and ranks into
`Evidence.retriever_meta`. `Evidence` gained a `retriever_meta` field for this
(SCHEMA_VERSION 1.0.0 → 1.1.0, additive).

**Alternatives.** Min-max or z-score normalisation then weighted sum; CombSUM/CombMNZ;
a learned fusion.

**Rationale.** BM25 scores are unbounded and query-dependent; cosine is bounded. The
only normalisation available at query time is over the candidate list itself, which
makes the fused score depend on the candidates' *spread*: a query where everything
retrieved is irrelevant has its best candidate rescaled to 1.0 exactly as a query with
a perfect match does. RRF uses ranks only and is invariant to that, and has one fewer
weight to not-tune. Cost, accepted: RRF discards magnitude, so it cannot express
"both arms scored everything terribly" — which is precisely the signal a
retrieval-failure detector would want. `retriever_meta` keeps the raw scores so that
signal is recoverable rather than destroyed.

**Revisit when.** The retrieval-failure-vs-insufficiency experiment needs a magnitude
signal. At that point the question is not "which fusion" but "what does the aggregator
get to see", which is the interesting question anyway.

---

## ADR-016 — 2026-08-28 — Gold sentence indices are derived, never typed

**Decision.** `scripts/build_mini_corpus.py` declares gold evidence as
`(doc_id, unique substring)` and resolves it to a `sent_id`, failing the build if the
substring is absent **or ambiguous**. `validate_against` then proves every resolved
key exists in the corpus before anything is written.

**Alternatives.** Hand-written `sent_id`s; first-match resolution.

**Rationale.** A hand-counted index is an off-by-one that produces a plausible
sentence from the right document — invisible on inspection, and it silently breaks the
oracle condition, which is the central measurement in the project. Ambiguity is an
error rather than a first-match because a needle matching two sentences means the
author did not know which they meant.

**Revisit when.** Never for hand-written corpora. FEVER-derived examples come with
their own indices, which must instead be *validated* against the corpus at build time
— same guarantee, different mechanism.

---

## ADR-017 — 2026-08-28 — `Recall@k` is `None`, not 0.0 or 1.0, when gold is empty

**Decision.** `recall_at_k` and `reciprocal_rank` return `Optional[float]` and give
`None` for examples with no gold evidence.

**Alternatives.** 1.0 (vacuous truth); 0.0; excluding such examples silently.

**Rationale.** FEVER's NEI class has empty evidence by construction, so roughly a
third of a FEVER slice has no denominator. 1.0 flatters the retriever; 0.0 punishes it
for a question nobody asked. Both corrupt any mean taken over them, and the corruption
is invisible in the aggregate. `None` forces whatever aggregates to decide in the open.

**Revisit when.** Never. If a future aggregation wants a convention, it states it.
