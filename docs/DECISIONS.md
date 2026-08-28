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

---

## ADR-018 — 2026-08-28 — FEVER's 2017 dump, not live Wikipedia

**Decision.** The debug corpus is built from `wiki-pages.zip` (1.71 GB, FEVER's June
2017 snapshot), downloaded from `fever.ai`. The `s3-eu-west-1.amazonaws.com/fever.public/*`
URLs are dead (403).

**Alternative considered and rejected.** Fetch only the ~3–5k needed pages from the
live Wikipedia API — ~30 MB instead of 1.71 GB, no dump, no streaming.

**Rationale for rejecting it.** Gold evidence is `(page, sentence_id)` into the 2017
snapshot. Live articles have different sentence boundaries, so indices would have to be
re-derived by string matching against changed text. That is lossy, and what it corrupts
is `is_gold` — the single signal the entire retrieved-vs-oracle comparison rests on.
A 1.7 GB download is far cheaper than an oracle that is quietly wrong.

**Revisit when.** Never for FEVER. A future dataset with stable text addressing could
be fetched live.

---

## ADR-019 — 2026-08-28 — FEVER `lines` is parsed by declared index, never by position

**Decision.** `parse_lines_field` builds `sentences[i]` by asking for index *i*,
padding absent indices with `""` and preserving empty sentences. It never uses
`enumerate`, and records with a non-integer leading field are dropped.

**Alternatives.** Positional enumeration after filtering blank records — shorter, and
what almost every FEVER preprocessing snippet does.

**Rationale.** Empty sentences are real and they occupy an index. Filtering them shifts
every later gold key by one; the key still *resolves*, just to the neighbouring
sentence. `Corpus.fingerprint` cannot catch it (shape is self-consistent) and
`validate_against` cannot catch it (the key exists). The only defences are an
index-driven parse and a direct test, so both exist: `tests/test_fever_parse.py` plus a
build-time tripwire that warns when any gold key lands on an empty sentence.

**Revisit when.** Never. If the dump format changes, `parse_stats.non_integer_index`
in the manifest spikes and says so.

---

## ADR-020 — 2026-08-28 — Distractors are other claims' gold pages, then uniform random

**Decision.** The distractor pool is (1) gold pages of dev claims *not* in the sample,
then (2) uniform random pages via single-pass reservoir sampling, to reach `--n-docs`.

**Alternatives.** Uniform random only; hard negatives selected by a retriever; use all
2,892 dev gold pages.

**Rationale.** Uniform random over 5.4 M pages yields mostly short stubs, so the gold
page is often the only on-topic document and retrieval becomes trivial. Other claims'
gold pages are substantive articles and cost nothing extra, and — critically — they are
**not selected by any retriever**, so the corpus does not become a function of the thing
being measured. Hard negatives would be the honest way to make retrieval difficult, but
that circularity is worse than an easy corpus that is labelled easy.

**Consequence, accepted and documented.** The pool contains pages that may genuinely
bear on our claims without being annotated gold for them, so `is_gold == False` means
"not annotated for this claim", never "irrelevant".

**Revisit when.** A retrieval experiment needs a genuinely adversarial pool. That is a
different corpus with a different name, not a tweak to this one.

---

## ADR-021 — 2026-08-28 — Evidence groups are preserved alongside flattened gold

**Decision.** `Example.gold_evidence` is the flattened union of all annotation groups;
`meta.evidence_groups` keeps the original grouped structure.

**Rationale.** Each FEVER group is a *complete alternative* evidence set, and 11.3% of
groups span more than one page (measured on dev). Flattening is right for Recall@k and
wrong for the official FEVER score, which asks whether at least one **complete** group
was recovered. Flattening alone would destroy that information permanently.

**Revisit when.** The FEVER score gets implemented — the data is already there.

---

## ADR-022 — 2026-08-28 — Claims whose gold does not fully resolve are dropped, and named

**Decision.** If any gold key for a claim is missing from the assembled corpus, the
claim is excluded and its id recorded in `manifest.json → dropped_claims`.

**Alternatives.** Keep it with partial evidence; keep it and mark it.

**Rationale.** The oracle condition substitutes gold evidence for retrieved evidence.
A claim with partially-resolvable gold gives a quietly incomplete oracle, which biases
the retrieved-vs-oracle gap — the central measurement — in an invisible direction.
Dropping loses a claim; keeping loses the meaning of the measurement.

---

## ADR-023 — 2026-08-28 — Wikipedia page titles are NFC-normalised at every boundary

**Decision.** `normalize_page_id()` applies `unicodedata.normalize("NFC", ...)` to every
page title entering the system — both from `shared_task_dev.jsonl` and from
`wiki-pages.zip`.

**Why this is not a hypothetical.** The two FEVER files disagree about normalisation
for the same title:

```
shared_task_dev.jsonl:  'Cléopâtre'   NFD, decomposed
wiki-pages.zip:         'Cl\xe9op\xe2tre'          NFC, precomposed
```

They render identically and compare unequal, so exact matching drops the page and the
claim with it. Found by the build's own "gold pages absent from the dump" warning —
which is also the line that crashed the first run on cp1252, because the character it
was printing was that combining acute.

**Measured scale over all 2,892 dev gold pages:** 32 are non-NFC. That is **1.1% of all
gold pages but 72.7% of the 44 non-ASCII gold pages** (Björk, Curaçao, Café Society,
Cléopâtre, Bañuela, 1974 Cypriot coup d'état, …). The aggregate figure is the dangerous
part: a 1% loss reads as noise, while what is actually happening is that three quarters
of an identifiable slice disappears. Any per-slice analysis of non-English entity names
would have been computed on a quarter of its sample with nothing indicating it.

**Alternatives.** NFD everywhere (would rewrite 5.4 M dump titles instead of 32 dev
titles); casefold/ASCII-fold matching (over-merges genuinely distinct titles); accept
the loss (it is biased, so no).

**Revisit when.** Another dataset joins with its own normalisation convention. The fix
generalises: normalise at the boundary, never at comparison sites.

---

## ADR-024 — 2026-08-28 — NLI label indices are read from the checkpoint, never hardcoded

**Decision.** `_resolve_label_indices` reads `model.config.id2label` at load and raises
if the three names are not each present exactly once. Everything downstream indexes
through the resolved dict, never through a literal.

**Why, measured.** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` ships
`{0: entailment, 1: neutral, 2: contradiction}` — the **reverse** of the common MNLI
convention. Hardcoding index 0 as contradiction would swap Supported and Contradicted
for every claim in the project. Nothing structural catches it: the probabilities still
form a simplex, so `EvidenceVerdict` validation passes and the numbers look like
plausible model output.

**Alternatives.** Hardcode the order (the bug). Trust the config without a behavioural
check (a config can be wrong or a checkpoint re-headed) — so a known-answer probe over
three obvious entail/refute/neutral triples runs in the tests as well.

**Revisit when.** Never. A `LABEL_0`-style checkpoint raises rather than getting an
arbitrary permutation.

---

## ADR-025 — 2026-08-28 — Every tokenizer call passes an explicit `max_length`

**Decision.** `truncation="longest_first", max_length=256` on every call. No bare
`truncation=True` anywhere.

**Why, measured.** `tokenizer.model_max_length` for this checkpoint is
`1000000000000000019884624838656` — the sentinel it ships when no limit is configured.
`truncation=True` alone is therefore **a complete no-op**: a 20,000-word evidence comes
back at >10,000 tokens. `test_without_explicit_max_length_truncation_is_a_no_op`
demonstrates it rather than asserting it.

`longest_first` rather than the default because `truncation_side="right"` means naive
truncation removes from the end of the concatenated pair. `longest_first` removes from
whichever member is currently longer, so a short claim survives intact against enormous
evidence — which is the only policy that never destroys the thing being verified.

**Cost, accepted.** `max_length=256` truncates a small fraction of FEVER sentences.
That fraction is a placeholder, unmeasured, and logged as OQ-027.

---

## ADR-026 — 2026-08-28 — `evidence_rank` / `evidence_score` are denormalised into the verdict

**Decision.** `EvidenceVerdict` gained two optional fields, set by `make_verdict` from
the `Evidence` being scored. SCHEMA_VERSION 1.1.0 → 1.2.0, additive.

**Why.** `Aggregator.aggregate(claim, verdicts)` receives no evidence, by design —
purity is what makes aggregators comparable and keeps them from reaching the corpus.
But `WeightedByRetrievalAggregator` needs retrieval position. The three options were:
change the ABC signature (breaks the spec and the isolation), hand the aggregator the
corpus (breaks purity), or put the signal in the verdict where the trace can see it.

This is exactly what `core-interfaces.md` already required: *"if an aggregator needs a
signal, that signal must already be inside an EvidenceVerdict, which forces it to be
visible in the trace."*

**Revisit when.** An aggregator needs something else off the Evidence. The answer will
be the same: denormalise it, visibly.

---

## ADR-027 — 2026-08-28 — Aggregator weights use rank, not retrieval score

**Decision.** `WeightedByRetrievalAggregator` computes `w_i = rank_i^-alpha`.
`evidence_score` is recorded in the trace but unused.

**Rationale.** Step A established that retriever scores are not comparable — BM25 is
unbounded and query-dependent, cosine is bounded in [-1,1], RRF scores live on a third
scale entirely. Weighting by score would make the aggregator's behaviour silently
depend on which retriever produced the evidence, which is precisely the kind of hidden
coupling this architecture exists to prevent. Rank means the same thing everywhere.

**Revisit when.** There is data. The scores are in the trace so the question can be
settled empirically rather than by argument.

---

## ADR-028 — 2026-08-28 — Similarity yields a support signal only, never a contradiction signal

**Decision.** `support_contra_neutral` maps a similarity-only verdict to
`((sim+1)/2, None, None)`. Consequently **no aggregator can output `Contradicted` from
a similarity-only pipeline**, and all five record
`signal: "similarity_only(no_contradiction_signal)"`.

**Rationale.** Cosine similarity is structurally incapable of detecting contradiction:
"Marie Curie was born in Warsaw" and "…in Paris" are near-identical strings with high
similarity. `test_similarity_cannot_tell_support_from_contradiction` demonstrates this
on the real encoder rather than asserting it.

**Alternatives.** Return `0.0` for contradiction — would let an aggregator conclude "no
contradiction detected" from a signal that cannot detect one. Refuse to aggregate
similarity verdicts at all — would make the `similarity` verifier unrunnable end to end,
hiding the limitation instead of displaying it.

**Revisit when.** A verifier emits both NLI and similarity in one verdict, which is what
the "is cosine redundant once a cross-encoder sees the pair?" ablation needs.

---

## ADR-029 — 2026-08-28 — Abstain is a separate outcome from Insufficient

**Decision.** `ThresholdWithAbstainAggregator` is the only aggregator that emits four
labels. `Insufficient` asserts that the evidence was read and does not settle the claim;
`Abstain` asserts nothing and sets `abstained=True`.

**Rationale.** Only the second belongs on a risk–coverage curve. Collapsing them —
which every threshold-free aggregator effectively does — makes selective prediction
unmeasurable, and that is the direction the project's most promising research question
points.

**Consequence, observed immediately.** On FEVER's NEI examples this aggregator returns
`Abstain` where gold says `NOT ENOUGH INFO`, so it scores as wrong. **A benchmark with
no abstain class structurally penalises abstention.** Worth knowing before any
risk–coverage work; logged as OQ-029.
