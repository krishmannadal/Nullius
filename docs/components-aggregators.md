# `src/components/aggregators.py` — five ways to turn k verdicts into one label

## 1. Problem

An aggregator answers: given *k* independent (claim, evidence) scores, what is the
claim's label, how confident are we, and **why**.

This is the slot most likely to become the research question, so the module is built
for comparison rather than for performance. Five implementations run on *identical*
per-pair inputs — that is only possible because `Verifier.score` is single-pair
(ADR-002), so the NLI model runs once and all five aggregators consume the same
verdicts. Swapping aggregator is a config change costing zero forward passes.

The three problems it must not fudge:

1. **Zero evidence is not an edge case.** `aggregate(claim, [])` happens whenever
   retrieval returns nothing, and it is exactly the boundary this project exists to
   study: *the corpus does not settle this* versus *the retriever missed it*. No
   aggregator may guess which.
2. **`Insufficient` and `Abstain` are different propositions.** *Insufficient* asserts
   something about the world (the evidence was read and does not settle the claim).
   *Abstain* asserts nothing (the system declines to answer). Only the second belongs on
   a risk–coverage curve.
3. **Mapping NLI-`neutral` onto `Insufficient` is a modelling decision, not a type
   conversion.** `Label.parse` refuses to do it (ADR-007) precisely so it has to happen
   here, visibly, in `aggregation_trace["rule"]`.

## 2. Formal I/O

`Aggregator.aggregate(claim: Claim, v: list[EvidenceVerdict]) -> ClaimVerdict`.

Pure: no retrieval, no model calls, no corpus access. Every signal it uses must already
be inside an `EvidenceVerdict`, which is why `evidence_rank` / `evidence_score` are
denormalised there.

Write $s_i = p_{\text{entail}}(e_i)$, $c_i = p_{\text{contra}}(e_i)$, $n_i = p_{\text{neutral}}(e_i)$.

| name | rule | can emit |
|---|---|---|
| `max_entailment` | $P_c = \max_i p_c(e_i)$, then $\arg\max_c$ | S · C · I |
| `noisy_or` | $P_c = 1 - \prod_i (1 - p_c(e_i))$ | S · C · I |
| `weighted_by_retrieval` | $w_i = \text{rank}_i^{-\alpha}$, $P_c = \frac{\sum_i w_i p_c(e_i)}{\sum_i w_i}$ | S · C · I |
| `threshold_abstain` | floors on $\max_i s_i$, $\max_i c_i$ | S · C · I · **A** |
| `majority` | one vote per verdict for its own $\arg\max$; plurality | S · C · I |

Every one writes `rule`, `explanation`, `decisive_evidence_ids` (enforced by
`ClaimVerdict.__post_init__`), plus `signal` and `n_evidence`.

**Similarity-only inputs.** Cosine has no contradiction signal — "born in Warsaw" and
"born in Paris" are near-identical strings. `support_contra_neutral` maps
$\text{sim} \mapsto (\text{sim}+1)/2$ as a *support* signal and returns `None` for the
other two, so **a similarity-only pipeline can never output `Contradicted`.** That is a
property of the signal, not a defect here, and every aggregator records it as
`signal: "similarity_only(no_contradiction_signal)"`.

## 3. Algorithm

```
support_contra_neutral(v):
    if v.has_nli:            return (p_entail, p_contra, p_neutral)
    if v.similarity is not None: return ((sim + 1)/2, None, None)   # support only
    raise

_empty_verdict(claim):                       # IDENTICAL across all five
    label ← Insufficient, confidence ← 0.0   # maximally unsure, not certainly-insufficient
    explanation names BOTH possibilities and says it cannot distinguish them

threshold_abstain(claim, v):
    s ← max_i s_i ; c ← max_i c_i
    if s >= support_floor    and s >= c : Supported
    elif c >= contradict_floor and c > s : Contradicted
    elif max(s, c) < abstain_below       : ABSTAIN, abstained=True
    else                                 : Insufficient
```

## 4. Code walkthrough — the lines that matter

**Similarity becomes a support signal only (`aggregators.py:74`).**
```python
if v.similarity is not None:
    return (float(v.similarity) + 1.0) / 2.0, None, None
```
The two `None`s are the whole point. Returning `0.0` for contradiction would let an
aggregator conclude "no contradiction detected" from a signal that is structurally
incapable of detecting one. `None` propagates into every branch as "this class does not
exist for this input", and the label set shrinks accordingly.

**The empty case is shared, so it is comparable (`aggregators.py:85`).**
```python
"explanation": (
    "Retrieval returned nothing for this claim ... This CANNOT distinguish 'the corpus "
    "does not settle this claim' from 'the corpus settles it and the retriever missed "
    "it' -- compare against the oracle run to tell them apart."
),
```
Confidence is `0.0`, not `1.0`: we are maximally unsure, not certainly-insufficient.
All five share this function so that "what happens with no evidence" is not a
per-aggregator accident, and the explanation points at the oracle run — which is the
only thing that *can* distinguish the two cases.

**Max-entailment is decided by exactly one piece of evidence (`aggregators.py:134`).**
```python
best_s = max(v, key=lambda x: support_contra_neutral(x)[0])
```
Its strength and its weakness are the same line. `test_max_entailment_is_decided_by_one_piece_of_evidence`
pins the failure: ten neutral verdicts give `Insufficient`; add one spurious 0.95
entailment and the verdict flips to `Supported`, because nothing else is consulted.

**Noisy-OR's independence assumption is false and the code says so
(`aggregators.py:221`).**
```python
p_s *= (1.0 - s)
```
Retrieved evidence is heavily correlated — the FEVER demo routinely returns three
sentences from the same page. Five near-duplicates at $p=0.3$ give
$1 - 0.7^5 = 0.832$, as though five independent witnesses had spoken.
`test_noisy_or_saturates_with_k_on_duplicate_evidence` shows the same evidence
producing `Insufficient` at k=1 and `Supported` at k=5. **This makes noisy-OR
k-dependent in a way max-entailment is not**, which is a direct, cheap experiment
rather than a defect.

**Weights come from rank, not score (`aggregators.py:304`).**
```python
weights = [1.0 / (float(x.evidence_rank or (i + 1)) ** self.alpha) for i, x in enumerate(v)]
```
Step A established that retriever scores are not comparable across retrievers or even
across queries: BM25 is unbounded and query-dependent, cosine is bounded, RRF lives on a
third scale. Weighting by score would make this aggregator's behaviour silently depend
on which retriever produced the evidence. `test_weighted_by_retrieval_uses_rank_not_score`
inverts the scores relative to the ranks so that a score-based implementation would
produce a different label. `evidence_score` is still recorded in the trace so the choice
can be revisited with data rather than by argument.

**Abstain sets the flag and says what it means (`aggregators.py:436`).**
```python
label, confidence, driver = Label.ABSTAIN, max(s, c), (best_s if s >= c else best_c)
abstained = True
```
`ClaimVerdict` enforces `label == ABSTAIN ⇒ abstained`. The explanation appends a
sentence spelling out that declining to answer is not the same as asserting
insufficiency — because that distinction is the one a reader of the UI will otherwise
collapse.

**The null baseline discards magnitudes (`aggregators.py:517`).**
```python
choice = max(triple, key=lambda k: triple[k])
votes[choice].append(x.evidence_id)
```
A 0.99 and a 0.34 count the same.
`test_a_real_aggregator_disagrees_with_majority_on_the_same_evidence` pins the case
where two marginal supports outvote one overwhelming contradiction — majority says
`Supported`, max-entailment says `Contradicted`. If a real aggregator ever *stops*
beating this baseline, the differences between the real ones are not about weighting
confidence, and the aggregation slot is not where the research question lives. That is a
finding, and a cheap one.

## 5. Data flow

```
Verifier ──► list[EvidenceVerdict] ─┬──► max_entailment        ─┐
   (k forward passes, ONCE)         ├──► noisy_or               │
                                    ├──► weighted_by_retrieval  ├──► ClaimVerdict
                                    ├──► threshold_abstain      │      label
                                    └──► majority (null)       ─┘      confidence
                                                                       abstained
                                                                       aggregation_trace
                                                                          rule
                                                                          explanation      ← Aggregation panel
                                                                          decisive_evidence_ids
                                                                          signal
                                                                       per_evidence        ← Verification panel
```

All five consume the same list. Comparing four aggregators costs one NLI run, not four.

## 6. How to verify it

```powershell
python -m pytest tests/test_aggregators.py -v
```
Expected: **63 passed** in under a second. No models, no corpus.

Every expected number is hand-computed in the test:

| Test | Hand-computed expectation |
|---|---|
| `test_noisy_or_matches_the_formula` | support $=1-(0.7\cdot0.6\cdot0.5)=0.79$ |
| `test_weighted_by_retrieval_uses_rank_not_score` | $(1\cdot0.9 + 0.5\cdot0.1)/1.5 = 0.6\overline{3}$ |
| `test_noisy_or_saturates_with_k_on_duplicate_evidence` | $1-0.7^5 = 0.832$; label flips I → S on identical evidence |
| `test_majority_counts_votes_not_magnitudes` | 2 × 0.34 beats 1 × 0.99 |

Seven properties are parameterised across all five aggregators, so a *new* aggregator
gets them for free by being added to `ALL_AGGREGATORS`: empty-evidence handling,
no-Contradicted-from-similarity, signal flagging, substantive explanation, valid
confidence, `per_evidence` passthrough, real decisive ids, and purity.

Live check that they genuinely differ:

```powershell
python -c "from src.components.aggregators import *; from src.core.types import *; c=Claim.new('r','x','t'); v=[EvidenceVerdict(c.id,'ev::A::0',0.34,0.33,0.33,None,'t',1.0,1,1.0),EvidenceVerdict(c.id,'ev::B::0',0.34,0.33,0.33,None,'t',1.0,2,1.0),EvidenceVerdict(c.id,'ev::C::0',0.005,0.99,0.005,None,'t',1.0,3,1.0)]; [print(a.__name__, a().aggregate(c,v).label.value) for a in (MajorityAggregator, MaxEntailmentAggregator)]"
```
Expected: `MajorityAggregator Supported` / `MaxEntailmentAggregator Contradicted`.

## 7. Limitations and failure modes

* **Every floor is an untuned placeholder.** `support_floor=0.5`,
  `contradict_floor=0.5`, `abstain_below=0.4`, `decision_floor=0.5`, `alpha=1.0`. None
  was chosen by measurement, none may be tuned on dev or test, and no number produced
  with them means anything yet.
* **Three of the five map NLI-`neutral` to `Insufficient`.** "The model was unsure" and
  "the corpus does not settle it" are different propositions. Each says so in its `rule`
  string, but the conflation is real and it is the most likely source of an inflated NEI
  score.
* **Observed on the FEVER debug corpus:** on the NEI example
  `fever-dev-6900`, four aggregators return `Insufficient` and `threshold_abstain`
  returns `Abstain` — scored against gold that only has three classes, the abstaining
  one is "wrong". **A benchmark with no abstain class structurally penalises abstention**,
  which is worth knowing before any risk–coverage work.
* **Noisy-OR is k-dependent by construction** (see §4). Comparing it against the others
  at a single k is not a fair comparison; comparing across k is the actual experiment.
* **`weighted_by_retrieval` falls back to list position** when `evidence_rank` is
  `None`. That only happens if a verifier bypasses `make_verdict`, which nothing does —
  but the fallback is silent rather than loud, which is the wrong side to err on.
* **No aggregator uses `similarity` and NLI together.** A verdict carries one or the
  other, so the "is cosine redundant once a cross-encoder sees the pair?" ablation needs
  a verifier that emits both — which does not exist yet.
* **Confidence is not calibrated and is not claimed to be.** `max_entailment` returns a
  raw softmax probability, `majority` returns a vote fraction, `noisy_or` returns a
  saturating product. These are not comparable to each other, and none has been checked
  against observed accuracy. Any ECE work has to start by picking one and calibrating it.
* **`decisive_evidence_ids` means different things per aggregator** — the argmax driver,
  everything above a 0.1 contribution, or the top half of weight mass. Useful per panel,
  not comparable across aggregators.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Letting the verifier pool over k** | The design flaw the whole repo is arranged against (ADR-002). Pooling would become an implementation detail of the NLI wrapper — untraceable, unswappable, and confounded with the checkpoint. Also, all five aggregators could not then share one set of forward passes. |
| **Passing `list[Evidence]` into `aggregate`** | Would give rank and score directly, but breaks aggregator purity: an aggregator holding evidence can reach the corpus text, and the next step is one that re-scores. Denormalising rank/score into the verdict keeps the signal visible in the trace instead. |
| **A learned aggregator (attention / set-transformer) now** | The genuinely interesting option and probably where the research question lives — but it needs a training split and a tuning protocol, which is a research step, not a placeholder. These five are the baselines it would have to beat. |
| **A single "fusion MLP over pooled scalars"** | Logistic regression with extra steps. It also destroys the property that makes this module useful: you cannot read *why* it decided, so `aggregation_trace` would degenerate to feature values. |
| **Dropping `majority` as filler** | It is the aggregation analogue of `claim_only`. If the real aggregators cannot beat a magnitude-blind vote, that is the finding. |
| **Collapsing `Abstain` into `Insufficient`** | They are different propositions (§1.2) and only one belongs on a risk–coverage curve. Keeping them separate costs one enum member. |
| **Softmax-normalising noisy-OR outputs into a distribution** | Would make the numbers look comparable to the other aggregators' while hiding the saturation that is noisy-OR's defining behaviour. |
| **Weighting by retrieval score instead of rank** | Makes the aggregator's behaviour depend on which retriever ran (§4). The scores are recorded so this can be revisited empirically. |
