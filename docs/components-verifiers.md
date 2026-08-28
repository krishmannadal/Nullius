# `src/components/verifiers.py` — NLI, similarity, and the claim-only null

## 1. Problem

Score one (claim, evidence) pair. Three implementations spanning the design axis, one
of which is built to fail.

The two problems this module actually exists to solve are not modelling problems.
Both were **measured on the real checkpoint before a line of it was written**, and both
are silent:

1. **The label index → name mapping is not what you would guess.** For
   `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` it is
   `{0: entailment, 1: neutral, 2: contradiction}` — the *reverse* of the common MNLI
   convention. Hardcoding index 0 as contradiction swaps Supported and Contradicted for
   every claim in the project, and every probability still sums to 1, still looks like
   plausible model output, and passes every structural check in `types.py`.
2. **`tokenizer.model_max_length` is `1000000000000000019884624838656`.** That is the
   sentinel this tokenizer ships when no limit is configured, and its consequence is
   that **`truncation=True` on its own does nothing at all**. Long evidence then
   produces a sequence far past the position embeddings.

Neither is a hypothetical. `tests/test_verifiers.py` demonstrates both directly.

## 2. Formal I/O

All three implement `Verifier.score(claim: Claim, ev: Evidence) -> EvidenceVerdict`.

| verifier | emits | range | reads evidence? |
|---|---|---|---|
| `nli_deberta` | `p_entail, p_contra, p_neutral` | simplex, sums to 1 ± 10⁻³ | yes |
| `similarity` | `similarity` (probs are `None`) | `[−1, 1]` | yes |
| `claim_only` | `p_entail, p_contra, p_neutral` | simplex | **no — by design** |

**NLI.** With evidence as premise $P$ and claim as hypothesis $H$:

$$(p_{\text{entail}}, p_{\text{neutral}}, p_{\text{contra}}) = \mathrm{softmax}\bigl(f_\theta(P, H)\bigr)\big|_{\pi}$$

where $\pi$ is the permutation read from `model.config.id2label` at load time. Input is
tokenised as a pair with `truncation="longest_first"` and an explicit
`max_length = 256`.

**Similarity.** With $e(\cdot)$ the L2-normalised bi-encoder embedding:

$$\mathrm{sim} = \langle e(\text{claim}), e(\text{evidence})\rangle = \cos\bigl(e(\text{claim}), e(\text{evidence})\bigr)$$

No threshold is applied. Thresholding is an aggregator decision that must land in
`aggregation_trace`.

**Measured on RTX 4050, fp16, DeBERTa-v3-base-mnli-fever-anli:**

| | |
|---|---|
| params | 184.4 M |
| weights VRAM | **359 MiB** |
| peak VRAM, batch 1 × 256 | 375 MiB |
| peak VRAM, batch 16 × 512 | **467 MiB** of 6140 MiB |
| forward latency | 50–95 ms |

VRAM is a non-issue for this checkpoint. That was worth measuring rather than assuming
— the brief required it, and the answer (7.6% of the card at the largest batch tested)
means the constraint binds somewhere else entirely.

## 3. Algorithm

```
_resolve_label_indices(id2label):          # called ONCE at load
    lowered ← {int(i): name.lower() for i, name in id2label.items()}
    for name in (entailment, neutral, contradiction):
        matches ← [i for i, got in lowered if got == name]
        if len(matches) != 1: RAISE            # refuse to guess
    return {name: index}

NLIVerifier.score(claim, ev):
    enc ← tokenizer(ev.text, claim.text,      # evidence is the PREMISE
                    truncation="longest_first",
                    max_length=256)            # REQUIRED, see §1.2
    logits ← model(enc).logits
    probs  ← softmax(logits.float())           # fp32 softmax, not fp16
    p_e, p_c, p_n ← probs[idx[entailment]], probs[idx[contradiction]], probs[idx[neutral]]
    renormalise, then make_verdict(...)        # copies ev.rank / ev.score

ClaimOnlyVerifier.score(claim, ev):
    return NLI._score_pair(premise=self.premise, hypothesis=claim.text)
    # ev.text is deliberately never read. ev.id still keys the verdict.
```

## 4. Code walkthrough — the lines that matter

**The label mapping is read, never written (`verifiers.py:155`).**
```python
self.label_index = _resolve_label_indices(dict(self.model.config.id2label))
```
Everything downstream indexes through `self.label_index["entailment"]` rather than
through a literal `0`. Swapping to a checkpoint with the opposite convention is then a
config change with no code change, and a checkpoint with `LABEL_0`-style anonymous
heads raises instead of quietly producing a 3-way permutation of the truth.

Why a structural check cannot catch this: a swapped mapping still yields three
probabilities summing to 1, so `EvidenceVerdict`'s simplex validation passes. Only
*behaviour* catches it, which is why
`test_label_mapping_behaves_as_the_names_claim` uses a known-answer probe
(entailed / refuted / neutral triples with obvious correct answers) rather than
inspecting the config it is meant to be validating.

**Explicit `max_length` on every call (`verifiers.py:166`).**
```python
truncation="longest_first",
max_length=self.max_length,   # REQUIRED: model_max_length is ~1e19 here
```
`test_without_explicit_max_length_truncation_is_a_no_op` proves the trap: a 20,000-word
evidence tokenised with `truncation=True` alone comes back at **>10,000 tokens**; with
`max_length` it comes back at exactly 256.

`longest_first` rather than the default is the second half. With
`truncation_side="right"`, naive truncation removes from the end of the concatenated
pair. `longest_first` removes tokens from whichever member is currently longer, so a
30-token claim survives intact against a 4,000-token evidence —
`test_longest_first_truncation_preserves_a_short_claim` asserts the claim text is still
in the decoded input.

**Evidence is the premise (`verifiers.py:184`).**
```python
p_e, p_c, p_n, latency = self._score_pair(ev.text, claim.text)
```
Order matters and is asymmetric. "Does this evidence entail this claim?" is the
question; the reverse is a different and mostly meaningless one for fact verification.
Getting it backwards produces a model that looks *merely bad* rather than
misconfigured, which is the worst kind of wrong.

**Softmax in fp32 (`verifiers.py:174`).**
```python
probs = logits.float().softmax(-1)[0]
```
An fp16 softmax over three logits can round a near-1.0 probability to exactly 1.0 and
the others to 0.0. That still sums to 1, so it passes validation — but it destroys the
calibration information the aggregators and any future ECE analysis depend on.

**`make_verdict` denormalises the retrieval signal (`verifiers.py:82`).**
```python
evidence_rank=ev.rank,
evidence_score=ev.score,
```
`Aggregator.aggregate` receives only verdicts, never evidence. Without this,
`WeightedByRetrievalAggregator` could not exist without either changing the ABC
signature or being handed the corpus — both of which would break the isolation that
makes aggregators comparable. Putting it in one shared helper means all three verifiers
do it identically; if each did it by hand, one would eventually forget and that
aggregator would silently fall back to uniform weights *for that verifier only*.

**`ClaimOnlyVerifier` composes rather than inherits (`verifiers.py:292`).**
```python
self._nli = NLIVerifier(...)
```
It shares the NLI machinery but is emphatically **not a kind of** `NLIVerifier` and must
never be substitutable for one by accident. `ev.text` is never read; `ev.id` still keys
the verdict, so it stays on exactly the same code path and every aggregator works
against it unchanged.

## 5. Data flow

```
Reranker ──► list[Evidence] ──┐
                              │  for each e:
Claim ────────────────────────┴──► Verifier.score(claim, e) ──► EvidenceVerdict
                                                                   │ p_entail/p_contra/p_neutral
                                                                   │ similarity
                                                                   │ evidence_rank / evidence_score  ← denormalised
                                                                   ▼
                                                             Aggregator.aggregate(claim, verdicts)
```

k forward passes per claim, not one batched call — the cost ADR-002 accepted so that
pooling cannot hide inside a model wrapper. At k=5 and ~70 ms that is ~0.35 s per claim.

## 6. How to verify it

```powershell
python -m pytest tests/test_verifiers.py -v
```
Expected: **22 passed** with the checkpoint present; 8 pass and 14 skip without it.

| Test | Failure signature |
|---|---|
| `test_label_mapping_behaves_as_the_names_claim` | Supported and Contradicted swapped across the entire project, with plausible-looking probabilities |
| `test_refuses_to_guess_when_labels_are_anonymous` | a `LABEL_0` checkpoint silently gets an arbitrary 3-way permutation |
| `test_without_explicit_max_length_truncation_is_a_no_op` | long evidence overruns position embeddings |
| `test_longest_first_truncation_preserves_a_short_claim` | the claim is silently cut and the model scores evidence against a fragment |
| `test_claim_only_ignores_the_evidence_entirely` | the null baseline is not null, so any comparison against it is meaningless |
| `test_similarity_cannot_tell_support_from_contradiction` | if this ever *fails*, the structural claim in the docs is wrong and should be rewritten |

Reproduce the VRAM and mapping measurements directly:

```powershell
python -c "from transformers import AutoModelForSequenceClassification as M; print(M.from_pretrained('MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli').config.id2label)"
```
Expected: `{0: 'entailment', 1: 'neutral', 2: 'contradiction'}`.

## 7. Limitations and failure modes

* **Nothing here is tuned or evaluated.** The checkpoint was chosen because the brief
  named DeBERTa-v3-base and it fits; no comparison against alternatives was run.
* **`max_length=256` is a placeholder.** It truncates roughly 1% of FEVER sentences and
  was picked for speed, not measured. A claim whose supporting detail sits past token
  256 of a long evidence sentence is unverifiable by construction, silently.
* **Similarity is structurally incapable of detecting contradiction.** "Marie Curie was
  born in Warsaw" and "…in Paris" are near-identical strings.
  `test_similarity_cannot_tell_support_from_contradiction` demonstrates it rather than
  asserting it. A similarity-only pipeline can never output Contradicted, and the
  aggregators record that in `aggregation_trace["signal"]`.
* **The `claim_only` null uses an empty premise.** An empty string and a fixed
  *irrelevant* premise are different nulls — "no premise at all" vs "a premise exists
  and says nothing" — and they can behave differently. The `premise` parameter exists to
  separate them; neither has been studied.
* **Early signal worth taking seriously:** on `fever-dev-5688` ("Amyotrophic lateral
  sclerosis is a disease") the claim-only null returns **Supported at 0.831 without
  reading any evidence**. n=1, but that is the artifact this baseline exists to expose,
  and it appeared on the first run.
* **No batching.** `score` takes one pair, so a response with 8 claims at k=5 does 40
  separate forward passes. Deliberate (per-pair latency stays honest); the obvious first
  optimisation is an internal micro-batch that still returns per-pair verdicts.
* **`latency_ms` covers tokenise + forward**, not model load and not the `.to(device)`
  transfer of the batch. It is what a caller waits for, which is the useful definition,
  but it is not a clean measure of model compute.
* **fp16 is silently ignored on CPU.** `dtype` falls back to fp32 without warning, so a
  CPU run is slower and numerically different from a GPU run under the same config.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Hardcoding the label indices** | The bug this module is built around. The measured order is the reverse of the common convention, and a wrong guess is invisible. |
| **Trusting `model.config.id2label` without a behavioural test** | A config can be wrong or a checkpoint re-headed. The known-answer probe is three forward passes and catches what the config cannot attest to. |
| **`truncation=True` with no `max_length`** | Measured to be a complete no-op on this tokenizer. |
| **`truncation="only_second"`** (truncate the evidence only) | Defensible — the claim is what you must not lose — but it fails on the rare long claim, and `longest_first` protects the shorter member in both directions without a special case. |
| **Claim as premise, evidence as hypothesis** | Answers a different question and produces a model that looks bad rather than misconfigured. |
| **A verifier that returns a single "support score"** | Collapses entail/contra/neutral into one number, which is exactly the conflation the 3-way contract exists to prevent, and would make the NEI class unrepresentable. |
| **Batching inside `score`** | Cannot: the signature takes one pair, deliberately (ADR-002). Batching belongs in a future `score_batch` that still returns per-pair verdicts. |
| **A larger NLI model (DeBERTa-v3-large)** | 435 M params, ~870 MiB fp16 — would still fit. Not used because nothing here is being optimised, and swapping it is a one-line config change when that becomes the question. |
| **Writing 0.0 into the probability fields for `similarity`** | Would make a similarity verdict indistinguishable from a confidently-neutral NLI verdict in every aggregate. `None` is the honest encoding. |
