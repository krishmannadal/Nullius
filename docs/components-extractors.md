# `src/components/extractors.py` and `rerankers.py`

Two small modules documented together: they sit either side of retrieval and share one
concern — **not silently corrupting what they pass along.**

## 1. Problem

**Extractors.** Turn a response into claims. Two implementations, and the honest
statement up front: **decomposition quality here is entirely unmeasured.** Neither has
been evaluated against anything.

That matters more than it sounds, because decomposition errors do not stay in the
decomposition stage:

* A sentence split that leaves two facts welded together ("Curie was born in Warsaw and
  won two Nobel Prizes") forces the verifier to emit **one** label for **two**
  propositions that may differ in truth value. The resulting error is then attributed to
  the verifier in every downstream table.
* An LLM decomposition that silently drops a clause removes a claim from the denominator
  entirely — the pipeline scores perfectly on a claim it never checked.

Both are *upstream* error sources that masquerade as *downstream* ones. The frontend's
editable-claims panel exists precisely so this can be probed by hand before an
experiment is committed to.

The second extractor problem is the **span contract**: `Claim.source_span` must index
the *original* response string. Normalise whitespace, split the normalised copy, report
offsets into it, and the frontend highlights the wrong text with no error anywhere.

**Rerankers.** Reorder and truncate a candidate list. The contract is that a reranker
may reorder and drop but may **not** create, substitute, or edit.

## 2. Formal I/O

```
ClaimExtractor.extract(response: str) -> list[Claim]
Reranker.rerank(claim: Claim, ev: list[Evidence], k: int) -> list[Evidence]
```

| component | behaviour | key metadata |
|---|---|---|
| `spacy_sentence` | one claim per sentence, no decomposition | `decomposed: False`, `sentence_index` |
| `llm_cached` | decomposition served from a JSON cache | `cache_hit`, `decomposed`, `span_is_exact`, `fallback` |
| `noop` | first k, ranks renumbered `1..k` | — |
| `cross_encoder` | reorder by relevance logit | `rerank_score`, `rank_before_rerank`, `rank_delta` |

Post-conditions (enforced by `check_claims`, `check_evidence_list`,
`check_rerank_is_subset`): claim ids unique; `span.text_from(response)` a substring of
the response; reranker output ids ⊆ input ids; ranks 1-based and contiguous.

**Cache format** — a bare mapping is accepted, so a hand-written file needs no
boilerplate:

```json
{"<blake2b-16 of the response text>": ["claim one.", "claim two."]}
```

## 3. Algorithm

```
SpacySentenceExtractor.extract(response):
    doc ← nlp(response)                       # response passed through UNMODIFIED
    for i, sent in enumerate(doc.sents):
        text  ← sent.text.strip()
        start ← sent.start_char + leading_whitespace_len(sent.text)
        emit Claim(text, span=[start, start+len(text)))   # indexes the ORIGINAL

LLMClaimExtractor.extract(response):
    key ← blake2b(response)[:16]
    if key not in cache:
        on_miss == "error" -> raise KeyError
        on_miss == "call"  -> raise NotImplementedError naming what is missing
        on_miss == "split" -> fall back to sentence split, mark fallback=True
    for text in cache[key], de-duplicated:
        span ← exact substring match, else best word-overlap chunk, else None
        emit Claim(text, span, span_is_exact=<bool>)

CrossEncoderReranker.rerank(claim, ev, k):
    scores ← model.predict([(claim.text, e.text) for e in ev])
    order  ← sort by (-score, original_rank)   # ties defer to the retriever
    emit ev[i].reranked(rank=new, score=scores[i]) + {rank_before_rerank, rank_delta}
```

## 4. Code walkthrough — the lines that matter

**The response is never modified (`extractors.py:103`).**
```python
doc = self.nlp(response)
```
No `.strip()`, no whitespace normalisation, no unicode fixing. Everything downstream
indexes this exact string. `test_leading_whitespace_does_not_shift_spans` runs the whole
extraction against `"   " + RESPONSE` and asserts every span still resolves.

**Re-deriving the span of the stripped text (`extractors.py:112`).**
```python
start = sent.start_char + (len(sent.text) - len(sent.text.lstrip()))
```
spaCy's `sent.text` includes leading whitespace but `claim.text` is stripped. Without
this correction the span would be one or two characters wide of the text it claims to
locate — small enough to look like a rendering quirk rather than a bug.
`test_every_span_indexes_the_original_response` asserts
`span.text_from(response) == claim.text` exactly.

**Abbreviations are why the statistical model is the default (`extractors.py:90`).**
```python
self.nlp = spacy.load(model, disable=["ner", "lemmatizer", "attribute_ruler"])
```
`en_core_web_sm` splits `"Dr. Curie worked in Paris, i.e. at the Sorbonne."` as one
sentence; the rule-based `sentencizer` breaks it into three. Those breaks would appear
downstream as *decomposition errors*, contaminating the one thing an extractor
comparison is trying to measure. The parser is kept and NER/tagging are disabled — we
use neither, and they cost time.

**A cache miss must be visible (`extractors.py:196`).**
```python
if self.on_miss == "error":
    raise KeyError(f"no cached decomposition for response {key!r} ...")
```
The default (`"split"`) falls back and marks `fallback: "sentence_split"` in the
metadata. In an experiment that is wrong — a silent fallback puts *two different
extractors* in one results table — so `on_miss="error"` exists to make the mixing
impossible. `on_miss="call"` raises a `NotImplementedError` that names every piece that
would have to be built, rather than pretending an API client exists.

**Rewritten claims get an approximate span, labelled as approximate
(`extractors.py:249`).**
```python
"span_is_exact": span is not None and span.text_from(response) == text,
```
A decomposed claim usually is *not* a substring of the response — that is what
decomposition means. `_locate` tries an exact match, then falls back to best word
overlap against sentence-ish chunks. The result is a guess, and the flag says so, so the
UI can render it differently rather than implying a precision it does not have.

**Cache de-duplication (`extractors.py:234`).**
```python
if not text or text in seen:
    continue
```
LLM decomposers repeat themselves. Two identical claim texts with the same span produce
the same content-addressed id, which would then fail `check_claims`' uniqueness
assertion. Dropping the duplicate is right; the alternative is a crash on perfectly
ordinary model output.

**`noop` renumbers ranks (`rerankers.py:47`).**
```python
e.reranked(rank=i, score=e.score, retriever_name=e.retriever_name)
for i, e in enumerate(ev[:k], start=1)
```
Not cosmetic. Truncating to k without renumbering leaves ranks like `[1, 2, 3]` from a
list that was `[1, 2, 3, 4, 5]` — fine — but a reranker that *dropped* item 2 would
leave `[1, 3, 4]`, violating the contiguity contract that `WeightedByRetrievalAggregator`
and RRF both read as position.

**Ties defer to the retriever (`rerankers.py:105`).**
```python
order = sorted(range(len(ev)), key=lambda i: (-float(scores[i]), ev[i].rank))
```
Cross-encoder scores tie more often than you would expect on short sentences. Breaking
ties by *original rank* means the reranker never reorders on noise, and the result is
deterministic across runs.

**The pre-rerank position is preserved (`rerankers.py:130`).**
```python
"rank_before_rerank": original.rank,
"rank_delta": original.rank - new_rank,   # +ve = promoted
```
The entire point of a reranker panel is seeing *what moved*. Discarding the old rank
makes that impossible, and it is the one thing the panel cannot recompute.

## 5. Data flow

```
response text ──► ClaimExtractor ──► list[Claim]           ──► check_claims(claims, response)
                                       │ text, source_span, extractor_meta
                                       ▼
                                    Retriever ──► list[Evidence]  (candidates_per_arm)
                                       │
                                       ▼
                                    Reranker ──► list[Evidence]   ──► check_rerank_is_subset
                                       │ rank 1..k, rank_before_rerank, rank_delta
                                       ▼
                                    Verifier
```

## 6. How to verify it

```powershell
python -m pytest tests/test_extractors.py -v
```
Expected: **25 passed** (23 without `sentence_transformers`, which the two
cross-encoder tests need).

| Test | Failure signature |
|---|---|
| `test_every_span_indexes_the_original_response` | the frontend highlights the wrong text, silently |
| `test_abbreviations_do_not_split_sentences` | "Dr." splits a sentence; the break shows up as a decomposition error downstream |
| `test_a_conjunction_survives_intact` | if this ever *fails*, the splitter has started decomposing and the docs are wrong |
| `test_on_miss_error_refuses_to_silently_mix_extractors` | two extractors end up in one results table |
| `test_cross_encoder_obeys_the_subset_contract_and_records_movement` | a reranker inventing or editing evidence |
| `test_noop_renumbers_ranks_after_truncation` | non-contiguous ranks corrupt rank-weighted aggregation |

Live check, showing the conjunction weakness directly:

```powershell
python -c "from src.core.registry import build; e=build('extractor','spacy_sentence'); [print(repr(c.text)) for c in e.extract('Curie was born in Warsaw and won two Nobel Prizes. Dr. Curie worked in Paris, i.e. at the Sorbonne.')]"
```
Expected: two claims, the first containing both facts.

## 7. Limitations and failure modes

* **Decomposition quality is unmeasured for both extractors.** No comparison against
  hand-written gold claims exists. Until it does, any claim-level metric carries an
  unknown amount of decomposition error attributed to the verifier.
* **`spacy_sentence` does no decomposition at all** — conjunctions, relative clauses and
  appositives survive intact. It is the honest floor, not a strawman: it is what most
  pipelines actually do when they say "atomic claims".
* **`llm_cached` calls no LLM.** It is an interface plus a cache. The point is that
  swapping in a real client is a one-class change; the point is *not* that decomposition
  is solved here.
* **Span attribution for rewritten claims is a heuristic.** `_locate` splits on `"."`,
  which is wrong for decimals and abbreviations, and matches on word overlap with a
  length filter of >3 characters. It is a UI affordance flagged `span_is_exact: false`,
  not a real alignment.
* **Cache keys are whitespace-sensitive.** Re-indenting a response invalidates its
  cached decomposition. Deliberate — whitespace is part of the input to a decomposer —
  but it will surprise you once.
* **The cross-encoder is a *relevance* model, not an entailment model.**
  `ms-marco-MiniLM-L-6-v2` answers "is this passage relevant to this query?". A sentence
  that flatly contradicts the claim is highly relevant and *should* rank high — which is
  what you want feeding a verifier, but it means **a high reranker score is not evidence
  of support.** Anything reading these scores as a truth signal is misreading them.
* **Reranker scores are unbounded logits**, not probabilities, and not comparable to
  BM25 or cosine. Same rule as everywhere: per-component scales, shown in their own
  column.
* **`spacy_sentence` has no `min_chars` justification.** The default of 3 is arbitrary;
  it exists to drop stray punctuation fragments and has not been checked against
  anything.
* **No sentence-boundary handling for lists or code blocks.** An LLM response containing
  a bulleted list or a code fence will be split badly, and real chat responses contain
  both. This will matter for the browser extension long before it matters here.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Regex/`nltk.sent_tokenize` splitting** | Breaks on "Dr.", "U.S.", "i.e.", decimals. Those breaks masquerade as decomposition errors, contaminating exactly what an extractor comparison measures. |
| **`spacy.blank("en")` + `sentencizer` as the default** | No model download (12 MB saved) but rule-based, so same abbreviation problem. Available via `use_sentencizer=True` for environments that cannot download. |
| **Keeping spaCy's full pipeline** | NER, tagger and lemmatizer cost time and nothing here uses them. |
| **Calling a real LLM for decomposition now** | Requires an API key or a local model, makes the pipeline non-offline and non-deterministic, and puts an unmeasured component on the critical path. The cache gives the same interface with none of that, and lets you hand-write *deliberately bad* decompositions to see how they propagate. |
| **Writing the LLM cache back on a miss** | Would make runs non-reproducible: the same config would behave differently on the second run. A cache that only ever reads is a fixture; one that writes is state. |
| **Storing decomposed claims with no span at all** | The frontend needs *somewhere* to highlight. An approximate span flagged as approximate beats no span, as long as the flag is honest. |
| **`Optional[Reranker]` instead of a `noop` component** | Would put an `if reranker is not None` in the pipeline, so "reranker off" would be a different program rather than a measurement (ADR-008). |
| **A cross-encoder trained for NLI rather than relevance** | Would conflate the reranker with the verifier — the reranker would then be doing the verifier's job on the verifier's inputs, and the ablation "what does reranking add?" would stop being answerable. |
