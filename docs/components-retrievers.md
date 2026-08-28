# `src/components/retrievers.py` — BM25, dense, hybrid (RRF)

## 1. Problem

Given a claim, produce the *k* corpus sentences most likely to settle it. Three
placeholder implementations, chosen to span the obvious design axis (lexical / dense
/ fused) rather than to be good.

The real problems this module has to not fall into:

1. **Index/corpus drift.** A FAISS index is an anonymous matrix. Nothing in it
   records which corpus it was built from, so a rebuilt corpus and a stale index
   produce confidently wrong evidence text at plausible scores.
2. **Incomparable scores.** BM25 scores are unbounded and depend on the query's IDF
   mass; cosine scores live in [−1, 1]. Adding them requires a normalisation, and
   every available normalisation introduces a query-dependent distortion.
3. **Losing per-arm detail in fusion.** The Retrieval panel must show BM25 score,
   dense score, and fused rank *separately*. A fusion that emits one number has
   destroyed the thing you wanted to look at.

## 2. Formal I/O

All three implement `Retriever.retrieve(claim: Claim, k: int) -> list[Evidence]`,
with the contract from `core-interfaces.md`: `len ≤ k`, ranks `1..len` contiguous,
ids unique.

| | `score` semantics | range | `retriever_meta` keys |
|---|---|---|---|
| `bm25` | Okapi BM25, query-dependent scale | `[0, ∞)` | `bm25_score`, `bm25_rank` |
| `dense` | cosine similarity | `[−1, 1]` | `dense_score`, `dense_rank` |
| `hybrid` | RRF score | `(0, n_arms/(K+1)]` | union of arms' + `rrf_score`, `fused_rank`, `n_arms_hit` |

**BM25.** With $f(q_i, D)$ the term frequency in sentence $D$, $|D|$ its length and
$\overline{|D|}$ the mean:

$$\mathrm{score}(D,Q)=\sum_{i=1}^{n}\mathrm{IDF}(q_i)\cdot\frac{f(q_i,D)\,(k_1+1)}{f(q_i,D)+k_1\left(1-b+b\frac{|D|}{\overline{|D|}}\right)}$$

Defaults $k_1 = 1.5$, $b = 0.75$ — the `rank_bm25` conventional values, **untuned**.
Tokenisation is `[a-z0-9]+` after lowercasing: no stemming, no stopword list, digits
kept (dates and numbers are a named failure slice).

**Dense.** Embeddings $e(\cdot) \in \mathbb{R}^{384}$ from `BAAI/bge-small-en-v1.5`,
L2-normalised, so with `IndexFlatIP`:

$$\mathrm{score}(D,Q)=\langle e(\texttt{prefix}+Q),\ e(D)\rangle=\cos\big(e(\texttt{prefix}+Q),e(D)\big)$$

dtype `float32` (FAISS requirement), index `IndexFlatIP`, exact search.

**Hybrid (RRF).**

$$\mathrm{RRF}(d)=\sum_{a\in\text{arms}}\frac{1}{K+r_a(d)},\qquad K=60$$

where $r_a(d)$ is $d$'s 1-based rank in arm $a$, and $d$ absent from an arm
contributes nothing. Each arm is asked for `candidates_per_arm` (default 50) before
fusion, then the fused list is truncated to `k`.

## 3. Algorithm

```
BM25.retrieve(claim, k):
    q ← tokenize(claim.text)
    scores ← bm25.get_scores(q)              # len == len(corpus), index == corpus row
    order  ← argsort by (-score, row)        # row breaks ties -> deterministic
    return [Evidence.new(corpus.at(row), rank=i) for i, row in enumerate(order[:k], 1)]

Dense.retrieve(claim, k):
    v ← encode([query_prefix + claim.text], normalize=True).astype(float32)
    scores, rows ← index.search(v, min(k, len(corpus)))
    skip rows < 0                            # FAISS pads with -1
    return [Evidence.new(corpus.at(row), score, rank=i) ...]

Hybrid.retrieve(claim, k):
    for arm in arms:
        for hit in arm.retrieve(claim, candidates_per_arm):
            fused[hit.id]  += 1 / (K + hit.rank)          # de-dup by corpus identity
            best_rank[hit.id] = min(best_rank[hit.id], hit.rank)
            meta[hit.id].update(hit.retriever_meta)       # keep BOTH arms' scores
    order ← sort by (-fused, best_rank, id)               # fully deterministic
    return top-k with meta + rrf_score + fused_rank + n_arms_hit
```

## 4. Code walkthrough — the lines that matter

**The index manifest (`retrievers.py:227`).** The most important thing in the
file:
```python
manifest_path.write_text(json.dumps({
    "corpus_fingerprint": self.corpus.fingerprint(),
    "model_name": self.model_name, "dim": self.dim,
    "n_sentences": len(self.corpus), "normalized": True, ...
}))
```
and on load (`retrievers.py:244`), every one of those is compared and any
mismatch raises `stale FAISS index`. Without this, the only symptom of a
corpus/index mismatch is wrong evidence text — which reads like a bad retriever, not
like a bug. The index filename also embeds the fingerprint, so two corpora coexist
rather than overwriting each other.

**Row order is the contract (`retrievers.py:214`).**
```python
vectors = self.encoder.encode(self.corpus.texts(), ...)
```
`corpus.texts()` is row order, and FAISS assigns row *i* to the *i*-th vector added.
That is the entire mapping. `retrieve` then does `self.corpus.at(row)` — never a
separate lookup table, because a separate table is a second thing to keep in sync.

**`.astype(np.float32)` (`retrievers.py:220`).** FAISS requires float32. Passing
float64 does not raise a clear error — depending on the binding it either throws
something opaque or silently reinterprets memory. Explicit cast, always.

**`normalize_embeddings=True` (`retrievers.py:217`).** This is what makes
`IndexFlatIP` a *cosine* index. Drop it and inner product becomes an unnormalised dot
product, which ranks long sentences higher and puts `score` outside [−1, 1] —
`EvidenceVerdict` would then reject the similarity downstream, which is at least a
loud failure, but the retrieval ranking would already be silently wrong.

**The BGE query prefix (`retrievers.py:163`).**
```python
query_prefix: str = "Represent this sentence for searching relevant passages: "
```
The BGE family is trained asymmetrically: the instruction goes on the **query** side
only, passages are embedded bare. Prefixing both sides or neither is a common silent
misuse that costs retrieval quality without any error. It is a constructor param so
the ablation is a config change.

**Deterministic tie-breaking (`retrievers.py:124` and `:390`).**
```python
order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))      # BM25
order = sorted(fused, key=lambda eid: (-fused[eid], best_rank[eid], eid))     # RRF
```
Ties are common (BM25 gives many sentences exactly 0.0; RRF gives equal scores to
equal rank-sums). Without an explicit tiebreak the order depends on dict iteration
and on numpy's sort stability, so two identical runs can differ — which would make
every "did my change do anything?" comparison unreadable.

**Fusion de-duplicates on `Evidence.id` (`retrievers.py:385`).**
```python
fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (self.rrf_k + hit.rank)
meta.setdefault(hit.id, {}).update(hit.retriever_meta)
```
Only sound because `Evidence.id` is corpus identity and nothing else (ADR-003). If
the id included the retriever name, the same sentence from two arms would be two
entries and RRF would rank it *lower* than either arm did — the exact opposite of
what fusion is for. The `meta.update` is what preserves both arms' scores for the
panel.

**Arms must share a corpus (`retrievers.py:362`).**
```python
fingerprints = {arm.corpus.fingerprint(): type(arm).__name__ for arm in self.arms ...}
if len(fingerprints) > 1: raise ValueError("hybrid arms are built over different corpora")
```
Two corpora with different sentence numbering produce colliding `Evidence.id`s that
mean different sentences. RRF would then "agree" about a document neither arm
returned. `corpus_path` is injected into arms that accept it as an ergonomic
convenience, but the *guarantee* is this check, not the injection.

## 5. Data flow

```
configs/*.yaml  components.retriever: {name: hybrid, params: {rrf_k: 60, candidates_per_arm: 50}}
        │ registry.build("retriever", spec)
        ▼
HybridRetriever ──builds──► BM25Retriever ──┐
                └─────────► DenseRetriever ─┤ both share load_corpus_cached(path)
                                            ▼
   claim ──► arm.retrieve(claim, 50) ──► list[Evidence] (per-arm rank + score in meta)
                                            │
                                    RRF fuse + de-dup by id
                                            ▼
                             list[Evidence] (rank 1..k, score = RRF, meta = both arms)
                                            │
                     mark_gold(...) ──► Retrieval panel: BM25 col, dense col, fused rank,
                                        gold highlighted, Recall@k for this example
                                            ▼
                                     Reranker ──► Verifier
```

Shapes: dense encode is `(1, 384) float32` per query, index is `(N, 384) float32`
(115 × 384 × 4 B ≈ 177 KB for the mini corpus; ~7 MB per 5k sentences).

## 6. How to verify it

```powershell
python -m pytest tests/test_retrievers.py -v
python -m pytest tests/test_retrievers.py -v -m slow    # dense; needs the model
```

Expected: **20 passed** with the full stack installed. Without `rank_bm25`, `faiss`
and `sentence-transformers`, 14 pass and 6 skip. A skip is not a pass — `pytest -rs`
prints the reason, and every reason here is a missing dependency, never a disabled
check.

The RRF arithmetic is verified by hand rather than by eyeball. With fake arms A
(D1, D2, D3 at ranks 1, 2, 3) and B (D3, D4 at ranks 1, 2), at K = 60:

| doc | arms | RRF | expected rank |
|---|---|---|---|
| D3 | both | 1/63 + 1/61 = 0.031268 | 1 |
| D1 | A only | 1/61 = 0.016393 | 2 |
| D2 | A only | 1/62 = 0.016129 | 3 (tie, broken by best arm rank) |
| D4 | B only | 1/62 = 0.016129 | 4 |

| Test | Failure signature |
|---|---|
| `test_rrf_scores_match_the_formula` | fusion is doing something other than RRF; the constant or the rank base is wrong |
| `test_rrf_deduplicates_a_sentence_seen_by_both_arms` | a sentence both arms found appears twice and ranks *below* single-arm hits |
| `test_bm25_evidence_text_matches_the_corpus_at_that_key` | row→sentence mapping is off; every result is the wrong text |
| `test_dense_index_is_aligned_and_refuses_a_stale_manifest` | a stale index loads and mis-maps every row |
| `test_ties_are_broken_deterministically` | two identical runs differ; no comparison between configs is readable |

Manual smoke test once the venv is complete:

```powershell
python -c "from src.core.registry import build; from src.core.types import Claim; r=build('retriever',{'name':'bm25','params':{'corpus_path':'data/debug/mini/corpus.jsonl'}}); [print(e.rank, round(e.score,2), e.id, e.text[:60]) for e in r.retrieve(Claim.new('r','Marie Curie was born in Warsaw.','manual'), 5)]"
```

## 7. Limitations and failure modes

* **Nothing here is tuned and the corpus makes recall meaningless.** `k1`, `b`,
  `rrf_k`, `candidates_per_arm`, and the choice of `bge-small` are all placeholders.
  The mini corpus contains gold evidence for its own examples by construction, so a
  high Recall@k describes the corpus, not the retriever.
* **`IndexFlatIP` is exact and O(N) per query.** Correct at a few thousand sentences
  (microseconds), hopeless at 25 M. The full-corpus path needs IVF-PQ or a two-stage
  page→sentence design, and that changes recall in a way that must be *measured*, not
  assumed.
* **The entire index is rebuilt on any corpus change.** No incremental add. Fine at
  this scale; at full scale a rebuild is hours.
* **Query encoding is one-at-a-time.** `retrieve` takes a single claim, so a response
  with 8 claims does 8 separate forward passes. Deliberate (per-claim latency stays
  honest and inspectable), and the obvious first optimisation if the UI feels slow.
* **BM25 holds a tokenised copy of the corpus in memory** — roughly the corpus size
  again. At full scale, `rank_bm25` is the wrong tool entirely; that is a Pyserini or
  a sqlite-FTS decision, not a tuning decision.
* **`load_corpus_cached` is keyed on the path string, not on file contents.** Edit
  the corpus file mid-process and the cache serves the old one. It is an
  `lru_cache(maxsize=8)`, so it also silently evicts on the ninth distinct path.
* **RRF discards magnitude.** It cannot distinguish "arm A is certain" from "arm A is
  guessing", and cannot express "both arms scored everything terribly" — which is
  exactly the signal a *retrieval-failure* detector would want. `retriever_meta`
  keeps the raw per-arm scores precisely so that this is recoverable later.
* **Observed on the mini corpus, n = 1, recorded not acted on.** For `mini-001`
  ("Marie Curie was born in Warsaw."), BM25 puts the gold sentence at rank 1 and the
  distractor *"He was the father of Marie Curie."* at rank 2; dense puts the
  distractor at 2 and gold at 4. RRF therefore ranks the **distractor above the
  gold**, because a document both arms rank 2nd beats a document ranked 1st by one arm
  and 4th by the other:

  | | bm25 rank | dense rank | RRF |
  |---|---|---|---|
  | "He was the father of Marie Curie." | 2 | 2 | 1/62 + 1/62 = 0.03226 |
  | gold: "She was born in Warsaw…" | 1 | 4 | 1/61 + 1/64 = 0.03202 |

  This is RRF behaving exactly as specified — agreement beating one arm's confidence —
  and producing a worse ordering on this example. One example on a 40-document toy
  corpus is not evidence about RRF; it is evidence that the instrument shows you what
  fusion did, which is what it is for. Chasing it would be tuning, which is out of
  scope. It belongs in `docs/EXPERIMENT_BACKLOG.md` as a hypothesis about when
  rank-agreement fusion underperforms its best single arm, with `K` and the
  arm-disagreement magnitude as the axes.
* **`n_arms_hit` is computed by looking for the literal keys `bm25_rank` and
  `dense_rank`.** A third arm with different meta keys will not be counted. It is a
  panel affordance, not a general mechanism.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Score normalisation fusion (min-max or z-score, then weighted sum)** | The only normalisation available at query time is over the candidate list itself, which makes the fused score depend on the *spread* of the candidates: a query where every candidate is irrelevant has its best candidate rescaled to 1.0 exactly as a query with a perfect match does. RRF uses only ranks and is invariant to that. It is also one fewer weight to not-tune. |
| **CombSUM/CombMNZ over raw scores** | Same incomparable-scale problem, plus it lets BM25's unbounded scale dominate cosine outright. |
| **A learned fusion (LambdaMART, or a small MLP over score features)** | Needs training data and a held-out split, which is a research step, not a placeholder. Also exactly the "logistic regression with extra steps" trap flagged in the brief. |
| **IVF-PQ or HNSW instead of flat** | Introduces approximate-search recall loss as a confound at a scale where exact search costs microseconds. Its recall-vs-memory curve is a real experiment for the full corpus, and belongs in the backlog rather than in a placeholder. |
| **`faiss-gpu`** | No official Windows wheel, and a 6 GB card is better spent on the NLI model than on an index that fits in RAM. |
| **Storing embeddings in a numpy array and doing `argsort` by hand** | Genuinely simpler at 115 sentences and avoids a dependency — but the whole point is that the interface survives to full-corpus scale, and hand-rolled search does not. |
| **Pyserini (real Lucene BM25)** | Better BM25 and the standard baseline in this literature, but it needs a JVM, and Windows setup is a known source of pain. `rank_bm25` is a placeholder that keeps the interface honest; swapping it later is a one-class change. |
| **`e5-small-v2` / `all-MiniLM-L6-v2` instead of `bge-small-en-v1.5`** | All three fit comfortably. BGE picked arbitrarily among them; choosing between them is a backlog experiment, not a build decision. The `query_prefix` param exists because they differ in exactly this convention. |
| **Sentence-level index over the full corpus** | ~25 M sentences × 384 × 2 B ≈ 19 GB, which does not fit in 16 GB RAM alongside anything else. This is why the full-corpus plan is a two-stage page→sentence design, and why it is a separate costed step. |
