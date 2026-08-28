# `src/data/corpus.py`, `examples.py`, `metrics.py` — the corpus layer

## 1. Problem

Three failures, all silent, all fatal to the error analysis rather than to the run:

1. **Row/key drift.** The dense index is a matrix; row *i* means "the i-th sentence
   in the corpus's canonical order". Rebuild the corpus with one extra sentence and
   forget to rebuild the index, and every dense result points at the wrong text. No
   exception, no crash, plausible-looking scores, and a Retrieval panel that lies.
2. **Dangling gold keys.** An example says its gold evidence is `(Marie_Curie, 3)`.
   If that key does not exist in the corpus, the *oracle* condition retrieves nothing
   — so the retrieved-vs-oracle gap, the central quantity in the whole project, gets
   computed against a broken oracle and reads as "retrieval was fine".
3. **Hand-counted sentence indices.** Writing "supported by sentence 3" by eye is an
   off-by-one waiting to happen, and it is invisible: the wrong sentence is usually
   from the right document and reads plausibly.

This layer makes (1) a load-time exception, (2) a build-time exception, and (3)
impossible — gold indices are *derived* from a substring match, never typed.

## 2. Formal I/O

**Corpus.** On disk, JSONL, one document per line:

```json
{"doc_id": "Marie_Curie", "title": "Marie Curie", "sentences": ["...", "..."]}
```

| Operation | Signature | Note |
|---|---|---|
| `Corpus.from_jsonl` | `Path → Corpus` | preserves file order |
| `corpus.row(doc_id, sent_id)` | `(str, int) → int` | raises on miss; never returns `-1` |
| `corpus.at(row)` | `int → CorpusSentence` | exact inverse of `row()` |
| `corpus.texts()` | `() → list[str]` | **row order**; what BM25 and the encoder consume |
| `corpus.fingerprint()` | `() → str` | blake2b-128 over ordered keys, 16 hex chars |

Canonical order is *load order*: documents as they appear in the file, sentences as
they appear in the document. `sent_id` is the 0-based index into `sentences`, which
is exactly what `src.core.types.evidence_id(doc_id, sent_id)` addresses. There is no
second sentence-numbering scheme anywhere in the project.

Fingerprint:

$$\mathrm{fp} = \mathrm{blake2b}_{128}\Big(\bigoplus_{i=0}^{N-1} \big(\texttt{doc\_id}_i \Vert \texttt{0x1f} \Vert \texttt{sent\_id}_i \Vert \texttt{0x1f}\big)\Big)[:16]$$

Note what it covers: **addressing, not content**. Reword a sentence and the
fingerprint is unchanged; insert, delete, or reorder one and it changes. That is the
correct sensitivity — the index depends on row→key alignment, not on wording.

**Example.**

| Field | Type | Note |
|---|---|---|
| `text` | `str` | goes in the response box |
| `gold_label` | `Optional[Label]` | `None` = unlabelled |
| `gold_evidence` | `tuple[tuple[str, int], ...]` | keys, never text; empty for NEI |
| `gold_evidence_ids` | `frozenset[str]` | as `Evidence.id`, for set membership |

**Metrics**, per example, over one retrieved list:

$$\mathrm{Recall@}k = \frac{|G \cap R_{1:k}|}{|G|}\ \text{ if } G \ne \emptyset,\quad \text{else undefined}$$

`recall_at_k` and `reciprocal_rank` return **`None`** when `G = ∅`. FEVER's NEI class
has empty evidence by construction, so a third of a FEVER slice has no denominator.
Returning 1.0 flatters the retriever, 0.0 punishes it for a question nobody asked;
both corrupt any mean taken over them.

## 3. Algorithm

```
Corpus.__init__(documents):
    rows ← []
    for doc in documents:                       # file order preserved
        reject duplicate doc_id
        for sent_id, text in enumerate(doc.sentences):
            rows.append(CorpusSentence(doc.doc_id, sent_id, text))
    _row_by_key ← {s.key: i for i, s in enumerate(rows)}     # key -> row
    # row -> key is just rows[i].key. Both directions exist; both are tested.

build_mini_corpus:
    for each example spec (doc_id, needle):
        hits ← [i for i, s in enumerate(corpus.document(doc_id).sentences) if needle in s]
        if len(hits) != 1: FAIL LOUDLY          # absent or ambiguous
        sent_id ← hits[0]                       # derived, never typed
    validate_against(examples, corpus) must report ok
```

## 4. Code walkthrough — the lines that matter

**The two directions of the mapping (`corpus.py:75`).**
```python
self.sentences = tuple(rows)
self._row_by_key = {s.key: row for row, s in enumerate(self.sentences)}
```
`row → key` is `self.sentences[row].key`; `key → row` is the dict. Both are needed
(FAISS returns rows; gold evidence arrives as keys) and
`test_row_and_key_are_exact_inverses_for_every_row` walks every row asserting the
round trip. That test is the reason this class exists rather than a dict of lists.

**Load order is not sorted (`corpus.py:57`, docstring).** Deliberate. Sorting
would silently reorder rows when a `doc_id` gains a diacritic or changes case —
exactly the kind of edit that does not *look* like it should invalidate an index.
Insertion order makes the invalidation visible via the fingerprint.

**Fingerprint hashes keys, not text (`corpus.py:137`).**
```python
h.update(s.doc_id.encode("utf-8")); h.update(_KEY_SEP)
h.update(str(s.sent_id).encode("ascii")); h.update(_KEY_SEP)
```
Same `0x1f` separator discipline as `stable_id`. `sent_id` is stringified rather
than packed as bytes so that the hash is readable-by-reimplementation — you can
recompute it in five lines of any language if you ever need to check an index by
hand.

**`row()` raises instead of returning a sentinel (`corpus.py:96`).** A missing
key is always a bug (a dangling gold key, or an index built from another corpus).
Returning `-1` would let it flow into `at(-1)`, which is a *valid* Python index and
returns the last sentence of the corpus. That is the single worst available failure
mode: confident, wrong, and last-in-file.

**Gold indices are derived (`scripts/build_mini_corpus.py:resolve_gold`).**
```python
hits = [i for i, s in enumerate(doc.sentences) if needle in s]
if not hits:      raise SystemExit("no sentence containing ...")
if len(hits) > 1: raise SystemExit("ambiguous; lengthen the substring")
```
Ambiguity is an error, not a first-match. If a substring matches two sentences, the
author did not know which one they meant, and picking one silently is how a wrong
gold index gets into a results table.

**`mark_gold` sets a real boolean (`metrics.py:32`).**
```python
return [replace(e, is_gold=(e.id in gold)) for e in evidence]
```
Before this call `is_gold` is `None` = *unknown*; after it, `True`/`False` = a real
annotation. The distinction matters in the UI: an example with no gold annotation at
all must not render as "retrieval missed the gold". `replace` returns new frozen
objects, so the input list is untouched — tested.

## 5. Data flow

```
scripts/build_mini_corpus.py ──► data/debug/mini/{corpus,examples}.jsonl
                                        │
Corpus.from_jsonl ──────────────────────┤
   │                                    ▼
   ├─ corpus.texts() ──► BM25Okapi(tokenized)          [row order]
   ├─ corpus.texts() ──► encoder.encode() ──► FAISS    [row order]
   ├─ corpus.at(row) ──► Evidence.new(doc_id, sent_id) [FAISS row -> evidence]
   └─ corpus.fingerprint() ──► index manifest ──► refuse on mismatch

load_examples ──► Example.gold_evidence_ids ──► mark_gold(retrieved) ──► Recall@k panel
                                             └─► oracle substitution (mode="oracle")
```

## 6. How to verify it

```powershell
python -m scripts.build_mini_corpus
python -m pytest tests/test_corpus.py -v
```

Expected build output:

```
  n_docs                   40
  n_sentences              115
  fingerprint              aa7c5b1202fff125
  n_examples               14
  labels                   {'Supported': 8, 'Contradicted': 4, 'Insufficient': 2}
  gold keys resolved       16
```

Expected tests: **17 passed**.

| Test | Failure signature |
|---|---|
| `test_row_and_key_are_exact_inverses_for_every_row` | dense retrieval returns the wrong sentence text for every query, with plausible scores |
| `test_fingerprint_changes_when_a_sentence_is_inserted` | a stale FAISS index loads happily and mis-maps every row |
| `test_fingerprint_ignores_sentence_wording` | every typo fix forces a full index rebuild for no reason |
| `test_mini_corpus_gold_keys_all_resolve` | the oracle condition silently retrieves nothing; retrieval-attributable error reads as zero |
| `test_recall_is_none_when_the_example_has_no_gold_evidence` | NEI examples contribute 1.0 or 0.0 to a recall mean that then means nothing |

Manual alignment check you can run by eye:

```powershell
python -c "from src.data.corpus import Corpus; c=Corpus.from_jsonl('data/debug/mini/corpus.jsonl'); r=c.row('Marie_Curie',1); print(r, repr(c.at(r).text))"
```
Expected: a row number, then `'She was born in Warsaw, in what was then the Kingdom of Poland, part of the Russian Empire.'`

## 7. Limitations and failure modes

* **The mini corpus is 40 documents and 115 sentences. It measures nothing.** It
  exists so tests run offline in milliseconds and so every stage can be watched on a
  real input. Any recall or accuracy computed over it describes the corpus's
  construction, not a method. The larger debug corpus (a few thousand docs) is the
  next artifact and is no better in this respect — see `configs/debug.yaml`'s header.
* **The whole corpus is held in memory as Python strings.** At 115 sentences that is
  nothing; at 25 M FEVER sentences it is roughly 10–15 GB and will not fit in 16 GB.
  The full-corpus path needs a memory-mapped or on-disk store, and `Corpus` will need
  an implementation swap behind the same interface. It is written to make that swap
  possible (everything goes through `at()`/`row()`), not to make it unnecessary.
* **`sent_id` is an `int` and sentences are pre-split in the file.** No sentence
  splitter runs here, so corpus sentence boundaries are whatever produced the file.
  For the mini corpus they are hand-written; for a FEVER-derived corpus they will be
  FEVER's own, which is the right choice because gold indices are defined against them.
* **The fingerprint does not cover sentence text**, by design — so a corpus whose
  wording changed but whose addressing did not will reuse an index built on the old
  wording. Deliberate trade (see §4), but it means "the index is valid" is a claim
  about alignment only. The corpus file's own hash belongs in the run manifest.
* **`validate_against` materialises the example list.** Fine at these sizes; it is
  not a streaming validator.
* **Metrics here are per-example only.** Nothing aggregates, deliberately: the
  `None`s above have to be handled explicitly by whatever does, rather than being
  silently coerced by a `sum()`.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Sorting the corpus by `doc_id` for a canonical order** | Makes order depend on collation, which differs by locale and Python version, and hides invalidating edits behind a stable-looking sort. Insertion order plus a fingerprint makes the invalidation loud. |
| **Storing gold evidence as text rather than keys** | Text drifts out of alignment with the corpus and nothing notices; keys can be *proved* to resolve, which `validate_against` does at build time. |
| **Hand-writing gold `sent_id`s in the example file** | The off-by-one this module exists to prevent. Substring resolution turns a typo into a build error. |
| **First-match substring resolution** | An ambiguous needle means the author did not know which sentence they meant. Failing is correct; picking the first is how a wrong index reaches a table. |
| **`Recall@k = 1.0` for empty gold (vacuous truth)** | Mathematically defensible, practically a lie: it inflates any mean over a set containing NEI examples, and NEI is a third of FEVER. |
| **SQLite for the corpus** | Real gains at full-corpus scale (memory-mapped access, no load time). At a few thousand sentences it adds a schema, a migration story, and a dependency for zero benefit. The `at()`/`row()` interface is what makes the later swap cheap. |
| **`datasets.Dataset` (HuggingFace) as the corpus type** | Arrow-backed and memory-mapped, which is genuinely attractive at scale — but it drags `pyarrow` + `datasets` into the trace-reading path and gives no control over canonical row order, which is the one property this module has to guarantee. |
