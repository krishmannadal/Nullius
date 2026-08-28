# `scripts/build_debug_corpus.py` — FEVER → the debug corpus

## 1. Problem

Turn FEVER dev claims plus the June-2017 Wikipedia snapshot into a corpus small
enough to index on a laptop, **without breaking the sentence addressing that gold
evidence depends on**.

The failure this script is built around is worth stating precisely, because it is the
one guard the rest of the repo cannot provide:

* `Corpus.fingerprint()` detects a corpus whose *shape* changed.
* `validate_against()` detects a gold key that does not *resolve*.
* **Neither detects a gold key that resolves to the wrong sentence.**

A shifted index still resolves, still reads plausibly, and quietly re-points every
gold annotation at its neighbour. Downstream it looks like a bad NLI model, not like a
parse bug. The two ways to cause it are (a) filtering out empty sentences, and (b)
enumerating file positions instead of reading FEVER's declared indices. This script
does neither, and `tests/test_fever_parse.py` checks the parse directly against
fixtures whose correct answer is obvious by eye.

The second problem is honesty about what the result can measure. See §7.

## 2. Formal I/O

**Inputs.**

| File | Size | Source |
|---|---|---|
| `shared_task_dev.jsonl` | 4.35 MB | `https://fever.ai/download/fever/shared_task_dev.jsonl` |
| `wiki-pages.zip` | 1.71 GB | `https://fever.ai/download/fever/wiki-pages.zip` |

(The older `s3-eu-west-1.amazonaws.com/fever.public/*` URLs now return **403**; the
`fever.ai` paths are live as of 2026-08-28.)

A dev record:

```json
{"id": 111897, "verifiable": "VERIFIABLE", "label": "REFUTES",
 "claim": "Telemundo is a English-language television network.",
 "evidence": [[[131371, 146144, "Telemundo", 0]],
              [[131371, 146150, "Telemundo", 4], [131371, 146150, "Hispanic_and_Latino_Americans", 0]]]}
```

`evidence` is `list[group]`, `group = list[[annotation_id, evidence_id, page, sent_id]]`.
**Each group is a complete alternative evidence set**; a group with more than one key
is multi-hop. NEI rows carry `[[id, null, null, null]]`.

A wiki record's `lines` field:

```
"0\tMarie Curie was a physicist.\tMarie Curie\tMarie_Curie\n1\tShe was born in Warsaw.\n2\t"
```

i.e. `\n`-separated records of `{index}\t{sentence}\t{hyperlink annotations…}`.
Record 2 above is a **real, empty, indexed sentence**.

**Output.**

| File | Contents |
|---|---|
| `data/debug/corpus.jsonl` | `Document(doc_id=page, title, sentences)` — `sentences[i]` **is** FEVER's sentence *i* |
| `data/debug/examples.jsonl` | `Example` with `gold_evidence` flattened, `meta.evidence_groups` preserved |
| `data/debug/manifest.json` | seed, counts, parse stats, and the corpus's own warning text |

**Measured dev-set statistics** (whole 19,998-row dev split):

| | |
|---|---|
| rows | 19,998 — exactly 6,666 per class |
| distinct gold pages | 2,892 |
| evidence groups per claim | 1: 9,682 · 2: 1,675 · 3: 561 · 4: 391 · 5: 346 · 6: 200 |
| keys per group | 1: 21,671 (87.6%) · 2: 2,629 · ≥3: 425 |
| groups spanning >1 page | 2,794 / 24,725 = **11.3%** (genuine multi-hop) |

## 3. Algorithm

```
parse_lines_field(raw):                       # index-driven, NEVER enumerate
    by_index ← {}
    for record in raw.split("\n"):
        fields ← record.split("\t")
        idx ← int(fields[0])                  # non-integer -> drop, cannot shift anything
        by_index[idx] ← fields[1] if present else ""
    return tuple(by_index.get(i, "") for i in range(max(by_index) + 1))
                                              # gaps padded, empties preserved

build:
    1. dev ← load(shared_task_dev.jsonl)
       picked ← stratified_sample(dev, n_claims, seed)      # equal per label
       required ← pages named by picked claims' evidence
       other_gold ← gold pages of every OTHER dev claim
    2. ONE streaming pass over wiki-pages.zip (never extracted):
         page in required     -> keep
         page in other_gold   -> keep as distractor
         otherwise            -> reservoir-sample toward n_docs
    3. corpus ← Corpus(required ∪ other_gold ∪ reservoir)   # sorted within each tier
    4. examples ← picked claims whose gold keys ALL resolve; others dropped and named
    5. assert validate_against(examples, corpus).ok
       warn on any gold key whose sentence is empty          # misparse tripwire
    6. write corpus.jsonl, examples.jsonl, manifest.json
```

Reservoir sampling gives a uniform sample over ~5.4 M non-gold pages in a single pass,
so the 1.7 GB zip is read once and never extracted (peak disk stays at the zip).
`parse_lines_field` is called only on pages actually kept — roughly
`n_fill·(1 + ln(N/n_fill)) ≈ 33 k` parses rather than 5.4 M.

## 4. Code walkthrough — the lines that matter

**Index-driven parsing (`build_debug_corpus.py:120`).**
```python
idx = int(fields[0])
by_index[idx] = text
...
size = max(by_index) + 1
out = tuple(by_index.get(i, "") for i in range(size))
```
This is the whole point of the file. The tuple is built by *asking for each index in
turn*, so a missing index becomes `""` in place rather than closing the gap, and a
record arriving out of order lands where it says it belongs. Replace these four lines
with `[f.split("\t")[1] for f in raw.split("\n") if f.strip()]` and everything still
runs, every test outside `test_fever_parse.py` still passes, and every gold annotation
on a page containing a blank line points at the wrong sentence.

**Dropping unindexed fragments is safe *because* of the above
(`build_debug_corpus.py:121`).**
```python
except (ValueError, IndexError):
    stats.non_integer_index += 1
    continue
```
Real records in the dump have continuation text with no leading index. Dropping them
is only harmless under index-driven assembly; under positional enumeration it would
corrupt every sentence after the fragment.

**Evidence groups are preserved, not just flattened
(`build_debug_corpus.py:evidence_groups` + the `meta` block).**
```python
"evidence_groups": [[[p, s] for p, s in g] for g in groups],
```
`gold_evidence` is the flattened union, which is the right denominator for Recall@k.
It is the *wrong* structure for the official FEVER score, which asks "did we recover at
least one **complete** group". Flattening alone would destroy that permanently, so the
grouped form rides along in `meta`.

**Stratified sampling is order-independent (`build_debug_corpus.py:sample_claims`).**
```python
pool = sorted(by_label[label], key=lambda r: r["id"])   # stable BEFORE sampling
picked.extend(rng.sample(pool, ...))
```
Without the sort, the sample depends on the order rows happen to appear in the file,
so a re-download or a re-serialisation silently changes which claims you are looking
at while the seed stays the same. Tested by
`test_sample_claims_does_not_depend_on_input_order`.

**NFC normalisation at both boundaries (`build_debug_corpus.py:145`).**
```python
def normalize_page_id(page_id: str) -> str:
    return unicodedata.normalize("NFC", page_id)
```
Applied in `iter_wiki_pages` (dump side) and in `evidence_groups` (dev side), so every
page title in the system is NFC before anything compares two of them. The dump is
already uniformly NFC — 0 non-NFC out of 49,999 sampled titles, 12,880 of them
non-ASCII — so this is a no-op there and cannot collide two distinct pages into one
`doc_id`. It moves exactly the 32 dev-side references that were NFD. See ADR-023 for
the measurement.

**The misparse tripwire (`build_debug_corpus.py`, step 5).**
```python
empty_gold = [(ex.id, key) for ex in examples for key in ex.gold_evidence
              if not corpus.get(*key).text.strip()]
```
FEVER claims are near-verbatim mutations of their gold sentence, so a gold key
pointing at an *empty* sentence is strong evidence of an index shift. This catches the
class of bug that `validate_against` provably cannot.

**Claims whose gold did not survive are dropped and named, not silently kept.** If a
gold page is missing from the dump, keeping the claim with partial evidence would make
its oracle condition quietly incomplete — and the oracle gap is the central
measurement. The dropped ids go in `manifest.json`.

## 5. Data flow

```
fever.ai ──► data/raw/shared_task_dev.jsonl   (4 MB, committed? no — data/raw/ is gitignored)
         └─► data/raw/wiki-pages.zip          (1.71 GB, never extracted)
                    │
        build_debug_corpus.py  (one streaming pass)
                    │
   ┌────────────────┼──────────────────┐
   ▼                ▼                  ▼
corpus.jsonl   examples.jsonl    manifest.json
   │                │
   │                └─► Example.gold_evidence_ids ─► mark_gold / oracle substitution
   └─► Corpus.from_jsonl ─► BM25Retriever
                         └─► DenseRetriever ─► FAISS index (manifest-guarded by fingerprint)
```

## 6. How to verify it

```powershell
python -m pytest tests/test_fever_parse.py -v     # 19 passed, no downloads needed
```

The parse tests run against fixtures and need neither file, so they are the check you
can run before committing to a 1.7 GB download.

Download and build:

```powershell
curl -L -o data/raw/shared_task_dev.jsonl https://fever.ai/download/fever/shared_task_dev.jsonl
curl -L -o data/raw/wiki-pages.zip        https://fever.ai/download/fever/wiki-pages.zip
python -m scripts.build_debug_corpus --n-claims 200 --n-docs 4000 --seed 1337
```

| Check | Where | Failure signature |
|---|---|---|
| every gold key resolves | build aborts with `dangling gold keys` | oracle silently retrieves nothing; retrieval-attributable error reads as zero |
| no gold key hits an empty sentence | `WARNING: N gold keys point at empty sentences` | index shift; every claim is checked against its neighbour |
| dropped claims are named | `manifest.json → dropped_claims` | claims with partial oracle evidence contaminate the gap measurement |
| parse anomalies are counted | `manifest.json → parse_stats` | a spike in `non_integer_index` means the dump format changed |

Spot-check by eye after building:

```powershell
python -c "from src.data.corpus import Corpus; from src.data.examples import load_examples; c=Corpus.from_jsonl('data/debug/corpus.jsonl'); e=load_examples('data/debug/examples.jsonl')[0]; print(e.text); [print(' GOLD:', c.get(*k).text) for k in e.gold_evidence]"
```
The claim and its gold sentence should be recognisably about the same fact. If they
are unrelated, you are looking at an index shift, not a hard example.

## 7. Limitations and failure modes

* **Recall over this corpus is an artifact of its construction.** The gold page for
  every shipped claim is present by design, and the distractor pool is not
  adversarial. This is exactly the "reduced corpus" option the project brief calls
  *the cheapest and the most dangerous*. It is a debug harness. Numbers computed over
  it describe the assembly, not a method, and `manifest.json` carries that sentence
  inside the artifact so it travels with the data.
* **`is_gold == False` means "not annotated for this claim", never "irrelevant".**
  FEVER's annotations are known to be incomplete, and the distractor pool deliberately
  contains pages that are gold for *other* claims, so genuinely supporting sentences
  will appear unmarked. Any precision-style metric over `is_gold` inherits this.
* **FEVER claims are human-written mutations of Wikipedia sentences, not LLM output.**
  So this measures *fact verification*, not *hallucination detection*, and a
  decomposition stage has almost nothing to do on a single-sentence FEVER claim. This
  is a named threat to external validity, not a footnote.
* **NEI claims have no gold evidence at all**, so for a third of the sample the oracle
  condition is empty and Recall@k is `None`. Whether the corpus happens to contain
  sentences that would settle an NEI claim is unknown and unannotated — which is
  precisely the "genuine insufficiency vs retrieval failure" boundary, and is not
  resolved by this corpus.
* **Flattened gold is wrong for the FEVER score.** `gold_evidence` is the union over
  groups; the official metric needs at least one complete group. `meta.evidence_groups`
  preserves what is needed, but nothing computes the FEVER score yet.
* **The dump is a 2017 snapshot.** Article text has since changed; do not cross-check
  a gold sentence against live Wikipedia and conclude the parse is broken.
* **Page titles are the `doc_id`,** including FEVER's `-LRB-`/`-RRB-` escaping for
  parentheses (`Soul_Food_-LRB-film-RRB-`). They are displayed as-is, but they are
  **NFC-normalised on the way in** — see ADR-023. The dev file and the wiki dump ship
  different normalisations of the same title, and exact matching drops 72.7% of
  non-ASCII gold pages while looking like a 1.1% loss overall.
* **`--n-docs` counts pages, not sentences.** 4,000 pages is roughly 100–150 k
  sentences, so the dense index is ~150–230 MB and takes a couple of minutes to build
  on the 4050. Sentence count, not page count, is what governs index cost.
* **One pass, but a slow one.** ~5.4 M `json.loads` calls; several minutes. It is a
  one-time build, cached by the corpus fingerprint in the index filename.
* **Two environment traps, both found by running rather than by reading.** (a) The zip
  ships 109 macOS AppleDouble forks (`__MACOSX/wiki-pages/._wiki-NNN.jsonl`) alongside
  the 109 data files; they end in `.jsonl`, they are binary, and `._` sorts *first*, so
  a naive `endswith(".jsonl")` filter reads one before any real file and dies with
  `JSONDecodeError` at char 0. `is_real_wiki_member()` handles it. (b) The Windows
  console is cp1252 and Wikipedia titles are not — the first build completed the entire
  5.4 M-page scan and then died inside a `print()` with
  `UnicodeEncodeError: 'charmap' codec can't encode character '́'`.
  `src.core.config.configure_console()` now fixes this for every entry point.
* **A duplicate page id in the dump would abort the build.** Gold and other-gold pages
  are collected into dicts (so duplicates collapse), but the random reservoir is a
  list, so a page id appearing twice in the dump reaches `Corpus` twice and raises
  `duplicate doc_id`. That is a loud failure rather than a silent one, which is the
  right side to err on, but it is not handled gracefully.
* **`--n-claims` is split as evenly as the label count allows.** 200 over 3 labels
  gives 67/67/66, with the remainder going to the first labels in sorted order — fixed,
  not random, so the sample stays reproducible.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Fetch current Wikipedia by title (~30 MB) instead of the 1.7 GB dump** | Current article text has different sentence boundaries, so gold `(page, sent_id)` indices would have to be re-derived by string matching. That is lossy, and it corrupts `is_gold` — the one signal the entire oracle comparison rests on — without any error. Rejected outright; the download is the cheaper cost. |
| **Extract the zip to disk** | ~4–5 GB extracted on a volume at 92% capacity, for data we read once. Streaming members keeps peak disk at the zip. |
| **Filter empty sentences out of the corpus** | The bug this module exists to prevent. Empty sentences hold an index; removing them shifts every later gold key silently. They cost a few MB and are never retrieved (zero BM25 score). |
| **Enumerate positions rather than read declared indices** | Same failure, arrived at differently — and it additionally breaks on out-of-order and unindexed records, both of which occur in the dump. |
| **Uniform-random distractors only** | Yields mostly short stubs, so the gold page is often the only on-topic document and retrieval becomes trivial. Seeding the pool with other claims' gold pages costs nothing and gives substantive articles. |
| **Hard negatives (topically near the gold pages)** | The honest way to make retrieval non-trivial — but selecting them needs a retriever, which makes the corpus a function of the thing being measured. That circularity is worse than an easy corpus that is *labelled* easy. |
| **All 2,892 dev gold pages + their claims** | Tempting (it is already "a few thousand"), but it makes every page in the corpus gold for something, which is a stranger distribution than the mixed pool and inflates the chance that a distractor genuinely settles a claim. |
| **`datasets.load_dataset("fever")`** | Convenient for claims, but the wiki config is the full ~5.4 M pages with no way to take a subset without downloading all of it, and it gives no control over `lines` parsing — which is the only part that matters here. |
