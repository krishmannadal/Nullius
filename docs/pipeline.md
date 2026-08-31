# `src/pipeline.py`, `src/core/trace_io.py`, `src/cli.py`

The runner, the record, and the command line.

## 1. Problem

Three things, in order of how much they matter.

**1. Run the five stages without becoming a sixth component.** The runner must make no
decisions. Every decision lives in a component; anything the runner decides is a
decision that cannot be swapped from YAML, cannot be named in a trace, and cannot be
ablated.

**2. Make the oracle comparison trustworthy.** Replacing retrieved evidence with gold
evidence and re-running is the central measurement of the project — the difference is
the *retrieval-attributable error*. It is only trustworthy if **nothing else differs**
between the two runs, and if the two can never be confused afterwards.

**3. Produce a record that survives the process.** A run directory has to be readable
six weeks later, on a machine with no model stack, and must not be mistakable for an
evaluation result.

## 2. Formal I/O

```
analyze(pipe, response, ctx, *, example=None, corpus=None,
        mode="retrieved"|"oracle", retrieve_k=None, check_contracts=True) -> Trace

analyze_example(pipe, example, corpus, ctx, ...) -> (Trace, Optional[Trace])
        # (retrieved, oracle) — oracle is None when the example has no gold evidence
```

Run directory:

```
results/<run_id>/
    traces.jsonl      one Trace per line, mode="retrieved"
    oracle.jsonl      one Trace per line, mode="oracle"
    config.json       the RESOLVED config, after --set overrides
    provenance.json   git sha + dirty flag + python/torch/cuda/transformers versions
    metrics.json      counts, timings, oracle gap — with a _warning inside it
```

Failure cases live elsewhere, **one file each**:

```
results/failure_cases/<UTC-stamp>-<claim_id>.json
    { note, claim_id, trace, oracle_trace, git_sha, config_hash, saved_at }
```

`timings` is milliseconds per stage, summed over all claims in the response:
`extract_ms`, `retrieve_ms`, `rerank_ms`, `verify_ms`, `aggregate_ms`, `total_ms`.

**Measured on 20 FEVER examples** (hybrid retrieval, no rerank, DeBERTa, k=5, RTX 4050):

| stage | total ms | share |
|---|---|---|
| retrieve | 4377 | 44% |
| verify | 5568 | 55% |
| extract | 94 | 0.9% |
| rerank (noop) | 2.3 | 0.02% |
| aggregate | 2.0 | 0.02% |

≈ 450 ms per example end to end. Verification and retrieval are the entire cost;
aggregation — the slot most likely to become the research question — is free.

## 3. Algorithm

```
analyze(pipe, response, ctx, mode):
    claims ← extractor.extract(response)                → check_claims
    for claim in claims:
        if mode == "oracle":  candidates ← gold_evidence_for(example, corpus)
        else:                 candidates ← retriever.retrieve(claim, retrieve_k)
                                                        → check_evidence_list
        reranked ← reranker.rerank(claim, candidates, k) → check_rerank_is_subset
                                                        → check_evidence_list
        reranked ← mark_gold(reranked, example.gold_evidence_ids)
        pairs    ← [verifier.score(claim, e) for e in reranked]   → check_pair_verdict
        verdict  ← aggregator.aggregate(claim, pairs)
    return Trace(..., mode=mode, timings=per-stage totals)
```

## 4. Code walkthrough — the lines that matter

**The oracle substitutes evidence and only evidence (`pipeline.py:183`).**
```python
if mode == "oracle":
    candidates = gold_evidence_for(example, corpus)
else:
    candidates = pipe.retriever.retrieve(claim, pool)
```
One branch, at one point, inside an otherwise identical loop. The same reranker,
verifier and aggregator run on both. `test_oracle_and_retrieved_differ_only_in_evidence`
asserts that claims, `config_hash` and component descriptions match between the two
traces and only `mode` differs — because a gap between conditions means nothing if the
conditions differ in some other way too.

**A missing gold key aborts rather than degrading (`pipeline.py:88`).**
```python
if not corpus.has(doc_id, sent_id):
    raise KeyError(f"gold evidence ... is not in this corpus. The oracle condition "
                   "would be silently incomplete; refusing.")
```
A partially-materialised oracle biases the central measurement in an invisible
direction — it makes retrieval look *better* by handicapping the oracle. Loud failure is
the only safe behaviour. (The corpus builder already guarantees this, so it is a
belt-and-braces check for hand-assembled examples.)

**Oracle ranks are flagged as meaningless (`pipeline.py:105`).**
```python
retriever_meta={"source": "gold_annotation", "rank_is_annotation_order": True},
```
Gold evidence has no retrieval ranking, so `rank` is just the order the annotator listed
it in. `WeightedByRetrievalAggregator` will happily weight by it — that is weighting by
annotation order, which is nothing. The flag means the trace says so rather than the
number quietly lying.

**`analyze_example` returns `None`, not an empty oracle trace (`pipeline.py:264`).**
```python
if not example.has_gold_evidence:
    return retrieved, None
```
"No oracle exists for this example" and "the oracle ran and found nothing" are different
facts. FEVER's NEI class has no gold evidence by construction, so a third of any FEVER
sample hits this. Returning an empty trace would put those into the gap computation as
oracle failures.

**Contract checks are on by default (`pipeline.py:170` onwards).** Four checks per
claim, O(k) set arithmetic against a ~70 ms forward pass — unmeasurable. They pay for
themselves the first time a component misbehaves:
`test_a_retriever_violating_the_rank_contract_is_caught`,
`test_a_reranker_inventing_evidence_is_caught`,
`test_a_verifier_answering_about_the_wrong_evidence_is_caught`.

And there is a second line of defence:
`test_checks_can_be_disabled_and_then_the_trace_catches_it_anyway` runs with
`--no-checks` and shows `Trace.__post_init__` still refusing a verdict that scores
evidence the claim never saw.

**Traces are flushed per write (`trace_io.py:59`).**
```python
self._fh.write(trace.to_json_line() + "\n")
self._fh.flush()
```
A 200-example run that dies at 180 should not cost the first 179.
`test_traces_are_flushed_per_write_so_a_crash_keeps_them` reads the file back *before*
`close()`.

**`read_traces` refuses a malformed line (`trace_io.py:84`).** Skipping it silently
would change every count computed from the file, and nothing would say so.

**The environment is recorded, because the config hash cannot cover it
(`trace_io.py:105`).** Two runs with the same `config_hash` and different
`transformers` versions are different experiments and the config would not say so.
`provenance.json` closes that gap with the versions, the CUDA device, and the git sha.

**The warning travels inside the artifact (`trace_io.py:150`).**
```python
payload["_warning"] = (f"harness_kind={harness_kind}. Every corpus in this repo contains "
                       "the gold evidence for its own examples by construction ... "
                       "Do not report them.")
```
A `metrics.json` gets copied out of `results/` and opened months later. A README warning
does not travel with it; a key inside the file does.

**Failure cases embed the whole trace (`trace_io.py:206`).** Six weeks from now the
question is "why did it decide that", and only `aggregation_trace` answers it. One file
per case rather than an appended log, because these get read, edited and moved by hand
and a corrupted append would take the whole corpus with it.

**The CLI contains no pipeline logic (`cli.py`).** It parses arguments, builds through
the registry, calls `analyze`, prints. Anything it did that the API could not would be a
divergence between what you debug here and what the frontend runs.

## 5. Data flow

```
configs/*.yaml ──load_config──► apply_overrides(--set) ──► resolved cfg
                                          │
                    ┌─────────────────────┼──────────────────────┐
                    ▼                     ▼                      ▼
            build_pipeline(cfg)    config_hash(cfg)      set_global_seed(cfg.seed)
                    │                     │
                    ▼                     ▼
              Pipeline ──────────► RunContext(run_id, config_hash, git_sha)
                    │
     for each Example:
        analyze_example ──► (retrieved Trace, oracle Trace|None)
                    │              │
                    ▼              ▼
            traces.jsonl     oracle.jsonl
                    └──────┬───────┘
                           ▼
                   summarise_run ──► metrics.json  (+ config.json, provenance.json)

Trace.from_json_line ◄── src.cli show / the Streamlit renderer — NO model stack needed
```

## 6. How to verify it

```powershell
python -m pytest tests/test_pipeline.py -v      # 29 passed, no models, ~12 s
python -m src.cli list-components
python -m src.cli run --config configs/mini.yaml --limit 14
python -m src.cli analyze --config configs/mini.yaml --text "Marie Curie was born in Paris."
```

The offline-renderer property is checkable directly — run this with an interpreter that
has **no** torch, transformers, faiss or spacy installed:

```powershell
python -m src.cli show results/<run_id>/traces.jsonl --limit 2
```
It renders. That is the property the Streamlit trace viewer depends on.

| Test | Failure signature |
|---|---|
| `test_oracle_and_retrieved_differ_only_in_evidence` | the oracle gap measures something other than retrieval |
| `test_oracle_refuses_a_gold_key_missing_from_the_corpus` | a handicapped oracle makes retrieval look better than it is |
| `test_analyze_example_skips_the_oracle_when_there_is_no_gold` | NEI examples enter the gap computation as oracle failures |
| `test_checks_can_be_disabled_and_then_the_trace_catches_it_anyway` | a bad component gets all the way into a saved trace |
| `test_metrics_carry_the_warning_inside_the_artifact` | a metrics file gets read as a result |
| `test_summarise_excludes_examples_without_gold_from_recall` | NEI examples contribute 0.0 or 1.0 to a recall mean |

## 7. Limitations and failure modes

* **Gold evidence is annotated per response, not per claim.** In oracle mode every claim
  in a response receives the *same* gold set. For FEVER (one claim per response) that is
  exact; for a multi-claim response it is an over-approximation that makes the oracle
  more generous than it should be. Nothing currently produces multi-claim FEVER
  responses, but the frontend's free-text box will.
* **The oracle can be *worse* than retrieval, and it was, immediately.** On
  `fever-dev-23797` the retrieved run is correct and the oracle run abstains. The cause
  is not retrieval: it is that gold sentences are often pronoun-initial (see the finding
  below), so the oracle hands the verifier a sentence it cannot resolve while retrieval
  happened to surface a self-contained one. **A retrieval-attribution number computed
  without noticing this is measuring a preprocessing artifact.**
* **Pronoun-initial evidence collapses the NLI signal.** Measured directly:

  | premise | p_entail |
  |---|---|
  | `"She was born in Warsaw, …"` (as stored in FEVER) | **0.001** |
  | `"Marie Curie . She was born in Warsaw, …"` | **0.997** |

  **14.7% of gold sentences in the FEVER debug corpus start with a pronoun or deictic.**
  The original FEVER baseline prepends the page title for exactly this reason. This
  repo does **not**, because changing it is a modelling decision and out of scope for a
  placeholder build — it is a backlog item (hypothesis, null, and a one-line config
  option), not a silent default change.
* **Everything is sequential.** One response at a time, one claim at a time, one pair at
  a time. 20 FEVER examples take ~9 s in both conditions. 200 would take ~90 s; the full
  dev set would take ~2.5 hours. Fine for a harness, wrong for an evaluation.
* **Components are rebuilt per process, not cached across runs.** The dense index loads
  in ~1 s and DeBERTa in ~2 s from warm cache, so a `run` invocation pays ~5 s of
  startup. Streamlit will need an instance cache (OQ-009).
* **`summarise_run` assumes one claim per trace when computing the oracle gap** — it
  compares `verdicts[0]`. Correct for FEVER, wrong for a multi-claim response, where it
  silently reports the first claim's outcome as the example's.
* **No per-class breakdown in `metrics.json`.** Aggregate agreement hides which class is
  failing, and NEI is the class most likely to be. Deliberate for now — per-class
  reporting is an evaluation activity, not a harness one — but it is the first thing to
  add when evaluation starts.
* **`--no-checks` exists and should not be used.** It is there to measure the cost of the
  checks (it is negligible), not to speed up a real run.

## 8. Rejected alternatives

| Alternative | Why not |
|---|---|
| **Batching all claims through the verifier at once** | Would be several times faster, and would break ADR-002: the verifier's signature takes one pair so that pooling cannot hide in it. Batching belongs inside a future `score_batch` that still returns per-pair verdicts. |
| **Running the oracle as a separate CLI command** | Two commands means two configs, two run ids, and eventually two different code paths. `analyze_example` returning both traces from one context guarantees the comparison is within-run. |
| **An empty oracle trace for NEI examples** | Would put "no oracle exists" and "the oracle found nothing" into the same bucket, which is precisely the distinction the project is about. |
| **One appended `failure_cases.jsonl`** | Cheaper to write, but these files get edited and moved by hand over weeks, and one corrupted append takes the corpus with it. One file per case is boring and robust. |
| **Storing only a summary in a failure case** | The question later is *why*, and only `aggregation_trace` answers it. Traces are a few tens of KB. |
| **Computing metrics lazily from the trace files** | Attractive (the trace is the source of truth) and still possible — `read_traces` exists precisely so `metrics.json` can be regenerated. It is written eagerly so a run directory is complete without a second step. |
| **Contract checks only under a debug flag** | They cost microseconds against a 70 ms forward pass, and the bugs they catch are silent. A check that is off by default is a check that is off. |
| **Making the CLI the only entry point and having the API shell out to it** | Would guarantee CLI/API parity by construction, but at the cost of process spawning per request and no streaming. Both call `src.pipeline` directly instead. |
