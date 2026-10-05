# S1 research readiness — 2026-10-05

**Formal S1 annotation: READY for the curated pilot under protocol v2.** This means
an operational, validated collection workflow, not completed human annotation.
**Scientific S1 evaluation: NOT READY.** Actual independent human submissions,
agreement, adjudication, frozen gold, real provider outputs and human matching
are absent. The gates reject execution instead of producing substitute evidence.
Practice is ready using separate teaching examples. Engineering implementation and
fixture validation do not establish scientific validity on real benchmark data.

| Requirement | Status | Evidence | Remaining limitation |
|---|---|---|---|
| Benchmark integrity | VERIFIED | 30 IDs validated; original response bytes unchanged | Curated sample only |
| Canonical hash | VERIFIED | LF, CRLF, mixed-EOL and lone-CR regression checks; benchmark CLI passes | Raw transport identity differs across EOL checkouts |
| Source provenance | UNVERIFIED; scoped | metadata.json and report limitation declare curated synthetic/provenance-unverified | No original generation or human authorship records; no source-effect claims |
| Annotation semantics | READY | Frozen v2 guidelines preserve qualifiers, attribution, polarity, conditions and disjunction | Human interpretation still requires trained annotators |
| Span policy | READY | Ordered disjoint trimmed Unicode-character regions; source_span is first region | Rewrites require human judgment |
| Guideline hash | VERIFIED | guidelines.sha256 and metadata content hash; stale-content gate tests | Any revision requires a reviewed new protocol |
| A/B independence | READY workflow | Session isolation, raw-only UI, independent attestation and formal review validation | Researcher attestations are not independent proof of conduct |
| Annotation completion | NOT COMPLETE | 30/30 completion and zero-claim checks; missing submissions gate rejects | Actual A/B work required |
| Agreement | IMPLEMENTED / FIXTURE-VALIDATED | Complete human pair and unmatched coverage; descriptive count agreement | Actual human agreement absent; claim-presence kappa requires a shared coded unit universe |
| Adjudication | IMPLEMENTED / FIXTURE-VALIDATED | Every response needs reviewed gold IDs, zero decision, notes and explicit groups | Actual human adjudicator required |
| Gold freeze | CLOSED | Linked hashes, human attestations, complete response reviews; tamper and incomplete cases rejected | No real gold file or freeze manifest created |
| Prediction freeze | IMPLEMENTED / FIXTURE-VALIDATED | Stable IDs, raw/processed indices, duplicates and source artifacts retained | Actual outputs not generated |
| LLM cache provenance | IMPLEMENTED / MOCK-TESTED | Real chat-completions client; raw JSON, prompts, settings, hashes and timestamps; fail before network without gold | Provider/model selection and credentials required; provider execution not exercised |
| spaCy reproducibility | IMPLEMENTED | Package and model bytes hashes, effective settings, raw outputs; cached-model extraction tests passed | No frozen S1 baseline yet; model artifacts must be retained |
| Human matching | IMPLEMENTED / FIXTURE-VALIDATED | All pair candidates retained; accept/reject/reassign; one-to-one and unmatched coverage; immutable receipt | Actual system-to-gold judgments required |
| Claim metrics | FIXTURE-VALIDATED | Numeric TP/FP/FN, micro/macro and empty-set conventions | No real S1 estimates |
| Span metrics | FIXTURE-VALIDATED | Exact/region IoU, missing-span zero scores, undefined boundaries and coverage | Conditional on human-accepted matches |
| Decomposition metrics | FIXTURE-VALIDATED | Explicit compound relations, same-group precision/recall, split ratio and judgments | Cross-group cases need documented human choices |
| Error taxonomy | FIXTURE-VALIDATED | Multilabel M/S/U/O/R/P/B/C/D/N with rationale and unmatched coverage | Actual error annotations absent |
| Paired bootstrap | FIXTURE-VALIDATED | Seeded whole-response resampling; macro and pooled micro intervals and paired differences | Small curated sample; undefined metrics use reported complete cases |
| Test safety | VERIFIED | Temporary research fixtures; service path injection; unchanged-file checks; audit hook blocks real research writes | Hook protects this Python process, not arbitrary external software |
| Service and reaggregation | REGRESSION-TESTED | Content-keyed caches, saved pairwise snapshots, effective aggregation config and original run identity | In-memory runs expire; trusted in-process plugins are not sandboxed |
| Model and trace identity | IMPLEMENTED / REGRESSION-TESTED | Revisions, model/data identities, stage contracts, complete traces and working-source hashes | Unknown upstream weight revisions remain unknown; config hash alone is not trace identity |
| Local HTTP boundary | REGRESSION-TESTED | Per-app service, explicit extension origins, repository config restriction and bounded overrides | Local trusted-operator service, no authentication or public deployment claim |
| Extension capture | FIXTURE-VALIDATED | Nested speaker markers, controls, code/table formatting, hidden/empty/streaming states, injection idempotence and popup refresh/timeout tests | No authenticated live ChatGPT session test; popup closure does not guarantee server cancellation |
| Extension package | VERIFIED | Deterministic build and source comparison; checksum in extension-package.json | Browser packaging is not store publication |
| Full test suite | PASSED: 437/437 | Final JUnit record; 0 failures, 0 skips | Test success is engineering evidence, not research results |

## Executed verification

Final full suite: **437 passed, 0 failed, 0 skipped** (2 warnings), 76.49 seconds.
Executed with `python -m pytest -o addopts='' -q --disable-warnings --maxfail=6
--junitxml=.test-artifacts/full-final.xml` and offline cached model weights.
The full run includes all extension, S1, numeric evaluator, API, UI and model tests.

Other executed checks:
- Ruff checks of src, tests and scripts passed; new workflow/test modules formatted.
- Focused S1/UI rerun: 77 passed, 0 failed, 0 skipped.
- Focused extension/hardening rerun: 40 passed, 0 failed, 0 skipped.
- Earlier full run: 425 passed, 3 failed, 0 skipped. All three failures exposed
  repeated UI comparison requests after removing a global cache. Fixed with a
  bounded session-local cache, then reran verification; failures were not suppressed.
- Subsequent full run: 436 passed, 1 failed, 0 skipped. The remaining assertion
  expected the obsolete provider-not-implemented message; it now checks the actual
  gated S1 workflow guidance. The final full rerun passed after that correction.
- Benchmark validation passed. prepare-adjudication, validate-gold and evaluation
  refused the actual incomplete research directory with expected exit code 1.
- Package generation and --check returned the same SHA-256:
  `942891d1df4e6adf5c87be983cc883f752b42c2b44e7c10bb16ef83a82b11091`.
- Canonical benchmark SHA-256:
  `8ef976d32392724acb468821a0314787b7e3c6c563e3556381e6b1a5e1672356`.
- Original/current raw benchmark SHA-256:
  `cd875b851262f2e8333733ba38aee1554b38c513397f6dc720ebea4ebef6ef21`.

The final run uses the existing Windows Python environment and cached model weights
with Hugging Face offline mode enabled. A fresh Linux CI run and clean dependency
installation were not executed in this session.

Local verification outputs are `.test-artifacts/full-final.xml` and
`.test-artifacts/gates.json`. They are engineering artifacts outside the real
research directory. No actual human gold, real provider predictions or scientific
metric report was produced. Synthetic fixture decisions remain in temporary test
directories and are not promoted to S1 evidence.

## Remaining external prerequisites

Two independent trained humans must annotate all responses, a human must finalize
agreement/adjudication, and the gold freeze must validate. Then configure an actual
provider/model and credentials, preserve its real outputs and freeze both systems.
Humans must finalize system matching before metrics are allowed. Retain the frozen
artifacts and source environment. The [runbook](../data/eval/s1/README.md) contains
exact commands and artifact fields. [Engineering inventory](ENGINEERING_CHANGES.md)
lists every changed file and delivered package.

The unresolved provenance, missing human work, provider weight identity and live
browser compatibility were not filled in with invented evidence. NLI softmax and
aggregation scores remain uncalibrated. No S1 result can establish retrieval,
verifier, aggregation, or hallucination-detection performance.
