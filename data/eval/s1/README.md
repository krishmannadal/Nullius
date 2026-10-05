# S1 curated extraction pilot — protocol v2

S1 compares sentence extraction with model-based atomic claim extraction on 30
frozen texts. It does not evaluate hallucination detection, retrieval, verification,
aggregation, confidence calibration, or real-world generalization. The dataset is
**curated synthetic / provenance-unverified**. Source labels do not establish
15 genuine LLM generations, 10 independently authored human texts, or authorship
subgroup effects. Length labels overlap. Preserve the texts for adjudication.

The authoritative state and executed checks are in [S1_READINESS](../../../docs/S1_READINESS.md).
Historical v1 annotations are drafts; formal v2 submissions require the frozen
[guidelines](ANNOTATION_GUIDELINES.md), disjoint character spans, explicit operations
and compound IDs. [Calibration examples](calibration_examples.json) are unrelated
teaching strings, not benchmark answers. [EVALUATION_PROTOCOL](EVALUATION_PROTOCOL.md)
fixes estimands, error codes, matching and bootstrap conventions before outputs exist.

## Identity and independence

The canonical SHA-256 is
`8ef976d32392724acb468821a0314787b7e3c6c563e3556381e6b1a5e1672356`.
Only CRLF becomes LF in memory; mixed LF/CRLF is accepted, lone CR is rejected.
Raw transport bytes are hashed separately. Never edit responses.jsonl to obtain
an easier annotation or result. A configuration hash is not a complete trace hash.

Two humans independently annotate all 30 responses, including explicit zero-claim
completion. They must not see each other's work or any system predictions. The
annotation UI performs no extraction and requires an independence attestation.
Attestations and file hashes are audit controls, not proof of a person's conduct.
Save and retain each original export before adjudication.

## Operator sequence

Run from the repository root with the pinned environment. Commands that write
scientific artifacts refuse existing destinations. Use a new directory for a new
run and retain superseded decisions; do not edit a frozen run in place.

```powershell
python -m src.eval.s1_workflow validate-benchmark
python -m streamlit run src/ui/s1_annotation.py
```

Each human selects identity A or B, completes every response, and downloads the
ZIP. Store its two files as `annotations_A.jsonl` / `review_A.json` or the B
counterparts under this directory. Renaming the files does not change their bytes.
Never fabricate these submissions from the teaching examples or tests.

```powershell
python -m scripts.validate_s1_annotations data/eval/s1/annotations_A.jsonl --annotator-id A --review data/eval/s1/review_A.json --formal --require-complete
python -m scripts.validate_s1_annotations data/eval/s1/annotations_B.jsonl --annotator-id B --review data/eval/s1/review_B.json --formal --require-complete
python -m src.eval.s1_workflow prepare-adjudication
```

This creates only `agreement.draft.json` and `adjudication.draft.json`. Humans
review all A/B claims and candidates, enter one-to-one `human_matches` containing
`claim_a_id` / `claim_b_id`, and explicitly list `unmatched_a` / `unmatched_b`.
Mark every response reviewed. Save finalized decisions as `agreement.json`.
Count agreement is descriptive; lexical suggestions are not semantic agreement.
Claim-presence kappa remains undefined without a human-defined shared unit universe.

The adjudicator writes `gold_annotations.jsonl` under the v2 claim schema and
`adjudication.json`, preserving draft fields. Record identity, no-output attestation,
raw SHA-256 of finalized agreement and gold, response-level notes, exact gold claim
IDs, zero-claim decisions, and `compound_groups` (`group_id`, `gold_claim_ids`).
Every decomposed gold claim must belong to exactly one group of at least two claims.
Use `Get-FileHash -Algorithm SHA256` to obtain raw file hashes (lowercase in JSON).
No tool generates the human gold for you.

```powershell
python -m src.eval.s1_workflow freeze-gold
python -m src.eval.s1_workflow validate-gold
```

Only after this succeeds, configure `NULLIUS_LLM_API_KEY` and
`NULLIUS_LLM_BASE_URL` (a chat-completions endpoint root ending in `/v1`) in the
process environment. Choose and document the actual provider and model; the
placeholders below are deliberately not defaults. No credentials are recorded.

```powershell
python -m scripts.generate_s1_llm_cache --provider PROVIDER --model MODEL --seed 1337
python -m src.eval.s1_predictions
python -m src.eval.extraction_eval --prepare-matching
```

Generation preserves full raw provider JSON, prompts, settings and timestamps;
parsing failure leaves raw evidence and no complete manifest. There is no automatic
retry. A descriptive requested revision is not a provider-enforced revision;
resolved model identity is recorded only when returned. The debug `llm_cached`
adapter and hand-written debug cache are not admissible S1 runs. Prediction freeze
preserves duplicates, empty outputs, missing/approximate spans, stable IDs, raw
indices and model/source identities for both systems.

Humans edit a copy of `predictions/matching.draft.json`, preserving every candidate
and immutable lexical field. Save as `matching.final.json`. Each candidate needs
`human_decision` (accept/reject/reassign), `final_match`, and rationale for accepted
or reassigned pairs. Matching is one-to-one. Include complete unmatched ID lists,
multilabel error judgments (M/S for unmatched gold/predictions), reviewer identity,
time, completion, and explicit compound relations with split judgments. Below-0.8
pairs remain available for human reassignment; suggestions never finalize decisions.

```powershell
python -m src.eval.extraction_eval --freeze-matching
python -m src.eval.extraction_eval --evaluate --seed 1337 --n-resamples 10000
```

These create a matching freeze receipt and `predictions/report.json` / `report.md`.
The JSON contains response-level metrics, pooled micro and response macro metrics,
matched-pair span coverage, compound measures, taxonomy counts, paired differences
and response bootstrap intervals. Undefined values are null, not invented zeros.
A later matching revision must name and hash the retained prior decision file.
All real-data stages are currently blocked until actual human submissions exist.

## Verification and boundaries

Tests use temporary synthetic fixtures and mocked provider responses only. They
are engineering checks, never benchmark observations. Install Playwright Chromium
for browser fixtures. CI installs the pinned dependencies and runs Ruff and pytest.
Keep the API local to trusted operators: it is not an authenticated public service.
The extension performs explicit capture; browser fixtures do not establish ongoing
compatibility with every live ChatGPT interface.

```powershell
python -m ruff check src tests scripts
python -m pytest
python -m scripts.package_extension
python -m scripts.package_extension --check
```

The package is built from `src/extension` with deterministic file ordering, LF text,
fixed ZIP timestamps and no compression variability. Its checksum is recorded in
`nullius-extension.sha256` and `docs/extension-package.json`. A differing previous
ZIP is preserved beside it. Load the source directory as an unpacked extension;
set `NULLIUS_EXTENSION_IDS` to its explicit ID before starting the local API.
