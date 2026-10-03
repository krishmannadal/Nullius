# Nullius S1 Scientific Evaluation Benchmark

This directory contains the dataset, guidelines, templates, and metadata for the **S1 Extraction Quality Scientific Evaluation**.

## 1. Scientific Objective

The S1 evaluation benchmark measures how accurately claim extractors (`SpacySentenceExtractor` and `LLMClaimExtractor`) extract atomic factual claims from raw model responses and locate their source spans.

To maintain strict scientific validity and prevent data snooping, the benchmark enforces a four-stage protocol:

```text
               responses.jsonl
               /             \
   [Independent]             [Independent]
         ↓                         ↓
    Annotator A               Annotator B
         ↓                         ↓
 annotations_A.jsonl        annotations_B.jsonl
               \             /
                \           /
            Inter-Annotator Agreement
                        ↓
                Human Adjudication
                        ↓
              gold_annotations.jsonl
                        ↓
             System Extractor Evaluation
```

---

## 2. Directory Contents

- `responses.jsonl`: 30 frozen responses (10 simple-factual, 12 compound-factual, 8 mixed). Contains 15 LLM-generated, 10 human-written, and 5 adversarial texts across short, medium, and long lengths.
- `ANNOTATION_GUIDELINES.md`: Operational rulebook for human annotators detailing claim criteria, linguistic edge cases, and protocol ambiguities.
- `annotations_A.template.jsonl`: Empty template file for Annotator A.
- `annotations_B.template.jsonl`: Empty template file for Annotator B.
- `metadata.json`: Dataset version, timestamps, and cryptographic hashes for reproducibility.

---

## 3. Strict Annotation Independence Protocol

The human annotation process must adhere to the following non-negotiable scientific constraints:

1. **Raw Text Only:** Annotators must read only `responses.jsonl` and `ANNOTATION_GUIDELINES.md`.
2. **No Extractor Hints:** Annotators must **never** be shown system extractor predictions, LLM outputs, pre-split sentence boundaries, or ranked candidate claims.
3. **Double-Blind Independence:**
   - **Annotator A** completes `annotations_A.jsonl` without seeing Annotator B's annotations, extractor outputs, or gold data.
   - **Annotator B** completes `annotations_B.jsonl` without seeing Annotator A's annotations, extractor outputs, or gold data.
4. **No Automated Adjudication:** Inter-annotator differences must be adjudicated by human consensus or a designated human adjudicator. No simulated or algorithmic adjudication is permitted.
5. **No Metric Calculation Before Adjudication:** Extractor precision, recall, F1, and span IoU may only be computed after `gold_annotations.jsonl` is adjudicated and frozen.

---

## 4. Annotation Record Schema

Each line in `annotations_A.jsonl` and `annotations_B.jsonl` must be a valid JSON object matching this schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `response_id` | `string` | Must match a valid ID in `responses.jsonl` (e.g. `"s1-resp-01"`). |
| `claim_id` | `string` | Unique identifier within the file (e.g. `"s1-A-001"`). |
| `claim_text` | `string` | Standalone, self-contained atomic factual assertion. |
| `source_span.start` | `integer` | 0-indexed character start position in `response["text"]` (`>= 0`). |
| `source_span.end` | `integer` | 0-indexed exclusive character end position (`> start`, `<= len(text)`). |
| `decomposed` | `boolean` | `true` if rewritten/decomposed from span; `false` if exact substring. |
| `hedged` | `boolean` | `true` if source text hedged or attributed the statement; `false` otherwise. |
| `annotator_id` | `string` | `"A"` or `"B"` indicating the annotator. |
| `notes` | `string` | (Optional) Annotator notes on edge cases or ambiguity. |

---

## 5. Step-by-Step Annotator Instructions

### For Annotator A:
1. Copy `annotations_A.template.jsonl` to `annotations_A.jsonl` (or start a new file).
2. Open `responses.jsonl` and read `ANNOTATION_GUIDELINES.md`.
3. For each response, identify atomic factual assertions, calculate source spans `[start, end)`, and record each claim.
4. Validate mechanical compliance:
   ```bash
   python scripts/validate_s1_annotations.py data/eval/s1/annotations_A.jsonl --annotator-id A --show-spans
   ```
5. Ensure 0 errors before locking `annotations_A.jsonl`.

### For Annotator B:
1. Copy `annotations_B.template.jsonl` to `annotations_B.jsonl` (or start a new file).
2. Independently open `responses.jsonl` and read `ANNOTATION_GUIDELINES.md`.
3. Annotate all 30 responses in complete isolation from Annotator A.
4. Validate mechanical compliance:
   ```bash
   python scripts/validate_s1_annotations.py data/eval/s1/annotations_B.jsonl --annotator-id B --show-spans
   ```
5. Ensure 0 errors before locking `annotations_B.jsonl`.

---

## 6. Post-Annotation Phase (Agreement & Adjudication)

Only after both `annotations_A.jsonl` and `annotations_B.jsonl` are submitted and validated:
1. Calculate inter-annotator agreement metrics (span IoU, Token F1, agreement on atomic decomposition count).
2. Perform human adjudication of discrepant claims to produce `gold_annotations.jsonl`.
3. Update `metadata.json` with the cryptographic hash of `gold_annotations.jsonl`.
4. Run downstream extractor evaluation scripts against `gold_annotations.jsonl`.


## 7. Independent annotation UI

From the repository root, with the virtual environment active:

```bash
python -m streamlit run src/ui/s1_annotation.py --server.headless true --server.port 8502
```

The page loads the frozen benchmark and guidelines. Enter your own annotator ID
and guideline version. Specify source offsets, check the displayed source slice,
and enter each factual assertion yourself. The tool supplies no extractor hints.
Mark each response reviewed, including any response with zero claims. Adding or
removing a claim reopens that response for review.

Download your ZIP regularly: session state is volatile. It contains
`annotations.jsonl` and `review.json`. Resume by uploading both files into a fresh
session using the same identity, guidelines and benchmark. Resume verifies the
annotation and response hashes. The app cannot authenticate an annotator or prove
independence: keep A and B's files and sessions separate yourself. It never writes
an annotation, gold file, or model cache into the benchmark directory.

After extracting each ZIP into separate local directories:

```bash
python -m scripts.validate_s1_annotations /path/to/A/annotations.jsonl \
  --annotator-id A --review /path/to/A/review.json --require-complete --show-spans
```

`--require-complete` needs the review record. Merely having claims for a response
cannot distinguish a reviewed zero-claim response from an unfinished response.
Validation checks structure and completion records, not human claim correctness.
The existing positional validator command remains supported.

## 8. Agreement review after both annotators finish

Only after independent annotation is complete, an adjudicator can run:

```bash
python -m src.eval.agreement \
  --responses data/eval/s1/responses.jsonl \
  --expected-sha256 8ef976d32392724acb468821a0314787b7e3c6c563e3556381e6b1a5e1672356 \
  --annotations-a /path/to/A/annotations.jsonl --review-a /path/to/A/review.json \
  --annotations-b /path/to/B/annotations.jsonl --review-b /path/to/B/review.json \
  --output /path/to/new-agreement-review.json
```

The report preserves all annotations and source text. Normalized exact matching
and token F1 >= 0.8 propose candidate pairs; maximum-weight bipartite matching
suggests one-to-one pairs. Every human decision remains blank. Unmatched claims
are visible and must also be reviewed; lexical similarity can hide opposite
polarity. The report never resolves disagreement or creates gold.

Count agreement uses Krippendorff interval alpha with responses as units and
atomic annotation counts as the measurement, including reviewed zero-claim
responses. No variation yields undefined alpha (`null`), not perfect agreement.
Equal counts are not semantic claim agreement. Claim-presence Cohen's kappa is
not calculated: a shared claim universe and explicit absence decisions still
need a defined, human-coded protocol. Do not infer that the S1 agreement gates
passed from lexical candidate counts or count alpha alone.

## 9. Integrity and remaining research gates

The original handoff SHA-256 (`cd875b...`) covers CRLF file bytes. Git's existing
`.gitattributes` stores this JSONL with LF endings; its byte hash is `8ef976d...`.
Metadata now records both hashes and the reason. Response strings and IDs were
not changed. Hashes always cover exact bytes; tools do not silently normalize
uploaded files.

All 30 records have `provenance: synthetic-s1`. Their source-type labels are not
proof of actual human authorship or LLM generation. Establish original sources
before making source-subgroup scientific claims. The benchmark can still be
used for annotation-tool practice. This work does not certify its scientific
provenance or resolve the guideline ambiguities in Section 5 of the guidelines.

The LLM-cache script has no provider implementation and now exits nonzero
without writing placeholder outputs. Evaluation refuses missing or unfrozen
gold. Candidate generation additionally requires a real frozen LLM cache with `llm_cache_hash_sha256`,
`llm_cache_model`, `llm_cache_prompt_version` and `llm_cache_frozen_date` in
metadata, never
overwrites an existing review CSV, and leaves unmatched decisions blank. Final
extractor metrics remain unimplemented and fail explicitly rather than reporting
success. Human annotation, agreement review, human adjudication, gold freezing,
real provider integration and final metrics are still outstanding. No S1
predictions or metrics were generated by this change.
