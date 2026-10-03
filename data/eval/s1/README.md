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

- `responses.jsonl`: 30 frozen responses (10 simple-factual, 10 compound-factual, 10 mixed). Contains 15 LLM-generated, 10 human-written, and 5 adversarial texts across short, medium, and long lengths.
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
