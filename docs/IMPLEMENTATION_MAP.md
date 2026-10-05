# Research engineering dependency map

Inspected before implementation: existing diff, S1 artifacts, annotation UI and
validators, agreement/evaluation, extractor adapters, pipeline/types/contracts,
service/API, test writes, extension source/tests/ZIP, dependency locks and CI.

- `s1_annotations`: transport identity and deterministic record/review validation.
- `s1_workflow`: benchmark, complete A/B, human agreement/adjudication and gold freeze.
- `generate_s1_llm_cache`: real provider requests only after the gold gate.
- `s1_predictions`: real provider + spaCy output freezing, independent of retrieval/NLI.
- `extraction_eval`: lexical suggestions, explicit human decisions, metrics and paired CI.
- `s1_annotation`: session-local manual collection; imports no inference component.
- `service` / `pipeline`: interactive inspection, distinct from S1; pairwise verification
  followed by separate pure aggregation. Volatile state is not a database.
- `src/extension`: authoritative browser source; reproducible ZIP is derived from it.

Existing benchmark response bytes are preserved. Legacy result files are historical
debug artifacts and are neither deleted nor promoted to scientific evidence.
