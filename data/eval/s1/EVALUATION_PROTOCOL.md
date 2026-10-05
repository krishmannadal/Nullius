# S1 extraction pilot protocol v2

This is a curated synthetic pilot with unverified source provenance. All 30 response
texts and IDs remain frozen. The 15/10/5 source labels are unverified design labels,
not evidence of authorship. Length labels overlap and are curated categories, not
disjoint numerical bins. Aggregate results describe this pilot only. Any subgroup
summary is exploratory and cannot establish source or population effects.

## Order and gates

Freeze these definitions and the annotation instructions before formal annotation.
A/B independently annotate raw responses. Complete reviews include explicit zero
claims. Humans finalize agreement and adjudication before gold freeze. No extractor
output may be used to establish gold. Hashes detect changes, not human honesty.
Human identities and independence remain researcher attestations.

Gold freeze binds all 30 response reviews, A/B files/reviews, finalized agreement,
adjudication, guideline content, this protocol, timestamp and software revision.
Every gold claim has source regions. An entirely empty gold file is rejected for
this benchmark. Individual zero-claim responses are valid and explicitly recorded.

Only then generate real provider outputs, freeze both extractor predictions and
complete human system-to-gold matching. No missing prerequisite is synthesized.
Keep human decision versions as separate files. Never overwrite a frozen artifact.
For revisions, record previous_matching_sha256, preserve the old file and refreeze
downstream reports. Re-run a changed experiment in a separate directory.

## Evaluated systems

spaCy sentence extraction uses its recorded model/package version and exact sentence
boundaries, with the extractor's min_chars=3 default recorded. It is a sentence
baseline, not a factual-claim selector. It does not predict epistemic status.

S1 LLM extraction uses the exact frozen system/user prompts, provider/model/settings,
raw API response and JSON processing version. It preserves duplicate claims and
model source_spans. No heuristic span inference or deduplication occurs in this S1
path. The older interactive LLMClaimExtractor cache adapter remains a distinct debug
system with explicit deduplication/heuristic alignment. Do not mix their results.
Unknown provider weight revision stays unknown. API model names alone do not prove
weight reproducibility. Frozen outputs allow deterministic evaluation replay.

## Human matching

Normalize with Unicode NFC, case folding and whitespace collapse only. Tokenization
preserves punctuation/operators and numeric structure. Exact normalized equality
or multiset token F1 >= 0.8 creates an eligible candidate. Maximum-weight bipartite
assignment suggests a one-to-one alignment. Neither step decides semantic identity.

All within-response pairs remain in the JSON decision form, including below-threshold
pairs for reassignment. A human accepts, rejects or reassigns each pair. Accepted
pairs require rationale. Explicit unmatched gold and predictions complete coverage.
One prediction matches at most one gold claim and vice versa. Duplicate texts retain
separate IDs/spans. An under-split prediction cannot earn several true positives.
Human matching must preserve assertion polarity, numbers, modality and attribution.

## Claim metrics

TP = accepted one-to-one pairs. FP = predictions minus TP. FN = gold minus TP.
Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN).
If both sets are empty, all three are 1. Otherwise a zero precision/recall denominator
scores 0. Report per-response values. Macro is the unweighted mean over responses.
Micro sums TP/FP/FN across responses before applying the same formulas.

## Span metrics

Evaluate semantically accepted pairs only. Exact match requires identical ordered
region lists. Character IoU is intersection/union over source character positions,
not the bounding interval. Missing prediction spans score 0 for exact and IoU.
Start/end boundary errors are absolute Unicode-character differences between the
outermost boundaries. Missing spans have undefined boundary errors, not zero error.
Report matched-pair and valid-boundary counts/coverage. No accepted pairs makes span
metrics undefined. Within a response average defined pair values; macro means then
average defined response values. These are conditional alignment metrics, alongside
claim recall, not a substitute for missed-claim accounting.

## Decomposition

Gold adjudication defines compound_groups as two or more independently asserted gold
claims sharing a compound source. Human system matching explicitly associates
prediction IDs with each group, independently of the legacy decomposed flag.
A prediction belongs to at most one compound group in this pilot. Cross-group
under-splitting needs a documented adjudication choice and error annotation.

Compound claim recall = accepted related prediction/gold pairs / gold compound claims.
Compound claim precision = accepted related pairs / predictions assigned to compound
groups. A pair counts only when its prediction and gold belong to the SAME group.
Split ratio = assigned compound predictions / compound gold claims (ideal count 1).
Under/over-split rates = fractions of compound groups explicitly judged under/over
split by the human reviewer. These judgments distinguish omission from under-splitting
and spurious content from over-splitting. Both may apply to a complex group.
No compound groups means undefined decomposition metrics. No related predictions
makes compound precision undefined, recall and split ratio zero if gold groups exist.

## Error taxonomy

M missed claim; S spurious claim; U under-split; O over-split; R unsupported rewrite;
P partial match; B boundary error; C coreference failure; D duplicate; N non-claim.
Humans may assign several codes to one target with a rationale. Every unmatched gold
requires M and every unmatched prediction requires S. Additional codes describe why.
Report target counts per code, not percentages that imply mutually exclusive errors.

## Uncertainty and interpretation

Primary contrast: mean response-level metric difference, LLM minus spaCy. Resample
whole paired responses with replacement, 10,000 times, using NumPy default_rng seed
1337 by default. Report mean, median, percentile 95% interval and paired-response
denominator. Never resample individual claims independently. Undefined metrics use
paired complete response cases, with that reduced denominator explicitly reported.
Positive differences favor LLM for accuracy metrics, but negative boundary-error
differences favor LLM. Split-ratio differences have no monotonic quality interpretation.
Individual extractor macro intervals use their own defined response units; micro
intervals recompute pooled counts after resampling responses. Paired micro intervals
use the SAME sampled response indices for both systems. The fixed curated sample
limits generalization regardless of interval width.

No S1 result establishes retrieval, verifier, aggregator, calibration or scientific
hallucination-detection effectiveness. Debug/FEVER-derived harness metrics are not
official FEVER evaluation. No claim of baseline superiority exists before results.
