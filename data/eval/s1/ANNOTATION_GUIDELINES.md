# Nullius S1 annotation guidelines

Version: s1-guidelines-v2. Frozen content identity: canonical SHA-256 recorded in
metadata.json. Changes require a new version/hash and explicitly reconciled or new
submissions. S1 measures extraction on a curated synthetic pilot with unverified
source provenance. Source-label subgroups cannot establish authorship effects.

## Independence

Use response text and these instructions only. Do not consult an extractor, search
engine, verifier, evidence corpus, another annotator's work or draft gold. Identify
assertions, not truth. Incorrect factual assertions count. Exclude greetings, pure
preferences, requests and standalone questions. Include factual presuppositions only
when the response commits to them. Document uncertain interpretations in notes.
Do not add world knowledge. Review every response, including zero-claim responses.
A and B submit independently before agreement/adjudication. No model output may be
used to create gold. Independence is a human attestation, not software proof.

## Meaning-preserving assertions

An atomic claim is one independently checkable asserted proposition with its scope
intact. Sentence, clause and claim are distinct units. Split a compound assertion
only when every resulting claim follows from the source.

1. **Modality:** preserve may, might, possibly, reportedly, believed to and equivalent
   uncertainty in claim_text. hedged=true never compensates for removing a hedge.
2. **Attribution:** "Mira said the vessel sank" asserts an attributed report. Keep
   the speaker and reporting verb. Do not also extract "The vessel sank" unless
   the response independently endorses it. Denied quotations are not endorsed facts.
3. **Disjunction:** "The key is brass or copper" is one disjunctive assertion, not
   two asserted material facts. Preserve either/or and exclusive qualifications.
4. **Conditions:** preserve if/unless/only if and their scope. An asserted conditional
   regularity is included with its condition. Pure imagined scenarios are excluded.
   "If pressure rises, the valve opens" must not become "The valve opens".
5. **Quantifiers:** retain all, some, most, exactly, at least, only and numeric bounds.
   Distribute a qualifier across split claims only when logically licensed.
6. **Negation:** preserve not, never, no and scope. "Not both A and B" does not
   assert "not A" and "not B" separately.
7. **Comparatives:** preserve direction, reference, units and time frame.
8. **Coreference:** resolve only to an unambiguous textual antecedent. Include that
   antecedent region in source_spans. If no name is given, retain the description
   (e.g. "the scientist") and note unresolved identity. Never invent a referent or
   silently choose among ambiguous ones. Preserve ambiguity for adjudication.
9. **Conjunction:** split independently asserted and/but facts, preserving shared
   subjects and qualifiers. Keep collective assertions such as "A and B met" intact.
   Appositives can assert independent properties.
10. **Rewrite:** improve readability only while preserving truth conditions.
    Keep numerical structure, dates, units and operators. Mark the operation and
    explain uncertainty in notes. Do not replace an assertion with a stronger one.

## Deterministic primary span policy

Offsets are zero-based Python Unicode characters, not UTF-8 bytes or UTF-16 units.
Spans are half-open [start,end), with 0 <= start < end <= len(text). No leading or
trailing whitespace is allowed.

Use the smallest region covering the assertion's lexical material and qualifiers.
Include attached punctuation internal to the region. Include a trailing sentence
terminator for a complete unchanged sentence. Exclude coordinating conjunctions
that merely connect independent assertions. For shared subjects/qualifiers or
coreference, record multiple ordered, disjoint source_spans, not a bounding region
containing unrelated words. Keep necessary function words within each region.
Different claims may share regions. Do not substitute full sentences for smaller
regions. Residual linguistic ambiguity needs notes and human adjudication under
this policy, without extractor outputs.

source_spans is authoritative. source_span is its first region for legacy display.
Exact span match compares the ordered regions. Character IoU compares their union.
Boundary errors compare outermost start/end. Missing prediction spans score zero
for exact/IoU and have undefined boundaries, reported with their coverage.

## Formal claim schema

Use schema_version="s1-claim-v2" with:

- response_id, claim_id: stable IDs. Include annotator identity in claim IDs. Never
  reuse a removed ID for a different claim.
- annotator_id, claim_text, notes.
- source_span and source_spans as above.
- operations: zero or more of decomposition, rewrite, coreference.
- decomposed: true exactly when operations includes decomposition.
- compound_id: response-local group ID for claims split from one compound assertion,
  otherwise null. Coreference or rewriting alone is not decomposition.
- hedged: whether the claim retains epistemic uncertainty.

Unchanged claims have no operations and exactly equal their sole source slice.
Legacy v1 claims remain readable drafts, not formal gold. During adjudication,
compound_groups explicitly link gold_claim_ids. Decomposition metrics use these
human relations, never infer compound membership from the old overloaded flag.

## Separate calibration examples

These teaching examples are not benchmark answers or human gold. Their exact
strings and spans are in calibration_examples.json and regression-tested.

- "The vessel may sink." stays modal: [0,20).
- "Mira said the vessel sank." stays attributed: [0,26).
- "The key is brass or copper." stays disjunctive: [0,27).
- "If pressure rises, the valve opens." stays conditional: [0,35).
- "The lamp is not lit." stays negative: [0,20).
- "Lena sings and dances." can become "Lena sings" [0,10) and "Lena dances"
  from [0,4) plus [15,21). Both have decomposition and compound_id="calibration-6".
  The second also has rewrite. A fragment's trailing sentence period is excluded.

## Submission, agreement and gold

The review record binds annotation bytes, canonical benchmark identity, original
raw transport hash, guideline content hash, annotator identity, independence
attestation, completed response IDs and explicit zero-claim IDs. Download drafts
frequently. Formal completion requires 30/30 responses, even with zero claims.

Validate through scripts.validate_s1_annotations with --review, --require-complete
and --formal. The shared implementation is authoritative. Lexical A/B matching is
a suggestion, not semantic agreement. Humans review every response and unmatched
claim, then record agreement and adjudication decisions. Gold freeze requires
linked A/B hashes and reviews, explicit response completion, adjudicator identity,
guideline identity and a no-system-output attestation.

Ambiguous benchmark attribution, unnamed subjects and pronouns remain unchanged.
Apply these rules and document the decision. Do not rewrite responses to simplify
annotation. Freeze metric definitions before viewing system outputs.
