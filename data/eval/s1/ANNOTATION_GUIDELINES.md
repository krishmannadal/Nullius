# Nullius S1 Annotation Guidelines

**Evaluation Phase:** S1 — Human Gold Extraction Benchmark  
**Document Version:** 1.0 (Operational)  
**Target Output:** `annotations_A.jsonl` (Annotator A), `annotations_B.jsonl` (Annotator B)  

---

## 1. Scientific Purpose & Workflow

Nullius evaluates hallucination detection pipelines. Stage 1 (S1) measures how accurately claim extractors (`SpacySentenceExtractor` and `LLMClaimExtractor`) extract atomic factual claims from model responses and align them to source spans.

To create an uncorrupted benchmark, **two independent human annotators** (Annotator A and Annotator B) annotate the 30 responses in `data/eval/s1/responses.jsonl` in strict isolation.
- Annotators see **only** the raw response texts.
- Annotators **never** see system extractor outputs, LLM predictions, or each other's work.
- Only after both files (`annotations_A.jsonl` and `annotations_B.jsonl`) are completed and mechanically validated will inter-annotator agreement be analyzed and adjudication take place.

---

## 2. Annotation Schema & Record Structure

Each line in `annotations_A.jsonl` and `annotations_B.jsonl` must be a single valid JSON object containing exactly the following fields:

```json
{
  "response_id": "s1-resp-01",
  "claim_id": "s1-A-001",
  "claim_text": "Marie Curie was born in Warsaw.",
  "source_span": {
    "start": 0,
    "end": 31
  },
  "decomposed": false,
  "hedged": false,
  "annotator_id": "A",
  "notes": ""
}
```

### Field Definitions:
- `response_id` (*string*, required): The identifier of the response from `data/eval/s1/responses.jsonl` (e.g. `"s1-resp-01"`).
- `claim_id` (*string*, required): A unique identifier for the claim within the annotation file (e.g. `"s1-A-001"` for Annotator A, `"s1-B-001"` for Annotator B).
- `claim_text` (*string*, required): The standalone, self-contained atomic factual assertion.
- `source_span` (*object*, required): Character offsets into the original response string (`response["text"]`):
  - `start` (*integer*): 0-indexed character start position (inclusive).
  - `end` (*integer*): 0-indexed character end position (exclusive).
  - Constraint: `0 <= start < end <= len(response_text)`.
- `decomposed` (*boolean*, required): `true` if `claim_text` was rewritten, decomposed from a compound sentence, or had pronouns resolved; `false` if `claim_text` is the exact character substring `response_text[start:end]`.
- `hedged` (*boolean*, required): `true` if the original response framed the claim with uncertainty, belief, or epistemic hedging (e.g., "is believed to", "reportedly", "might"); `false` if asserted as a definite fact.
- `annotator_id` (*string*, required): Identifier of the annotator (`"A"` or `"B"`).
- `notes` (*string*, optional, default `""`): Human annotator notes documenting rationale, uncertainty, or linguistic edge cases for adjudication.

---

## 3. What Counts as a Gold Claim?

An annotator marks a claim if and only if the statement satisfies three fundamental criteria:

1. **Factual:** It asserts a proposition that is empirically verifiable or falsifiable against real-world evidence (e.g. Wikipedia, scientific consensus, historical records).
   - *Factual:* "Mount Everest is 8,848 meters tall."
   - *Non-factual:* "Python is a beautiful language." (Subjective aesthetic opinion).
2. **Atomic:** It expresses exactly *one* verifiable fact. If a sentence asserts multiple distinct facts, it must be decomposed into multiple atomic claims.
   - *Sentence:* "Marie Curie was born in Warsaw and won two Nobel Prizes."
   - *Claim 1:* "Marie Curie was born in Warsaw."
   - *Claim 2:* "Marie Curie won two Nobel Prizes."
3. **Self-Contained:** When read in total isolation, the claim must be fully interpretable without requiring reference to the surrounding context, discourse markers, or unresolved pronouns.
   - *Not self-contained:* "It was completed in 1889."
   - *Self-contained:* "The Eiffel Tower was completed in 1889."

---

## 4. Operational Guidelines by Linguistic Phenomenon

### 4.1 Atomic Factual Claims
- If a sentence contains a single, unambiguous factual proposition, extract it directly.
- If the exact sentence is self-contained and requires no rewriting, set:
  - `source_span`: covers the exact sentence span.
  - `decomposed`: `false`
  - `claim_text`: exact text of the span.

### 4.2 Compound Claims & Conjunctions
- When clauses are joined by coordinating conjunctions ("and", "but", "or"), split them into separate atomic claims.
- *Example:* "Water boils at 100°C and freezes at 0°C at sea level."
  - Claim 1: "Water boils at 100°C at sea level." (`decomposed: true`)
  - Claim 2: "Water freezes at 0°C at sea level." (`decomposed: true`)

### 4.3 Multiple Facts in One Sentence
- Sentences often pack multiple facts through appositives, relative clauses, or multiple modifiers. Decompose each distinct fact into its own claim.
- *Example:* "Charles Darwin, an English naturalist, proposed the theory of evolution by natural selection in 1859."
  - Claim 1: "Charles Darwin was an English naturalist." (`decomposed: true`)
  - Claim 2: "Charles Darwin proposed the theory of evolution by natural selection in 1859." (`decomposed: true`)

### 4.4 Opinions, Beliefs, & Subjective Judgments
- Do **not** extract pure opinions, aesthetic judgments, emotional states, or conversational pleasantries.
- *Examples to EXCLUDE:*
  - "Python is a beautiful language." (Opinion)
  - "It's truly a majestic sight to behold." (Opinion)
  - "It is a very sad part of history." (Subjective comment)
  - "They are very cool." (Opinion)
  - "I think John is funny." (Opinion)

### 4.5 Questions & Conversational Openers
- Do not extract questions as claims.
- Do not extract conversational preamble or AI boilerplate (e.g., "Sure, I can help!", "Here is the information you requested.", "Please let me know if you need anything else.").

### 4.6 Embedded Factual Assertions & Presuppositions
- If a question or introductory clause contains an embedded factual proposition that the text assumes as true, extract the factual proposition.
- *Example:* "Did you know that Mount Everest is 8,848 meters tall?"
  - Extract Claim: "Mount Everest is 8,848 meters tall." (`decomposed: true`, `source_span` covering the question or the factual clause).
- *Example:* "You asked if Paris is in Germany, but actually, Paris is in France."
  - Extract Claim: "Paris is in France." (`decomposed: true`). Do not extract the user premise "You asked...".

### 4.7 Conditionals & Counterfactuals
- **General scientific laws / rules stated conditionally:** If a conditional expresses an invariant factual relationship, extract the factual assertion:
  - "If water is heated to 100°C at 1 atm, it boils." $\rightarrow$ Extract: "Water boils at 100°C at 1 atm." (`decomposed: true`).
- **Hypotheticals and Counterfactuals:** Do **not** extract hypothetical or counterfactual statements as facts:
  - "If unicorns existed, they would be majestic." $\rightarrow$ DO NOT EXTRACT.
  - However, in "However, rhinos do exist, and they live in Africa and Asia", extract the factual assertions about rhinos!

### 4.8 Numerical Claims, Quantities, & Measurements
- Preserve exact numbers, dates, units, and ranges.
- Do not round or alter numerical quantities during rewriting.
- *Example:* "The population of Paris is 2.1 million." $\rightarrow$ Claim: "The population of Paris is 2.1 million."

### 4.9 Citations, References, & URLs
- Do not annotate raw URLs or citation markers (`[1]`, `http://...`) as claims.
- Extract the underlying factual assertion made in the sentence containing the citation.

### 4.10 Hedged Statements & Epistemic Modality
- When a response hedges an assertion (using verbs like "believed to", "reported", "suggests", "might", "alleged"), extract the underlying factual proposition and set `hedged: true`.
- *Example:* "Curie is believed to have died from radiation exposure."
  - Claim: "Marie Curie died from radiation exposure." (or "Marie Curie is believed to have died from radiation exposure.")
  - Set: `hedged: true`, `decomposed: true`.
  - (See Section 5 regarding attribution vs assertion nuances).

### 4.11 Pronouns, Coreference, & Anaphora
- Pronouns ("it", "he", "she", "they", "this") **must** be resolved to their explicit nominal referent to ensure the claim is self-contained.
- *Example:* In Response 05: "Jupiter is the largest planet in our solar system. It is a gas giant. It has 79 known moons."
  - Sentence 2: "It is a gas giant." $\rightarrow$ Claim: "Jupiter is a gas giant." (`decomposed: true`).
  - Sentence 3: "It has 79 known moons." $\rightarrow$ Claim: "Jupiter has 79 known moons." (`decomposed: true`).

### 4.12 Rewritten & Decomposed Claims (The Minimal Rewrite Principle)
- When resolving pronouns, splitting conjunctions, or extracting appositives:
  - Make the **minimal grammatical edit** necessary to render the claim self-contained and grammatically sound.
  - Do not introduce external background knowledge not implied by the text.
  - Whenever any character in `claim_text` differs from `response_text[start:end]` (beyond identical substring match), you MUST set `decomposed: true`.
  - If `claim_text == response_text[start:end]`, set `decomposed: false`.

### 4.13 Source Spans: Indexing & Boundary Rules
- Every claim must have a `source_span` with integer `start` and `end`.
- Indexing is **0-based, half-open interval `[start, end)`** into Python string indices of the response text:
  `extracted_slice = response_text[start:end]`
- The span must cover the **minimal clause or sentence** from which the claim is derived.
- Boundary hygiene: Do not include extraneous leading or trailing whitespace.

### 4.14 Overlapping & Shared Spans
- Multiple atomic claims derived from the same sentence or clause **may share the exact same `source_span`** (or share overlapping spans).
- *Example:* "Marie Curie was born in Warsaw and won two Nobel Prizes." (span `[0, 58)`). Both Claim 1 and Claim 2 may point to `[0, 58)` with `decomposed: true`.

### 4.15 Ambiguous Cases & Annotator Notes
- When facing ambiguous syntax, dual referents, or borderline subjective phrasing, use your best judgment to follow the core definitions.
- Document your reasoning in the `notes` field.
- During human adjudication, all claims with discrepant annotations or flagged notes will be resolved in conference.

---

## 5. Protocol Ambiguities Requiring Human Decision

During the formal review of the 30 responses against the S1 protocol, the following seven linguistic and structural ambiguities were identified. **Annotators should annotate according to their independent judgment, and adjudicators must explicitly review these cases.**

### Ambiguity 1: Span Granularity for Decomposed Coordinate Clauses
*Context:* When decomposing a sentence like Response 11:  
`"Marie Curie was born in Warsaw and won two Nobel Prizes."`  
- *Option A (Sentence-level span):* Both atomic claims share the entire sentence span `[0, 58)`.
- *Option B (Clause-level sub-span):* Claim 1 takes `"Marie Curie was born in Warsaw"` (`[0, 31)`), while Claim 2 takes `"and won two Nobel Prizes"` (`[32, 58)`) or the full sentence.  
*Guideline:* Annotators may choose either; the validator permits both. Adjudication will harmonize the granularity convention.

### Ambiguity 2: Reported Speech & Adversarial Framing
*Context:* In Response 27:  
`"According to John, 'The sky is green', but we know the sky is blue due to Rayleigh scattering. I think John is funny."`  
- Does `"The sky is green"` count as an extracted claim (with `hedged: true` or attribution)?
- Or does the annotator extract `"John said that the sky is green"` (a claim about John's utterance)?
- Or is `"The sky is green"` rejected because the response explicitly refutes it in the same sentence?  
*Guideline:* Extract the factual claims asserted by the speaker ("The sky is blue", "The sky is blue due to Rayleigh scattering"). If you annotate John's statement, mark `hedged: true` and add a note.

### Ambiguity 3: Conversational AI Persona Assertions
*Context:* In Response 22:  
`"As an AI, I don't have feelings, but I can tell you that the Moon orbits the Earth."`  
- Is `"An AI does not have feelings"` or `"The speaker is an AI"` a verifiable factual claim, or conversational filler / persona metadata outside the scope of factual inspection?  
*Guideline:* The primary factual proposition is "The Moon orbits the Earth." Annotators should note whether they consider the persona clause factual.

### Ambiguity 4: Presuppositions in Rhetorical Questions
*Context:* In Response 23:  
`"Did you know that Mount Everest is 8,848 meters tall? It's truly a majestic sight to behold."`  
- The question presupposes that Mount Everest is 8,848 meters tall.  
*Guideline:* Extract the presupposed fact: "Mount Everest is 8,848 meters tall." Discard the question framing and the subjective second sentence ("It's truly a majestic sight...").

### Ambiguity 5: Multi-Relational Comparative Claims
*Context:* In Response 20:  
`"The Atlantic Ocean is smaller than the Pacific Ocean but larger than the Indian Ocean."`  
- Does this yield two atomic claims:
  1. "The Atlantic Ocean is smaller than the Pacific Ocean."
  2. "The Atlantic Ocean is larger than the Indian Ocean."
- Or does it also imply a third claim: "The Pacific Ocean is larger than the Indian Ocean"?  
*Guideline:* Extract only direct assertions (the two claims comparing the Atlantic Ocean). Do not derive transitive inferences.

### Ambiguity 6: Multi-Referent Temporal Alignment ("respectively")
*Context:* In Response 29:  
`"Alice and Bob were born in 1990 and 1992, respectively. They are both wonderful people."`  
- Decomposing "respectively" requires mapping referent 1 to date 1 and referent 2 to date 2.  
*Guideline:* Extract two claims: "Alice was born in 1990." and "Bob was born in 1992." Set `decomposed: true` for both. Omit the subjective second sentence.

### Ambiguity 7: Complex Multi-Predicate Relative Clauses
*Context:* In Response 30:  
`"The scientist, who was born in 1901, won the prize in 1954 and he donated it to charity, which was very generous."`  
- Contains: birth date (1901), winning prize (1954), donating to charity, and subjective assessment ("generous").  
*Guideline:* Decompose into 3 atomic factual claims (born in 1901, won prize in 1954, donated prize to charity). Exclude the subjective clause ("which was very generous").

---

## 6. Annotator Self-Validation Checklist

Before submitting your annotation file (`annotations_A.jsonl` or `annotations_B.jsonl`), run the mechanical validator:

```bash
python scripts/validate_s1_annotations.py data/eval/s1/annotations_A.jsonl --annotator-id A --show-spans
```

Verify that:
1. All 30 responses have been inspected.
2. Every `claim_id` is unique.
3. Every `source_span` correctly slices into the response text: `0 <= start < end <= len(text)`.
4. No subjective opinions or conversational chit-chat are included as claims.
5. All pronouns are resolved (`decomposed: true`).
6. All hedged or attributed claims have `hedged: true`.
7. Exit code from the validator is 0.
