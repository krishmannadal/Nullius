"""Contract tests for src/components/verifiers.py.

Two of the four silent-bug surfaces named at the start of this project live here:
**label mapping** and **tokenizer truncation**. Both were measured on the real
checkpoint before this file was written, and both turned out to be traps:

  * ``id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}`` — the reverse
    of the common MNLI convention. Hardcoding would swap Supported/Contradicted
    silently while still producing plausible-looking probabilities.
  * ``tokenizer.model_max_length == 1000000000000000019884624838656`` — so
    ``truncation=True`` on its own is a **no-op**.

The pure tests run in milliseconds. The ones that need the 370 MB checkpoint are
marked ``slow`` so a skip is visible rather than silent.
"""

from __future__ import annotations

import pytest

from src.components.verifiers import DEFAULT_MAX_LENGTH, _resolve_label_indices, make_verdict
from src.core.types import Claim, Evidence

CLAIM = Claim.new(response_id="r", text="Marie Curie was born in Warsaw.", extractor_name="t")
EV = Evidence.new(
    doc_id="Marie_Curie", sent_id=1,
    text="She was born in Warsaw, in what was then the Kingdom of Poland.",
    score=12.5, retriever_name="bm25", rank=3,
)


# --------------------------------------------------------------------------- #
# label mapping — read from config, never assumed
# --------------------------------------------------------------------------- #

def test_resolves_the_real_checkpoints_order():
    """MEASURED from MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli."""
    got = _resolve_label_indices({0: "entailment", 1: "neutral", 2: "contradiction"})
    assert got == {"entailment": 0, "neutral": 1, "contradiction": 2}


def test_resolves_the_opposite_order_too():
    """Many MNLI checkpoints ship this. Reading the config is what makes both work."""
    got = _resolve_label_indices({0: "contradiction", 1: "neutral", 2: "entailment"})
    assert got == {"entailment": 2, "neutral": 1, "contradiction": 0}


def test_label_names_are_matched_case_insensitively():
    assert _resolve_label_indices({0: "ENTAILMENT", 1: " Neutral ", 2: "Contradiction"}) == {
        "entailment": 0, "neutral": 1, "contradiction": 2
    }


def test_refuses_to_guess_when_labels_are_anonymous():
    """A LABEL_0/LABEL_1/LABEL_2 head has no recoverable mapping. Guessing one would
    swap two classes silently, so this must raise."""
    with pytest.raises(ValueError, match="cannot resolve NLI label"):
        _resolve_label_indices({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_refuses_a_two_class_head():
    with pytest.raises(ValueError, match="cannot resolve NLI label"):
        _resolve_label_indices({0: "entailment", 1: "not_entailment"})


def test_refuses_a_duplicated_label_name():
    with pytest.raises(ValueError, match="cannot resolve NLI label"):
        _resolve_label_indices({0: "entailment", 1: "entailment", 2: "contradiction"})


# --------------------------------------------------------------------------- #
# make_verdict — the retrieval signal an aggregator would otherwise never see
# --------------------------------------------------------------------------- #

def test_make_verdict_copies_rank_and_score_off_the_evidence():
    """WeightedByRetrievalAggregator reads these. If a verifier forgot to set them it
    would silently fall back to uniform weights for that verifier only."""
    v = make_verdict(CLAIM, EV, verifier_name="t", latency_ms=1.0,
                     p_entail=0.7, p_contra=0.2, p_neutral=0.1)
    assert v.evidence_rank == 3
    assert v.evidence_score == pytest.approx(12.5)
    assert v.claim_id == CLAIM.id and v.evidence_id == EV.id


def test_make_verdict_supports_similarity_only():
    v = make_verdict(CLAIM, EV, verifier_name="similarity", latency_ms=1.0, similarity=0.42)
    assert not v.has_nli and v.similarity == pytest.approx(0.42)


# --------------------------------------------------------------------------- #
# the real model
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def nli():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from src.components.verifiers import NLIVerifier

    return NLIVerifier()


@pytest.mark.slow
def test_label_mapping_behaves_as_the_names_claim(nli):
    """Known-answer probe. The simplex check in EvidenceVerdict cannot catch a swapped
    mapping — the probabilities still sum to 1. Only behaviour can."""
    entailed = nli.score(
        Claim.new("r", "A person is making music.", "t"),
        Evidence.new("D", 0, "A man is playing a guitar.", 1.0, "t", 1),
    )
    refuted = nli.score(
        Claim.new("r", "Nobody is playing any instrument.", "t"),
        Evidence.new("D", 0, "A man is playing a guitar.", 1.0, "t", 1),
    )
    neutral = nli.score(
        Claim.new("r", "The weather in Oslo is cold today.", "t"),
        Evidence.new("D", 0, "A man is playing a guitar.", 1.0, "t", 1),
    )
    assert entailed.p_entail > 0.8, entailed
    assert refuted.p_contra > 0.8, refuted
    assert neutral.p_neutral > 0.8, neutral


@pytest.mark.slow
def test_evidence_is_the_premise_not_the_hypothesis(nli):
    """Direction is asymmetric and getting it backwards yields a model that looks
    merely bad. 'Marie Curie was Polish' does not entail the longer biography, but the
    biography does entail it."""
    long_ev = ("Marie Salomea Sklodowska-Curie was a Polish and naturalised-French "
               "physicist and chemist who conducted pioneering research on radioactivity.")
    correct = nli.score(
        Claim.new("r", "Marie Curie was Polish.", "t"),
        Evidence.new("D", 0, long_ev, 1.0, "t", 1),
    )
    assert correct.p_entail > 0.5, correct


@pytest.mark.slow
def test_probabilities_form_a_simplex(nli):
    v = nli.score(CLAIM, EV)
    assert v.p_entail + v.p_contra + v.p_neutral == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= p <= 1.0 for p in (v.p_entail, v.p_contra, v.p_neutral))


@pytest.mark.slow
def test_truncation_actually_truncates(nli):
    """THE test for footgun #2. model_max_length is ~1e19, so truncation=True alone
    does nothing; only an explicit max_length works. A 20,000-word evidence must come
    out at exactly max_length tokens, not at 20,000."""
    huge = "Warsaw is a city in Poland. " * 4000
    enc = nli.tokenizer(
        huge, CLAIM.text,
        truncation="longest_first", max_length=nli.max_length, return_tensors="pt",
    )
    assert enc["input_ids"].shape[1] == nli.max_length == DEFAULT_MAX_LENGTH


@pytest.mark.slow
def test_without_explicit_max_length_truncation_is_a_no_op(nli):
    """Demonstrates the trap directly, so the reason for the explicit argument is not
    something you have to take on faith."""
    assert nli.tokenizer.model_max_length > 10 ** 12   # the sentinel
    huge = "Warsaw is a city in Poland. " * 4000
    unbounded = nli.tokenizer(huge, CLAIM.text, truncation=True)   # NO max_length
    bounded = nli.tokenizer(huge, CLAIM.text, truncation=True, max_length=nli.max_length)
    assert len(unbounded["input_ids"]) > 10_000     # truncation=True did nothing
    assert len(bounded["input_ids"]) == nli.max_length


@pytest.mark.slow
def test_a_very_long_evidence_still_scores_without_error(nli):
    huge = Evidence.new("D", 0, "Warsaw is a city in Poland. " * 4000, 1.0, "t", 1)
    v = nli.score(CLAIM, huge)
    assert v.p_entail + v.p_contra + v.p_neutral == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
def test_longest_first_truncation_preserves_a_short_claim(nli):
    """With truncation_side='right', the naive policy would eat the end of the pair.
    longest_first removes from whichever member is longer, so a short claim survives
    intact even against enormous evidence."""
    huge = "Warsaw is a city in Poland. " * 4000
    short_claim = "Marie Curie was born in Warsaw."
    ids = nli.tokenizer(huge, short_claim, truncation="longest_first",
                        max_length=nli.max_length)["input_ids"]
    decoded = nli.tokenizer.decode(ids)
    assert "Marie Curie" in decoded and "Warsaw" in decoded


@pytest.mark.slow
def test_latency_is_measured_not_constant(nli):
    a = nli.score(CLAIM, EV)
    b = nli.score(CLAIM, EV)
    assert a.latency_ms > 0 and b.latency_ms > 0
    assert a.latency_ms != b.latency_ms      # a constant would be identical


@pytest.mark.slow
def test_verifier_is_deterministic_on_the_same_pair(nli):
    a = nli.score(CLAIM, EV)
    b = nli.score(CLAIM, EV)
    assert a.p_entail == pytest.approx(b.p_entail, abs=1e-5)


# --------------------------------------------------------------------------- #
# null baseline
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_claim_only_ignores_the_evidence_entirely():
    """The null baseline's defining property: two contradictory pieces of evidence for
    the same claim must produce IDENTICAL scores, because neither is read."""
    pytest.importorskip("transformers")
    from src.components.verifiers import ClaimOnlyVerifier

    v = ClaimOnlyVerifier()
    supporting = Evidence.new("D", 0, "She was born in Warsaw, Poland.", 1.0, "t", 1)
    refuting = Evidence.new("D", 1, "She was born in Paris, France.", 1.0, "t", 1)
    a = v.score(CLAIM, supporting)
    b = v.score(CLAIM, refuting)
    assert a.p_entail == pytest.approx(b.p_entail, abs=1e-6)
    assert a.p_contra == pytest.approx(b.p_contra, abs=1e-6)


@pytest.mark.slow
def test_claim_only_still_keys_its_verdict_to_the_evidence():
    """It must stay on the same code path so aggregators work unchanged."""
    pytest.importorskip("transformers")
    from src.components.verifiers import ClaimOnlyVerifier

    out = ClaimOnlyVerifier().score(CLAIM, EV)
    assert out.evidence_id == EV.id
    assert out.evidence_rank == EV.rank


@pytest.mark.slow
def test_real_nli_and_claim_only_disagree_on_a_refuting_pair():
    """If these ever agreed, the pipeline would not be reading evidence at all."""
    pytest.importorskip("transformers")
    from src.components.verifiers import ClaimOnlyVerifier, NLIVerifier

    refuting = Evidence.new("D", 0, "Marie Curie was born in Paris, France.", 1.0, "t", 1)
    real = NLIVerifier().score(CLAIM, refuting)
    null = ClaimOnlyVerifier().score(CLAIM, refuting)
    assert real.p_contra > null.p_contra


# --------------------------------------------------------------------------- #
# similarity
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_similarity_returns_a_cosine_and_no_probabilities():
    pytest.importorskip("sentence_transformers")
    from src.components.verifiers import SimilarityVerifier

    out = SimilarityVerifier().score(CLAIM, EV)
    assert out.similarity is not None and -1.0 <= out.similarity <= 1.0
    assert not out.has_nli
    assert out.evidence_rank == EV.rank


@pytest.mark.slow
def test_similarity_cannot_tell_support_from_contradiction():
    """The structural limitation, demonstrated rather than asserted: 'born in Warsaw'
    and 'born in Paris' are near-identical strings, so cosine ranks the REFUTING
    evidence about as high as the supporting evidence."""
    pytest.importorskip("sentence_transformers")
    from src.components.verifiers import SimilarityVerifier

    v = SimilarityVerifier()
    supporting = v.score(CLAIM, Evidence.new("D", 0, "Marie Curie was born in Warsaw.", 1.0, "t", 1))
    refuting = v.score(CLAIM, Evidence.new("D", 1, "Marie Curie was born in Paris.", 1.0, "t", 1))
    assert abs(supporting.similarity - refuting.similarity) < 0.15, (
        f"support={supporting.similarity:.3f} refute={refuting.similarity:.3f} — "
        "if these separated cleanly, the claim in the docs would be wrong"
    )
