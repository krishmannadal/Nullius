"""Contract tests for src/components/aggregators.py.

All pure — no models, no corpus, milliseconds. Every expected value is computed by
hand in the test or its docstring, so a failure tells you which arithmetic changed
rather than "the number moved".

The four things being pinned down:
  1. the pooling formulas are what the docstrings claim;
  2. zero evidence is handled identically by all five and never guesses;
  3. similarity-only verdicts cannot produce Contradicted (a property of cosine, not a
     bug) and every aggregator records that it is running on a crippled signal;
  4. every aggregator writes a usable explanation — enforced by the type, but the
     content is checked here.
"""

from __future__ import annotations

import pytest

from src.components.aggregators import (
    MajorityAggregator,
    MaxEntailmentAggregator,
    NoisyOrAggregator,
    ThresholdWithAbstainAggregator,
    WeightedByRetrievalAggregator,
    support_contra_neutral,
)
from src.core.types import Claim, EvidenceVerdict, Label, evidence_id

CLAIM = Claim.new(response_id="r", text="Marie Curie was born in Warsaw.", extractor_name="t")

ALL_AGGREGATORS = [
    MaxEntailmentAggregator,
    NoisyOrAggregator,
    WeightedByRetrievalAggregator,
    ThresholdWithAbstainAggregator,
    MajorityAggregator,
]


def nli(doc: str, e: float, c: float, n: float, rank: int = 1, score: float = 1.0) -> EvidenceVerdict:
    return EvidenceVerdict(
        claim_id=CLAIM.id, evidence_id=evidence_id(doc, 0),
        p_entail=e, p_contra=c, p_neutral=n, similarity=None,
        verifier_name="test", latency_ms=1.0, evidence_rank=rank, evidence_score=score,
    )


def sim(doc: str, cos: float, rank: int = 1) -> EvidenceVerdict:
    return EvidenceVerdict(
        claim_id=CLAIM.id, evidence_id=evidence_id(doc, 0),
        p_entail=None, p_contra=None, p_neutral=None, similarity=cos,
        verifier_name="similarity", latency_ms=1.0, evidence_rank=rank, evidence_score=cos,
    )


# --------------------------------------------------------------------------- #
# signal extraction
# --------------------------------------------------------------------------- #

def test_nli_verdict_yields_its_three_probabilities_unchanged():
    assert support_contra_neutral(nli("A", 0.7, 0.2, 0.1)) == (0.7, 0.2, 0.1)


def test_similarity_is_rescaled_to_a_support_signal_with_no_contradiction():
    """Cosine cannot express contradiction: 'born in Warsaw' and 'born in Paris' are
    near-identical strings. So it maps to support only, and the other two are None."""
    s, c, n = support_contra_neutral(sim("A", 0.5))
    assert s == pytest.approx(0.75)      # (0.5 + 1) / 2
    assert c is None and n is None


def test_a_verdict_with_no_signal_at_all_raises():
    with pytest.raises(ValueError):
        EvidenceVerdict(CLAIM.id, "ev::A::0", None, None, None, None, "t", 1.0)


# --------------------------------------------------------------------------- #
# 1. max entailment
# --------------------------------------------------------------------------- #

def test_max_entailment_takes_the_per_class_maximum():
    """Hand-computed: max entail = 0.8 (B), max contra = 0.6 (A), max neutral = 0.3 (C).
    argmax -> Supported at 0.8, driven by B."""
    v = [nli("A", 0.3, 0.6, 0.1), nli("B", 0.8, 0.1, 0.1), nli("C", 0.5, 0.2, 0.3)]
    out = MaxEntailmentAggregator().aggregate(CLAIM, v)
    assert out.label is Label.SUPPORTED
    assert out.confidence == pytest.approx(0.8)
    assert out.aggregation_trace["decisive_evidence_ids"] == [evidence_id("B", 0)]
    assert out.aggregation_trace["pooled"]["contradict"] == pytest.approx(0.6)


def test_max_entailment_is_decided_by_one_piece_of_evidence():
    """The stated weakness: a single spurious high-entailment sentence outvotes any
    number of neutral ones, because nothing else is consulted."""
    neutral_crowd = [nli(f"N{i}", 0.05, 0.05, 0.90) for i in range(10)]
    assert MaxEntailmentAggregator().aggregate(CLAIM, neutral_crowd).label is Label.INSUFFICIENT
    with_one_outlier = neutral_crowd + [nli("X", 0.95, 0.03, 0.02)]
    assert MaxEntailmentAggregator().aggregate(CLAIM, with_one_outlier).label is Label.SUPPORTED


def test_max_entailment_records_the_neutral_conflation_in_its_rule():
    out = MaxEntailmentAggregator().aggregate(CLAIM, [nli("A", 0.1, 0.1, 0.8)])
    assert out.label is Label.INSUFFICIENT
    assert "neutral" in out.aggregation_trace["rule"].lower()
    assert "conflate" in out.aggregation_trace["rule"].lower()


# --------------------------------------------------------------------------- #
# 2. noisy-OR
# --------------------------------------------------------------------------- #

def test_noisy_or_matches_the_formula():
    """P = 1 - prod(1 - p). Support: 1 - (0.7 * 0.6 * 0.5) = 1 - 0.21 = 0.79.
    Contra:  1 - (0.9 * 0.9 * 0.9) = 1 - 0.729 = 0.271."""
    v = [nli("A", 0.3, 0.1, 0.6), nli("B", 0.4, 0.1, 0.5), nli("C", 0.5, 0.1, 0.4)]
    out = NoisyOrAggregator().aggregate(CLAIM, v)
    assert out.aggregation_trace["pooled"]["support"] == pytest.approx(0.79)
    assert out.aggregation_trace["pooled"]["contradict"] == pytest.approx(1 - 0.9 ** 3)
    assert out.label is Label.SUPPORTED
    assert out.confidence == pytest.approx(0.79)


def test_noisy_or_saturates_with_k_on_duplicate_evidence():
    """The named weakness. Five near-duplicate sentences at p=0.3 are counted as five
    independent witnesses: 1 - 0.7^5 = 0.832. This is why the winner may change with k."""
    one = NoisyOrAggregator().aggregate(CLAIM, [nli("A", 0.3, 0.0, 0.7)])
    five = NoisyOrAggregator().aggregate(CLAIM, [nli(f"D{i}", 0.3, 0.0, 0.7) for i in range(5)])
    assert one.aggregation_trace["pooled"]["support"] == pytest.approx(0.3)
    assert five.aggregation_trace["pooled"]["support"] == pytest.approx(1 - 0.7 ** 5)
    assert one.label is Label.INSUFFICIENT      # 0.30 < floor 0.5
    assert five.label is Label.SUPPORTED        # 0.83 >= floor 0.5, on identical evidence


def test_noisy_or_below_the_floor_is_insufficient():
    out = NoisyOrAggregator(decision_floor=0.9).aggregate(CLAIM, [nli("A", 0.5, 0.1, 0.4)])
    assert out.label is Label.INSUFFICIENT
    assert out.aggregation_trace["decision_floor"] == 0.9


# --------------------------------------------------------------------------- #
# 3. weighted by retrieval
# --------------------------------------------------------------------------- #

def test_weighted_by_retrieval_uses_rank_not_score():
    """Rank 1 gets weight 1, rank 2 gets 1/2. Support = (1*0.9 + 0.5*0.1)/1.5 = 0.6333.
    The retrieval SCORES are deliberately inverted here: if the aggregator used score,
    the answer would differ."""
    v = [nli("A", 0.9, 0.05, 0.05, rank=1, score=0.1),
         nli("B", 0.1, 0.05, 0.85, rank=2, score=99.0)]
    out = WeightedByRetrievalAggregator(alpha=1.0).aggregate(CLAIM, v)
    assert out.aggregation_trace["pooled"]["support"] == pytest.approx((0.9 + 0.5 * 0.1) / 1.5)
    assert out.label is Label.SUPPORTED
    # scores are recorded even though unused, so the choice can be revisited with data
    assert out.aggregation_trace["retrieval_scores"] == [0.1, 99.0]


def test_alpha_zero_is_a_uniform_mean():
    v = [nli("A", 0.9, 0.05, 0.05, rank=1), nli("B", 0.1, 0.05, 0.85, rank=2)]
    out = WeightedByRetrievalAggregator(alpha=0.0).aggregate(CLAIM, v)
    assert out.aggregation_trace["pooled"]["support"] == pytest.approx(0.5)


def test_weights_are_recorded_normalised_and_sum_to_one():
    v = [nli("A", 0.5, 0.3, 0.2, rank=1), nli("B", 0.5, 0.3, 0.2, rank=2),
         nli("C", 0.5, 0.3, 0.2, rank=3)]
    trace = WeightedByRetrievalAggregator().aggregate(CLAIM, v).aggregation_trace
    assert sum(trace["weights"]) == pytest.approx(1.0, abs=1e-3)
    assert trace["ranks"] == [1, 2, 3]


def test_decisive_evidence_is_the_top_half_of_weight_mass():
    v = [nli("A", 0.5, 0.3, 0.2, rank=1), nli("B", 0.5, 0.3, 0.2, rank=2),
         nli("C", 0.5, 0.3, 0.2, rank=3)]
    trace = WeightedByRetrievalAggregator().aggregate(CLAIM, v).aggregation_trace
    # weights 1, 0.5, 0.333 -> total 1.833; rank 1 alone is 0.545 of it, > half
    assert trace["decisive_evidence_ids"] == [evidence_id("A", 0)]


# --------------------------------------------------------------------------- #
# 4. threshold + abstain
# --------------------------------------------------------------------------- #

def test_threshold_supported_above_the_floor():
    out = ThresholdWithAbstainAggregator().aggregate(CLAIM, [nli("A", 0.8, 0.1, 0.1)])
    assert out.label is Label.SUPPORTED and out.abstained is False


def test_threshold_contradicted_needs_to_beat_support_too():
    out = ThresholdWithAbstainAggregator().aggregate(CLAIM, [nli("A", 0.1, 0.85, 0.05)])
    assert out.label is Label.CONTRADICTED and out.abstained is False


def test_threshold_abstains_when_nothing_clears_the_abstention_floor():
    """s=0.3, c=0.2, both < abstain_below=0.4 -> ABSTAIN, and abstained is True."""
    out = ThresholdWithAbstainAggregator().aggregate(CLAIM, [nli("A", 0.3, 0.2, 0.5)])
    assert out.label is Label.ABSTAIN
    assert out.abstained is True
    assert "decline" in out.aggregation_trace["explanation"].lower()


def test_threshold_insufficient_sits_between_the_floors():
    """s=0.45: above abstain_below (0.4) but below support_floor (0.5). This is the
    band where the evidence WAS read and does not settle the claim -- a different
    statement from abstaining."""
    out = ThresholdWithAbstainAggregator().aggregate(CLAIM, [nli("A", 0.45, 0.1, 0.45)])
    assert out.label is Label.INSUFFICIENT
    assert out.abstained is False


def test_abstain_and_insufficient_are_distinguishable_in_the_trace():
    abstain = ThresholdWithAbstainAggregator().aggregate(CLAIM, [nli("A", 0.3, 0.2, 0.5)])
    insuff = ThresholdWithAbstainAggregator().aggregate(CLAIM, [nli("A", 0.45, 0.1, 0.45)])
    assert abstain.label is not insuff.label
    assert abstain.abstained != insuff.abstained


def test_threshold_rejects_out_of_range_floors():
    with pytest.raises(ValueError, match="must be in "):
        ThresholdWithAbstainAggregator(support_floor=1.5)


def test_threshold_records_its_floors_as_placeholders():
    trace = ThresholdWithAbstainAggregator().aggregate(CLAIM, [nli("A", 0.8, 0.1, 0.1)]).aggregation_trace
    assert trace["support_floor"] == 0.5
    assert "UNTUNED" in trace["rule"]


# --------------------------------------------------------------------------- #
# 5. majority (null baseline)
# --------------------------------------------------------------------------- #

def test_majority_counts_votes_not_magnitudes():
    """Two marginal support votes (0.34) beat one overwhelming contradiction (0.99).
    That is the point: this baseline throws away exactly what the others use."""
    v = [nli("A", 0.34, 0.33, 0.33), nli("B", 0.34, 0.33, 0.33), nli("C", 0.005, 0.99, 0.005)]
    out = MajorityAggregator().aggregate(CLAIM, v)
    assert out.label is Label.SUPPORTED
    assert out.confidence == pytest.approx(2 / 3)
    assert out.aggregation_trace["votes"] == {"Supported": 2, "Contradicted": 1, "Insufficient": 0}


def test_majority_ties_resolve_to_insufficient():
    v = [nli("A", 0.9, 0.05, 0.05), nli("B", 0.05, 0.9, 0.05)]
    out = MajorityAggregator().aggregate(CLAIM, v)
    assert out.label is Label.INSUFFICIENT
    assert out.aggregation_trace["tie"] is True


def test_a_real_aggregator_disagrees_with_majority_on_the_same_evidence():
    """If this ever stopped being true, the aggregation slot would not be interesting."""
    v = [nli("A", 0.34, 0.33, 0.33), nli("B", 0.34, 0.33, 0.33), nli("C", 0.005, 0.99, 0.005)]
    assert MajorityAggregator().aggregate(CLAIM, v).label is Label.SUPPORTED
    assert MaxEntailmentAggregator().aggregate(CLAIM, v).label is Label.CONTRADICTED


# --------------------------------------------------------------------------- #
# properties every aggregator must satisfy
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", ALL_AGGREGATORS, ids=lambda c: c.__name__)
def test_empty_evidence_is_insufficient_and_says_it_cannot_tell_why(cls):
    """Zero retrieved evidence is the retrieval-failure/genuine-insufficiency boundary.
    No aggregator may guess which one it is."""
    out = cls().aggregate(CLAIM, [])
    assert out.label is Label.INSUFFICIENT
    assert out.confidence == 0.0
    assert out.abstained is False
    assert out.per_evidence == ()
    expl = out.aggregation_trace["explanation"].lower()
    assert "retriever missed it" in expl and "does not settle" in expl


@pytest.mark.parametrize("cls", ALL_AGGREGATORS, ids=lambda c: c.__name__)
def test_similarity_only_can_never_produce_contradicted(cls):
    """A property of cosine, not a bug: it has no contradiction signal at all."""
    for cos in (-1.0, -0.5, 0.0, 0.5, 0.99):
        out = cls().aggregate(CLAIM, [sim("A", cos), sim("B", cos)])
        assert out.label is not Label.CONTRADICTED


@pytest.mark.parametrize("cls", ALL_AGGREGATORS, ids=lambda c: c.__name__)
def test_similarity_only_is_flagged_in_the_trace(cls):
    out = cls().aggregate(CLAIM, [sim("A", 0.8)])
    assert "no_contradiction_signal" in out.aggregation_trace["signal"]


@pytest.mark.parametrize("cls", ALL_AGGREGATORS, ids=lambda c: c.__name__)
def test_every_aggregator_writes_a_substantive_explanation(cls):
    """The type enforces that the key exists; this checks it is not a stub."""
    out = cls().aggregate(CLAIM, [nli("A", 0.7, 0.2, 0.1), nli("B", 0.2, 0.7, 0.1, rank=2)])
    trace = out.aggregation_trace
    assert len(trace["explanation"]) > 60
    assert len(trace["rule"]) > 30
    assert trace["n_evidence"] == 2
    assert isinstance(trace["decisive_evidence_ids"], list)


@pytest.mark.parametrize("cls", ALL_AGGREGATORS, ids=lambda c: c.__name__)
def test_confidence_is_always_a_valid_probability(cls):
    for v in ([nli("A", 1.0, 0.0, 0.0)], [nli("A", 0.0, 1.0, 0.0)],
              [nli(f"D{i}", 0.99, 0.005, 0.005) for i in range(20)]):
        out = cls().aggregate(CLAIM, v)
        assert 0.0 <= out.confidence <= 1.0


@pytest.mark.parametrize("cls", ALL_AGGREGATORS, ids=lambda c: c.__name__)
def test_per_evidence_is_carried_through_untouched(cls):
    """The Verification panel renders these; an aggregator must not drop them."""
    v = [nli("A", 0.7, 0.2, 0.1), nli("B", 0.2, 0.7, 0.1, rank=2)]
    out = cls().aggregate(CLAIM, v)
    assert out.per_evidence == tuple(v)


@pytest.mark.parametrize("cls", ALL_AGGREGATORS, ids=lambda c: c.__name__)
def test_decisive_ids_are_always_real_evidence_ids(cls):
    v = [nli("A", 0.7, 0.2, 0.1), nli("B", 0.2, 0.7, 0.1, rank=2)]
    out = cls().aggregate(CLAIM, v)
    known = {x.evidence_id for x in v}
    assert set(out.aggregation_trace["decisive_evidence_ids"]) <= known


@pytest.mark.parametrize("cls", ALL_AGGREGATORS, ids=lambda c: c.__name__)
def test_aggregators_are_pure_functions_of_their_input(cls):
    v = [nli("A", 0.7, 0.2, 0.1), nli("B", 0.2, 0.7, 0.1, rank=2)]
    agg = cls()
    first = agg.aggregate(CLAIM, v)
    assert agg.aggregate(CLAIM, v) == first
