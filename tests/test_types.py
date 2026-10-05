"""Contract tests for src/core/types.py.

These target the four places silent bugs live in this pipeline: evidence-id
mapping, label mapping, probability handling, and trace cross-references.
Tokenizer truncation and FAISS index alignment get their own tests once those
components exist (build order steps 2-3).
"""

from __future__ import annotations

import json

import pytest

from src.core.types import (
    Claim,
    ClaimVerdict,
    Evidence,
    EvidenceVerdict,
    Label,
    SourceSpan,
    Trace,
    evidence_id,
    stable_id,
)

RESPONSE = "Marie Curie was born in Warsaw. She won two Nobel Prizes."


def make_claim(text: str = "Marie Curie was born in Warsaw.", start: int = 0) -> Claim:
    return Claim.new(
        response_id="resp_1",
        text=text,
        extractor_name="test",
        source_span=SourceSpan(start, start + len(text)),
    )


def make_evidence(doc_id: str = "Marie_Curie", sent_id: int = 0, rank: int = 1) -> Evidence:
    return Evidence.new(
        doc_id=doc_id,
        sent_id=sent_id,
        text="Marie Curie was born in Warsaw, Poland.",
        score=12.5,
        retriever_name="test",
        rank=rank,
        is_gold=True,
    )


# --------------------------------------------------------------------------- #
# ids
# --------------------------------------------------------------------------- #

def test_stable_id_is_deterministic_and_separator_safe():
    assert stable_id("x", "ab", "c") == stable_id("x", "ab", "c")
    # The 0x1f separator is why these two do not collide.
    assert stable_id("x", "ab", "c") != stable_id("x", "a", "bc")


def test_evidence_id_depends_only_on_corpus_position():
    a = Evidence.new("Doc", 3, "t", score=1.0, retriever_name="bm25", rank=1)
    b = Evidence.new("Doc", 3, "t", score=99.0, retriever_name="dense", rank=7)
    assert a.id == b.id == evidence_id("Doc", 3)
    # ...which is what makes RRF de-duplication and oracle substitution sound.
    assert len({a.id, b.id}) == 1


def test_evidence_rejects_a_hand_written_id():
    with pytest.raises(ValueError, match="derived from"):
        Evidence(
            id="whatever",
            doc_id="Doc",
            sent_id=0,
            text="t",
            score=1.0,
            retriever_name="bm25",
            rank=1,
        )


def test_evidence_rank_is_one_based():
    with pytest.raises(ValueError, match="1-based"):
        Evidence.new("Doc", 0, "t", score=1.0, retriever_name="bm25", rank=0)


def test_reranked_keeps_identity_changes_position():
    ev = make_evidence()
    out = ev.reranked(rank=3, score=0.91, retriever_name="cross_encoder")
    assert out.id == ev.id and out.doc_id == ev.doc_id and out.text == ev.text
    assert (out.rank, out.score, out.retriever_name) == (3, 0.91, "cross_encoder")


def test_claim_id_changes_when_text_is_edited():
    original = make_claim()
    edited = original.edited("Marie Curie was born in Paris.")
    assert edited.id != original.id
    assert edited.extractor_meta["edited_from"] == original.id


def test_source_span_indexes_the_original_response():
    claim = make_claim()
    assert claim.source_span.text_from(RESPONSE) == claim.text


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SUPPORTS", Label.SUPPORTED),
        ("supports", Label.SUPPORTED),
        ("REFUTES", Label.CONTRADICTED),
        ("NOT ENOUGH INFO", Label.INSUFFICIENT),
        ("not  enough   info", Label.INSUFFICIENT),
        ("NEI", Label.INSUFFICIENT),
    ],
)
def test_label_parse_dataset_vocabularies(raw, expected):
    assert Label.parse(raw) is expected


@pytest.mark.parametrize("raw", ["neutral", "entailment", "contradiction", "hallucinated"])
def test_label_parse_refuses_nli_class_names(raw):
    """NLI 'neutral' is not automatically 'Insufficient'.

    That mapping is a modelling decision and must be made by an aggregator, where
    it lands in aggregation_trace and is visible in the UI.
    """
    with pytest.raises(ValueError):
        Label.parse(raw)


def test_label_serialises_as_its_string_value():
    assert json.dumps({"label": Label.SUPPORTED}) == '{"label": "Supported"}'


# --------------------------------------------------------------------------- #
# EvidenceVerdict
# --------------------------------------------------------------------------- #

def test_probabilities_must_be_all_present_or_all_absent():
    with pytest.raises(ValueError, match="all-None or all-present"):
        EvidenceVerdict("c", "e", 0.7, None, None, 0.5, "v", 1.0)


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        EvidenceVerdict("c", "e", 0.7, 0.5, 0.4, None, "v", 1.0)


def test_probabilities_outside_the_unit_interval_are_rejected():
    """Catches logits stored where probabilities were expected."""
    with pytest.raises(ValueError, match=r"p_entail must be in \[0.0, 1.0\]"):
        EvidenceVerdict("c", "e", 2.1, -0.4, 0.3, None, "v", 1.0)


def test_fp16_rounding_is_tolerated():
    v = EvidenceVerdict("c", "e", 0.7005, 0.2, 0.0995, None, "nli", 3.0)
    assert v.has_nli


def test_similarity_only_verdict_is_legal():
    v = EvidenceVerdict("c", "e", None, None, None, 0.42, "similarity", 0.5)
    assert not v.has_nli and v.similarity == 0.42


def test_empty_verdict_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        EvidenceVerdict("c", "e", None, None, None, None, "v", 1.0)


# --------------------------------------------------------------------------- #
# ClaimVerdict
# --------------------------------------------------------------------------- #

def _trace_dict() -> dict:
    return {"rule": "max_entailment", "explanation": "because", "decisive_evidence_ids": []}


def test_aggregation_trace_keys_are_mandatory():
    with pytest.raises(ValueError, match="missing required keys"):
        ClaimVerdict("c", Label.SUPPORTED, 0.9, False, (), "agg", {"rule": "x"})


def test_verdict_rejects_per_evidence_from_another_claim():
    stray = EvidenceVerdict("other_claim", "ev::D::0", None, None, None, 0.5, "sim", 1.0)
    with pytest.raises(ValueError, match="inside a ClaimVerdict"):
        ClaimVerdict("c", Label.SUPPORTED, 0.9, False, (stray,), "agg", _trace_dict())


def test_abstain_label_requires_the_flag():
    with pytest.raises(ValueError, match="requires abstained=True"):
        ClaimVerdict("c", Label.ABSTAIN, 0.1, False, (), "agg", _trace_dict())


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #

def build_trace() -> Trace:
    claim = make_claim()
    ev = make_evidence()
    pair = EvidenceVerdict(claim.id, ev.id, 0.8, 0.1, 0.1, 0.63, "nli", 12.0)
    verdict = ClaimVerdict(
        claim_id=claim.id,
        label=Label.SUPPORTED,
        confidence=0.8,
        abstained=False,
        per_evidence=(pair,),
        aggregator_name="max_entailment",
        aggregation_trace={
            "rule": "argmax_e p_entail(e)",
            "explanation": "Top entailment 0.80 from ev::Marie_Curie::0 exceeds the support floor.",
            "decisive_evidence_ids": [ev.id],
        },
    )
    return Trace(
        run_id="20260828T000000Z-abcdef",
        config_hash="deadbeef1234",
        git_sha=None,
        timestamp="2026-08-28T00:00:00+00:00",
        response_text=RESPONSE,
        claims=(claim,),
        evidence_by_claim={claim.id: (ev,)},
        verdicts=(verdict,),
        timings={"retrieve_ms": 4.2, "verify_ms": 12.0},
    )


def test_trace_round_trips_through_json():
    original = build_trace()
    restored = Trace.from_json_line(original.to_json_line())
    assert restored == original
    assert restored.verdicts[0].label is Label.SUPPORTED


def test_trace_rejects_a_verdict_scoring_unseen_evidence():
    claim = make_claim()
    ev = make_evidence()
    ghost = EvidenceVerdict(claim.id, evidence_id("Other_Doc", 4), 0.9, 0.05, 0.05, None, "nli", 1.0)
    verdict = ClaimVerdict(
        claim.id, Label.SUPPORTED, 0.9, False, (ghost,), "agg", _trace_dict()
    )
    with pytest.raises(ValueError, match="index-alignment bug"):
        Trace(
            run_id="r",
            config_hash="c",
            git_sha=None,
            timestamp="t",
            response_text=RESPONSE,
            claims=(claim,),
            evidence_by_claim={claim.id: (ev,)},
            verdicts=(verdict,),
        )


def test_trace_rejects_evidence_for_an_unknown_claim():
    claim = make_claim()
    with pytest.raises(ValueError, match="unknown claim"):
        Trace(
            run_id="r",
            config_hash="c",
            git_sha=None,
            timestamp="t",
            response_text=RESPONSE,
            claims=(claim,),
            evidence_by_claim={"clm_ghost": (make_evidence(),)},
            verdicts=(),
        )


def test_trace_rejects_an_incompatible_schema_version():
    payload = build_trace().to_dict()
    payload["schema_version"] = "9.0.0"
    with pytest.raises(ValueError, match="incompatible"):
        Trace.from_dict(payload)


def test_oracle_mode_is_recorded():
    base = build_trace().to_dict()
    base["mode"] = "oracle"
    assert Trace.from_dict(base).mode == "oracle"
    base["mode"] = "gold-ish"
    with pytest.raises(ValueError, match="mode must be"):
        Trace.from_dict(base)


def test_adversarial_trace_serialization():
    # Construct an adversarial trace
    adv_text = "Unicode \u2603\nNew line\r\n\tTabs and empty strings '' \"\" \x1f"
    claim = Claim.new("resp_adv", adv_text, "adv_extractor", None, {"adv": "\n\0"})
    ev = Evidence.new("Doc\nID", 0, adv_text, 0.0, "adv_retriever", 1, True, {"meta": {"nested": "value\n"}})
    
    # Missing optional probabilities, empty evidence list allowed
    pair = EvidenceVerdict(claim.id, ev.id, None, None, None, 0.1, "adv_ver", 0.0, 1, 0.0)
    
    verdict = ClaimVerdict(
        claim_id=claim.id,
        label=Label.ABSTAIN,
        confidence=0.0,
        abstained=True,
        per_evidence=(pair,),
        aggregator_name="adv_agg",
        aggregation_trace={
            "rule": "adv_rule\n",
            "explanation": "adv_exp",
            "decisive_evidence_ids": [ev.id],
            "nested": {"array": [1, 2, 3]}
        }
    )
    
    trace = Trace(
        run_id="run\n1",
        config_hash="c\th",
        git_sha="",
        timestamp="t",
        response_text=adv_text,
        claims=(claim,),
        evidence_by_claim={claim.id: (ev,)},
        verdicts=(verdict,),
        timings={},
        resolved_config={"adv": "config"}
    )
    
    serialized = trace.to_json_line()
    restored = Trace.from_json_line(serialized)
    
    assert restored == trace
    assert restored.claims[0].text == adv_text
    assert restored.evidence_by_claim[claim.id][0].text == adv_text

