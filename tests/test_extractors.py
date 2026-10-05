"""Contract tests for src/components/extractors.py and rerankers.py.

The extractor tests are mostly about the **span contract**: `Claim.source_span` must
index the ORIGINAL response string. Get that wrong and the frontend highlights the
wrong text with no error anywhere — which matters because the editable-claims panel is
how decomposition sensitivity gets probed by hand.

The reranker tests are about the **subset contract**: a reranker may reorder and drop,
never invent. Since `Evidence.id` is corpus identity, that check also catches a
reranker rebuilding Evidence from a different corpus copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.interfaces import (
    ContractError,
    check_claims,
    check_evidence_list,
    check_rerank_is_subset,
)
from src.core.types import Claim, Evidence, SourceSpan

RESPONSE = (
    "Marie Curie was born in Warsaw. She won two Nobel Prizes. "
    "Dr. Curie worked in Paris, i.e. at the Sorbonne."
)


@pytest.fixture(scope="module")
def splitter():
    pytest.importorskip("spacy")
    from src.components.extractors import SpacySentenceExtractor

    return SpacySentenceExtractor()


# --------------------------------------------------------------------------- #
# the span contract
# --------------------------------------------------------------------------- #

def test_every_span_indexes_the_original_response(splitter):
    """The load-bearing property. If an extractor normalises whitespace before
    splitting and reports offsets into the normalised copy, this fails."""
    claims = splitter.extract(RESPONSE)
    assert claims
    for c in claims:
        assert c.source_span is not None
        assert c.source_span.text_from(RESPONSE) == c.text


def test_claims_pass_the_pipeline_contract_check(splitter):
    check_claims(splitter.extract(RESPONSE), RESPONSE)


def test_claim_ids_are_unique(splitter):
    claims = splitter.extract(RESPONSE)
    assert len({c.id for c in claims}) == len(claims)


def test_all_claims_share_one_response_id(splitter):
    claims = splitter.extract(RESPONSE)
    assert len({c.response_id for c in claims}) == 1


def test_the_same_response_yields_the_same_claim_ids(splitter):
    """Content-addressed ids: a re-run must produce identical ids or nothing caches."""
    assert [c.id for c in splitter.extract(RESPONSE)] == [c.id for c in splitter.extract(RESPONSE)]


def test_abbreviations_do_not_split_sentences(splitter):
    """'Dr.' and 'i.e.' are why the statistical model is preferred over a rule-based
    splitter — a break here would surface downstream as a decomposition error and
    contaminate exactly what you would be trying to measure."""
    texts = [c.text for c in splitter.extract(RESPONSE)]
    assert len(texts) == 3, texts
    assert any(t.startswith("Dr. Curie") and "Sorbonne" in t for t in texts), texts


def test_leading_whitespace_does_not_shift_spans(splitter):
    padded = "   " + RESPONSE
    for c in splitter.extract(padded):
        assert c.source_span.text_from(padded) == c.text


def test_sentence_splitting_marks_itself_as_undecomposed(splitter):
    """Honest metadata: this extractor does no decomposition and says so, so a later
    comparison against an LLM decomposer is not confounded by mislabelled provenance."""
    for c in splitter.extract(RESPONSE):
        assert c.extractor_meta["decomposed"] is False


def test_a_conjunction_survives_intact(splitter):
    """The named weakness. Two facts stay welded into one claim, so the verifier is
    forced to return one label for two propositions that may differ in truth value."""
    both = "Curie was born in Warsaw and won two Nobel Prizes."
    claims = splitter.extract(both)
    assert len(claims) == 1
    assert "Warsaw" in claims[0].text and "Nobel" in claims[0].text


# --------------------------------------------------------------------------- #
# LLM extractor — cache behaviour
# --------------------------------------------------------------------------- #

@pytest.fixture()
def cache_file(tmp_path: Path) -> Path:
    from src.components.extractors import LLMClaimExtractor

    key = LLMClaimExtractor.cache_key(RESPONSE)
    payload = {
        "entries": {
            key: [
                "Marie Curie was born in Warsaw.",
                "Marie Curie won two Nobel Prizes.",
                "Marie Curie worked at the Sorbonne in Paris.",
            ]
        }
    }
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_cache_hit_returns_the_written_decomposition(cache_file: Path):
    pytest.importorskip("spacy")
    from src.components.extractors import LLMClaimExtractor

    claims = LLMClaimExtractor(cache_path=str(cache_file)).extract(RESPONSE)
    assert [c.text for c in claims] == [
        "Marie Curie was born in Warsaw.",
        "Marie Curie won two Nobel Prizes.",
        "Marie Curie worked at the Sorbonne in Paris.",
    ]
    assert all(c.extractor_meta["cache_hit"] for c in claims)
    assert all(c.extractor_meta["decomposed"] for c in claims)


def test_cache_miss_falls_back_and_says_so(cache_file: Path):
    pytest.importorskip("spacy")
    from src.components.extractors import LLMClaimExtractor

    claims = LLMClaimExtractor(cache_path=str(cache_file)).extract("An unseen response entirely.")
    assert claims
    assert all(c.extractor_meta["cache_hit"] is False for c in claims)
    assert all(c.extractor_meta["fallback"] == "sentence_split" for c in claims)


def test_on_miss_error_refuses_to_silently_mix_extractors(cache_file: Path):
    """In an experiment, a silent fallback puts two different extractors in one table."""
    from src.components.extractors import LLMClaimExtractor

    ex = LLMClaimExtractor(cache_path=str(cache_file), on_miss="error")
    with pytest.raises(KeyError, match="no cached decomposition"):
        ex.extract("An unseen response entirely.")


def test_debug_call_directs_users_to_gated_s1_provider_workflow(cache_file: Path):
    from src.components.extractors import LLMClaimExtractor

    ex = LLMClaimExtractor(cache_path=str(cache_file), on_miss="call")
    with pytest.raises(NotImplementedError, match="scripts.generate_s1_llm_cache after human gold freeze"):
        ex.extract("An unseen response entirely.")


def test_cache_key_is_whitespace_sensitive():
    """Two responses differing only in whitespace are different inputs to a decomposer."""
    from src.components.extractors import LLMClaimExtractor

    assert LLMClaimExtractor.cache_key("a b") != LLMClaimExtractor.cache_key("a  b")


def test_duplicate_claims_in_the_cache_are_dropped(tmp_path: Path):
    """LLM decomposers repeat themselves; a duplicate would otherwise collide on id."""
    pytest.importorskip("spacy")
    from src.components.extractors import LLMClaimExtractor

    key = LLMClaimExtractor.cache_key(RESPONSE)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"entries": {key: ["A claim.", "A claim.", "Another."]}}), encoding="utf-8")
    claims = LLMClaimExtractor(cache_path=str(p)).extract(RESPONSE)
    assert [c.text for c in claims] == ["A claim.", "Another."]


def test_bare_mapping_cache_file_is_accepted(tmp_path: Path):
    """A hand-written cache should not need boilerplate."""
    pytest.importorskip("spacy")
    from src.components.extractors import LLMClaimExtractor

    key = LLMClaimExtractor.cache_key(RESPONSE)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({key: ["One claim."]}), encoding="utf-8")
    assert [c.text for c in LLMClaimExtractor(cache_path=str(p)).extract(RESPONSE)] == ["One claim."]


def test_rewritten_claims_are_marked_as_having_inexact_spans(cache_file: Path):
    """A decomposed claim usually is not a substring of the response. The span is a
    best-effort guess and must be labelled as one."""
    pytest.importorskip("spacy")
    from src.components.extractors import LLMClaimExtractor

    claims = LLMClaimExtractor(cache_path=str(cache_file)).extract(RESPONSE)
    rewritten = [c for c in claims if not c.extractor_meta["span_is_exact"]]
    assert rewritten, "expected at least one rewritten claim in this fixture"


def test_missing_cache_file_is_not_an_error(tmp_path: Path):
    pytest.importorskip("spacy")
    from src.components.extractors import LLMClaimExtractor

    ex = LLMClaimExtractor(cache_path=str(tmp_path / "absent.json"))
    assert ex.cache == {}
    assert ex.extract(RESPONSE)          # falls back to splitting


def test_bad_on_miss_value_is_rejected():
    from src.components.extractors import LLMClaimExtractor

    with pytest.raises(ValueError, match="on_miss must be"):
        LLMClaimExtractor(on_miss="whatever")


# --------------------------------------------------------------------------- #
# rerankers
# --------------------------------------------------------------------------- #

def ev(doc: str, rank: int, score: float = 1.0) -> Evidence:
    return Evidence.new(doc, 0, f"text of {doc}", score, "bm25", rank)


def test_noop_renumbers_ranks_after_truncation():
    """Truncating to k without renumbering is a no-op that violates the rank contract
    and quietly corrupts anything reading rank as position."""
    from src.components.rerankers import NoOpReranker

    out = NoOpReranker().rerank(Claim.new("r", "c", "t"), [ev(f"D{i}", i) for i in range(1, 6)], k=3)
    assert [e.rank for e in out] == [1, 2, 3]
    check_evidence_list(out, k=3, stage="noop")


def test_noop_preserves_identity_and_order():
    from src.components.rerankers import NoOpReranker

    inp = [ev(f"D{i}", i) for i in range(1, 4)]
    out = NoOpReranker().rerank(Claim.new("r", "c", "t"), inp, k=3)
    assert [e.id for e in out] == [e.id for e in inp]
    assert [e.text for e in out] == [e.text for e in inp]
    check_rerank_is_subset(inp, out)


def test_noop_on_empty_input():
    from src.components.rerankers import NoOpReranker

    assert NoOpReranker().rerank(Claim.new("r", "c", "t"), [], k=5) == []


def test_a_claim_with_wrong_span_is_caught():
    c = Claim.new("r1", "Marie Curie was born in Paris", "ext", SourceSpan(0, 5))
    with pytest.raises(ContractError, match="does not match the claim text"):
        check_claims([c], RESPONSE)


def test_approximate_span_is_allowed():
    c = Claim.new(
        "r1",
        "approximate match",
        "ext",
        SourceSpan(0, 5),
        extractor_meta={"span_is_exact": False}
    )
    check_claims([c], RESPONSE)


def test_subset_check_catches_invented_evidence():
    before = [ev("A", 1)]
    after = [ev("B", 1)]
    with pytest.raises(ContractError, match="introduced evidence not in its input"):
        check_rerank_is_subset(before, after)


@pytest.mark.slow
def test_cross_encoder_obeys_the_subset_contract_and_records_movement():
    pytest.importorskip("sentence_transformers")
    from src.components.rerankers import CrossEncoderReranker

    claim = Claim.new("r", "Marie Curie was born in Warsaw.", "t")
    inp = [
        Evidence.new("Paris", 0, "Paris is the capital of France.", 5.0, "bm25", 1),
        Evidence.new("Curie", 0, "Marie Curie was born in Warsaw, Poland.", 4.0, "bm25", 2),
        Evidence.new("Rome", 0, "Rome is a city in Italy.", 3.0, "bm25", 3),
    ]
    out = CrossEncoderReranker().rerank(claim, inp, k=3)
    check_rerank_is_subset(inp, out)
    check_evidence_list(out, k=3, stage="cross_encoder")
    # the relevant sentence should be promoted off rank 2
    assert out[0].doc_id == "Curie", [(e.doc_id, e.score) for e in out]
    assert out[0].retriever_meta["rank_before_rerank"] == 2
    assert out[0].retriever_meta["rank_delta"] == 1
    assert out[0].text == inp[1].text          # text never edited


@pytest.mark.slow
def test_cross_encoder_on_empty_input():
    pytest.importorskip("sentence_transformers")
    from src.components.rerankers import CrossEncoderReranker

    assert CrossEncoderReranker().rerank(Claim.new("r", "c", "t"), [], k=5) == []
