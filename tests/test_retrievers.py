"""Contract tests for src/components/retrievers.py and src/data/metrics.py.

The RRF tests use fake arms with hand-computed expected scores, so fusion is verified
arithmetically rather than "it returned something plausible". The BM25 tests need
rank_bm25; the dense tests need faiss + sentence-transformers + a model download, and
are marked so a skip is visible rather than silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core import registry
from src.core.interfaces import Retriever, check_evidence_list
from src.core.types import Claim, Evidence, evidence_id
from src.data.corpus import Corpus
from src.data.metrics import gold_ranks, mark_gold, recall_at_k, reciprocal_rank

MINI = Path(__file__).resolve().parents[1] / "data" / "debug" / "mini"
MINI_CORPUS = MINI / "corpus.jsonl"

needs_mini = pytest.mark.skipif(
    not MINI_CORPUS.is_file(), reason="run: python -m scripts.build_mini_corpus"
)


def claim(text: str) -> Claim:
    return Claim.new(response_id="r", text=text, extractor_name="test")


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

@needs_mini
def test_bm25_obeys_the_retriever_contract():
    pytest.importorskip("rank_bm25")
    from src.components.retrievers import BM25Retriever

    r = BM25Retriever(corpus_path=str(MINI_CORPUS))
    out = r.retrieve(claim("Marie Curie was born in Warsaw."), k=5)
    check_evidence_list(out, k=5, stage="bm25")   # ranks 1..5, unique ids, len <= k
    assert len(out) == 5
    assert all(e.retriever_name == "bm25" for e in out)


@needs_mini
def test_bm25_finds_the_gold_sentence_for_a_high_overlap_claim():
    """Not a quality measurement -- a wiring check. If this misses, something is wrong
    with tokenisation or with row->sentence mapping, not with BM25."""
    pytest.importorskip("rank_bm25")
    from src.components.retrievers import BM25Retriever

    r = BM25Retriever(corpus_path=str(MINI_CORPUS))
    out = r.retrieve(claim("Marie Curie was born in Warsaw."), k=5)
    assert evidence_id("Marie_Curie", 1) in {e.id for e in out}


@needs_mini
def test_bm25_evidence_text_matches_the_corpus_at_that_key():
    """The alignment check: retrieved text must equal corpus[doc_id][sent_id]."""
    pytest.importorskip("rank_bm25")
    from src.components.retrievers import BM25Retriever

    corpus = Corpus.from_jsonl(MINI_CORPUS)
    r = BM25Retriever(corpus_path=str(MINI_CORPUS))
    for e in r.retrieve(claim("radium was discovered in 1898"), k=10):
        assert e.text == corpus.get(e.doc_id, e.sent_id).text


@needs_mini
def test_bm25_is_deterministic():
    pytest.importorskip("rank_bm25")
    from src.components.retrievers import BM25Retriever

    r = BM25Retriever(corpus_path=str(MINI_CORPUS))
    a = r.retrieve(claim("polonium atomic number"), k=8)
    b = r.retrieve(claim("polonium atomic number"), k=8)
    assert [e.id for e in a] == [e.id for e in b]


def test_bm25_tokenizer_keeps_digits():
    """Date and numeric claims are a named failure slice; dropping digits would hide them."""
    pytest.importorskip("rank_bm25")
    from src.components.retrievers import bm25_tokenize

    assert bm25_tokenize("Discovered in 1898, at 84 degrees!") == [
        "discovered", "in", "1898", "at", "84", "degrees",
    ]


# --------------------------------------------------------------------------- #
# RRF, with fake arms and hand-computed expectations
# --------------------------------------------------------------------------- #

def _ev(doc: str, rank: int, score: float, name: str, meta_key: str) -> Evidence:
    return Evidence.new(
        doc_id=doc, sent_id=0, text=f"text of {doc}", score=score,
        retriever_name=name, rank=rank, retriever_meta={meta_key: score, f"{name}_rank": rank},
    )


class _ArmA(Retriever):
    """Returns docs D1, D2, D3 at ranks 1, 2, 3."""

    def retrieve(self, c: Claim, k: int) -> list[Evidence]:
        return [_ev(f"D{i}", i, 10.0 - i, "bm25", "bm25_score") for i in (1, 2, 3)][:k]


class _ArmB(Retriever):
    """Returns docs D3, D4 at ranks 1, 2 -- D3 overlaps with arm A at rank 3."""

    def retrieve(self, c: Claim, k: int) -> list[Evidence]:
        return [_ev("D3", 1, 0.9, "dense", "dense_score"), _ev("D4", 2, 0.8, "dense", "dense_score")][:k]


@pytest.fixture()
def hybrid_with_fake_arms():
    saved = {kind: dict(reg) for kind, reg in registry._REGISTRY.items()}
    registry.register("retriever", "arm_a")(_ArmA)
    registry.register("retriever", "arm_b")(_ArmB)
    yield
    for kind, reg in saved.items():
        registry._REGISTRY[kind] = reg


def build_hybrid(rrf_k: int = 60):
    from src.components.retrievers import HybridRetriever

    return HybridRetriever(
        corpus_path=str(MINI_CORPUS),
        arms=[{"name": "arm_a"}, {"name": "arm_b"}],
        rrf_k=rrf_k,
    )


def test_rrf_scores_match_the_formula(hybrid_with_fake_arms):
    """RRF(d) = sum over arms of 1/(K + rank). Computed by hand at K=60:
        D3 = 1/63 + 1/61 = 0.031268...   (in both arms -> wins)
        D1 = 1/61        = 0.016393...
        D2 = 1/62        = 0.016129...
        D4 = 1/62        = 0.016129...
    """
    out = build_hybrid(rrf_k=60).retrieve(claim("anything"), k=4)
    got = {e.doc_id: e.score for e in out}
    assert got["D3"] == pytest.approx(1 / 63 + 1 / 61)
    assert got["D1"] == pytest.approx(1 / 61)
    assert got["D2"] == pytest.approx(1 / 62)
    assert got["D4"] == pytest.approx(1 / 62)


def test_rrf_ranks_the_doc_both_arms_found_first(hybrid_with_fake_arms):
    """The whole point of fusion: agreement beats one arm's confident top hit."""
    out = build_hybrid().retrieve(claim("anything"), k=4)
    assert out[0].doc_id == "D3"
    assert out[0].rank == 1


def test_rrf_deduplicates_a_sentence_seen_by_both_arms(hybrid_with_fake_arms):
    out = build_hybrid().retrieve(claim("anything"), k=10)
    ids = [e.id for e in out]
    assert len(ids) == len(set(ids)) == 4  # D1..D4, D3 not doubled


def test_fused_evidence_carries_both_arms_scores(hybrid_with_fake_arms):
    """The Retrieval panel shows BM25 score, dense score and fused rank separately."""
    out = build_hybrid().retrieve(claim("anything"), k=4)
    d3 = next(e for e in out if e.doc_id == "D3")
    assert d3.retriever_meta["bm25_score"] == pytest.approx(7.0)
    assert d3.retriever_meta["dense_score"] == pytest.approx(0.9)
    assert d3.retriever_meta["bm25_rank"] == 3
    assert d3.retriever_meta["dense_rank"] == 1
    assert d3.retriever_meta["n_arms_hit"] == 2
    assert d3.retriever_meta["fused_rank"] == 1


def test_ties_are_broken_deterministically(hybrid_with_fake_arms):
    """D2 and D4 both score 1/62; order must not depend on dict iteration."""
    first = [e.id for e in build_hybrid().retrieve(claim("x"), k=4)]
    for _ in range(5):
        assert [e.id for e in build_hybrid().retrieve(claim("x"), k=4)] == first


def test_hybrid_obeys_the_retriever_contract(hybrid_with_fake_arms):
    out = build_hybrid().retrieve(claim("anything"), k=3)
    check_evidence_list(out, k=3, stage="hybrid")


def test_smaller_rrf_k_sharpens_the_advantage_of_rank_one(hybrid_with_fake_arms):
    """K controls how much rank 1 beats rank 20; at K=1 the gap is large."""
    tight = {e.doc_id: e.score for e in build_hybrid(rrf_k=1).retrieve(claim("x"), k=4)}
    loose = {e.doc_id: e.score for e in build_hybrid(rrf_k=60).retrieve(claim("x"), k=4)}
    assert tight["D1"] / tight["D2"] > loose["D1"] / loose["D2"]


def test_rrf_k_must_be_positive(hybrid_with_fake_arms):
    from src.components.retrievers import HybridRetriever

    with pytest.raises(ValueError, match="rrf_k must be >= 1"):
        HybridRetriever(corpus_path=str(MINI_CORPUS), arms=[{"name": "arm_a"}], rrf_k=0)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _retrieved(*doc_ids: str) -> list[Evidence]:
    return [
        Evidence.new(d, 0, "t", 1.0, "test", rank=i) for i, d in enumerate(doc_ids, start=1)
    ]


def test_recall_is_none_when_the_example_has_no_gold_evidence():
    """NEI examples have empty gold by construction. 1.0 flatters, 0.0 punishes."""
    assert recall_at_k(_retrieved("A", "B"), gold_ids=[]) is None
    assert reciprocal_rank(_retrieved("A", "B"), gold_ids=[]) is None


def test_recall_counts_gold_found_within_k():
    ev = _retrieved("A", "B", "C")
    gold = [evidence_id("B", 0), evidence_id("Z", 0)]
    assert recall_at_k(ev, gold) == pytest.approx(0.5)
    assert recall_at_k(ev, gold, k=1) == pytest.approx(0.0)


def test_gold_ranks_distinguishes_missed_from_low_ranked():
    """'Retrieved at rank 34' and 'never returned' are different diagnoses."""
    ev = _retrieved("A", "B", "C")
    ranks = gold_ranks(ev, [evidence_id("C", 0), evidence_id("Z", 0)])
    assert ranks[evidence_id("C", 0)] == 3
    assert ranks[evidence_id("Z", 0)] is None


def test_reciprocal_rank_is_zero_when_gold_exists_but_was_missed():
    assert reciprocal_rank(_retrieved("A"), [evidence_id("Z", 0)]) == 0.0


def test_mark_gold_sets_a_real_boolean_not_none():
    ev = _retrieved("A", "B")
    marked = mark_gold(ev, [evidence_id("A", 0)])
    assert marked[0].is_gold is True
    assert marked[1].is_gold is False   # annotated-not-gold, not "unknown"
    assert ev[0].is_gold is None        # the input is untouched


def test_mark_gold_preserves_identity_and_rank():
    ev = _retrieved("A", "B")
    marked = mark_gold(ev, [evidence_id("A", 0)])
    assert [e.id for e in marked] == [e.id for e in ev]
    assert [e.rank for e in marked] == [1, 2]


# --------------------------------------------------------------------------- #
# dense (needs faiss + sentence-transformers + a model download)
# --------------------------------------------------------------------------- #

@needs_mini
@pytest.mark.slow
def test_dense_index_is_aligned_and_refuses_a_stale_manifest(tmp_path: Path):
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")
    from src.components.retrievers import DenseRetriever

    r = DenseRetriever(corpus_path=str(MINI_CORPUS), index_dir=str(tmp_path))
    corpus = Corpus.from_jsonl(MINI_CORPUS)
    assert r.index.ntotal == len(corpus)

    out = r.retrieve(claim("Where was Marie Curie born?"), k=5)
    check_evidence_list(out, k=5, stage="dense")
    for e in out:
        assert e.text == corpus.get(e.doc_id, e.sent_id).text   # row -> text alignment
        assert -1.0 <= e.score <= 1.0                            # normalised -> cosine

    # Corrupt the manifest's fingerprint; loading must refuse rather than mis-map.
    import json as _json

    manifest = next(tmp_path.glob("*.manifest.json"))
    payload = _json.loads(manifest.read_text(encoding="utf-8"))
    payload["corpus_fingerprint"] = "deadbeefdeadbeef"
    manifest.write_text(_json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale FAISS index"):
        DenseRetriever(corpus_path=str(MINI_CORPUS), index_dir=str(tmp_path))
