"""Contract tests for src/data/corpus.py and src/data/examples.py.

This is the index-alignment surface. Every test here exists because getting it wrong
produces confidently wrong evidence text rather than a crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.types import Label
from src.data.corpus import Corpus, Document
from src.data.examples import Example, load_examples, save_examples, validate_against

MINI = Path(__file__).resolve().parents[1] / "data" / "debug" / "mini"


def toy() -> Corpus:
    return Corpus(
        [
            Document("A", "Doc A", ("a0", "a1", "a2")),
            Document("B", "Doc B", ("b0", "b1")),
        ]
    )


# --------------------------------------------------------------------------- #
# addressing
# --------------------------------------------------------------------------- #

def test_row_and_key_are_exact_inverses_for_every_row():
    """The single most important property in this module."""
    corpus = toy()
    for row in range(len(corpus)):
        sent = corpus.at(row)
        assert corpus.row(sent.doc_id, sent.sent_id) == row


def test_row_order_is_document_order_then_sentence_order():
    corpus = toy()
    assert [(s.doc_id, s.sent_id) for s in corpus] == [
        ("A", 0), ("A", 1), ("A", 2), ("B", 0), ("B", 1),
    ]


def test_texts_are_in_row_order():
    """BM25 and the encoder both consume this; a mismatch mis-aligns every score."""
    corpus = toy()
    assert corpus.texts() == [corpus.at(i).text for i in range(len(corpus))]


def test_missing_key_raises_rather_than_returning_a_sentinel():
    with pytest.raises(KeyError, match="not in this corpus"):
        toy().row("A", 99)


def test_duplicate_doc_id_is_rejected():
    with pytest.raises(ValueError, match="duplicate doc_id"):
        Corpus([Document("A", "x", ("s",)), Document("A", "y", ("t",))])


# --------------------------------------------------------------------------- #
# fingerprint
# --------------------------------------------------------------------------- #

def test_fingerprint_is_stable_across_identical_builds():
    assert toy().fingerprint() == toy().fingerprint()


def test_fingerprint_changes_when_a_sentence_is_inserted():
    """An index built before the insert must refuse to load after it."""
    changed = Corpus(
        [Document("A", "Doc A", ("a0", "NEW", "a1", "a2")), Document("B", "Doc B", ("b0", "b1"))]
    )
    assert changed.fingerprint() != toy().fingerprint()


def test_fingerprint_changes_when_documents_are_reordered():
    reordered = Corpus(
        [Document("B", "Doc B", ("b0", "b1")), Document("A", "Doc A", ("a0", "a1", "a2"))]
    )
    assert reordered.fingerprint() != toy().fingerprint()


def test_fingerprint_ignores_sentence_wording():
    """It covers addressing, not content -- row->key alignment is what indexes need."""
    reworded = Corpus(
        [Document("A", "Doc A", ("TOTALLY DIFFERENT", "a1", "a2")), Document("B", "Doc B", ("b0", "b1"))]
    )
    assert reworded.fingerprint() == toy().fingerprint()
    assert reworded.content_fingerprint() != toy().content_fingerprint()


def test_content_fingerprint_is_stable_across_identical_builds():
    assert toy().content_fingerprint() == toy().content_fingerprint()


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def test_jsonl_round_trip_preserves_addressing(tmp_path: Path):
    original = toy()
    restored = Corpus.from_jsonl(original.to_jsonl(tmp_path / "c.jsonl"))
    assert restored.fingerprint() == original.fingerprint()
    assert restored.texts() == original.texts()


def test_missing_corpus_file_names_the_builder(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="build_debug_corpus"):
        Corpus.from_jsonl(tmp_path / "nope.jsonl")


# --------------------------------------------------------------------------- #
# examples
# --------------------------------------------------------------------------- #

def test_gold_evidence_ids_match_the_evidence_id_scheme():
    ex = Example("e", "t", Label.SUPPORTED, (("A", 1),), "toy")
    assert ex.gold_evidence_ids == frozenset({"ev::A::1"})


def test_validate_against_catches_a_dangling_gold_key():
    ex = Example("e", "t", Label.SUPPORTED, (("A", 99),), "toy")
    report = validate_against([ex], toy())
    assert not report["ok"] and report["n_dangling_gold_keys"] == 1


def test_validate_against_accepts_an_nei_example_with_no_evidence():
    ex = Example("e", "t", Label.INSUFFICIENT, (), "toy")
    report = validate_against([ex], toy())
    assert report["ok"] and report["n_with_gold_evidence"] == 0


def test_example_round_trip(tmp_path: Path):
    original = [
        Example("e1", "claim one", Label.SUPPORTED, (("A", 0),), "toy", {"kind": "x"}),
        Example("e2", "claim two", Label.INSUFFICIENT, (), "toy"),
    ]
    assert load_examples(save_examples(original, tmp_path / "e.jsonl")) == original


# --------------------------------------------------------------------------- #
# the checked-in mini corpus
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not (MINI / "corpus.jsonl").is_file(), reason="run scripts.build_mini_corpus")
def test_mini_corpus_gold_keys_all_resolve():
    """If this fails, every oracle comparison on the mini corpus is against a broken oracle."""
    corpus = Corpus.from_jsonl(MINI / "corpus.jsonl")
    report = validate_against(load_examples(MINI / "examples.jsonl"), corpus)
    assert report["ok"], report["dangling"]
    assert report["n_examples"] == 14


@pytest.mark.skipif(not (MINI / "corpus.jsonl").is_file(), reason="run scripts.build_mini_corpus")
def test_mini_corpus_is_the_size_we_think_it_is():
    corpus = Corpus.from_jsonl(MINI / "corpus.jsonl")
    assert corpus.n_docs == 40
    assert len(corpus) == 115
