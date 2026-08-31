"""Tests for FEVER `lines` parsing — the one alignment bug nothing else can catch.

`Corpus.fingerprint` detects a corpus that changed shape. `validate_against` detects a
gold key that does not resolve. **Neither detects a gold key that resolves to the
WRONG sentence**, which is exactly what happens if empty sentences are filtered out or
if parsing enumerates positions instead of reading FEVER's declared indices. A shifted
index still resolves, still reads plausibly, and turns every downstream verdict into
noise that looks like a bad NLI model.

So these tests check the parse directly, against hand-written fixtures whose correct
answer is obvious by inspection.
"""

from __future__ import annotations

from scripts.build_debug_corpus import ParseStats, evidence_groups, parse_lines_field, sample_claims

# --------------------------------------------------------------------------- #
# the alignment invariant
# --------------------------------------------------------------------------- #

def test_empty_sentences_keep_their_index():
    """The bug this whole module exists to prevent.

    Sentence 1 is empty. If it were dropped, "Third." would move from index 2 to 1,
    and gold evidence (page, 2) would silently resolve to the wrong sentence.
    """
    raw = "0\tFirst.\n1\t\n2\tThird."
    assert parse_lines_field(raw) == ("First.", "", "Third.")


def test_declared_index_wins_over_file_position():
    """Records arrive out of order; index 0 must still be 'Zero.'."""
    raw = "2\tTwo.\n0\tZero.\n1\tOne."
    assert parse_lines_field(raw) == ("Zero.", "One.", "Two.")


def test_gaps_are_padded_not_closed():
    """Index 1 is absent entirely. Closing the gap would shift index 2 to 1."""
    raw = "0\tZero.\n2\tTwo."
    out = parse_lines_field(raw)
    assert out == ("Zero.", "", "Two.")
    assert out[2] == "Two."


def test_hyperlink_annotations_are_stripped_but_the_sentence_is_not():
    """FEVER puts entity annotations in trailing tab fields; text is field 1."""
    raw = "0\tMarie Curie was a physicist.\tMarie Curie\tMarie_Curie\tphysicist\tPhysicist"
    assert parse_lines_field(raw) == ("Marie Curie was a physicist.",)


def test_sentence_containing_no_annotations_survives():
    raw = "0\tA plain sentence with no annotations."
    assert parse_lines_field(raw) == ("A plain sentence with no annotations.",)


def test_record_with_no_text_field_becomes_empty_not_dropped():
    raw = "0\tFirst.\n1\n2\tThird."
    assert parse_lines_field(raw) == ("First.", "", "Third.")


def test_unindexed_continuation_fragment_cannot_shift_anything():
    """Real occurrence in the dump. Dropping it is only safe because we index by
    declared id -- with positional enumeration it would corrupt everything after it."""
    stats = ParseStats()
    raw = "0\tFirst.\nstray continuation text\n1\tSecond."
    assert parse_lines_field(raw, stats) == ("First.", "Second.")
    assert stats.non_integer_index == 1


def test_empty_lines_field_yields_no_sentences():
    stats = ParseStats()
    assert parse_lines_field("", stats) == ()
    assert stats.empty_lines_field == 1


def test_trailing_empty_record_is_preserved():
    """A page ending in an empty sentence still has that index."""
    assert parse_lines_field("0\tOnly.\n1\t") == ("Only.", "")


def test_stats_count_what_they_claim():
    stats = ParseStats()
    parse_lines_field("0\tA.\n2\tC.\n3\t", stats)
    assert stats.total_sentences == 4      # indices 0..3
    assert stats.padded_gaps == 1          # index 1 was absent
    assert stats.empty_sentences == 2      # the pad, and index 3


# --------------------------------------------------------------------------- #
# FEVER claim records
# --------------------------------------------------------------------------- #

def test_nei_rows_yield_no_evidence_groups():
    row = {"label": "NOT ENOUGH INFO", "evidence": [[[108548, None, None, None]]]}
    assert evidence_groups(row) == []


def test_each_group_is_a_complete_alternative_evidence_set():
    """Two single-key groups = two independent ways to verify. Not one 2-hop set."""
    row = {
        "evidence": [
            [[1, 2, "Telemundo", 0]],
            [[1, 3, "Telemundo", 1]],
        ]
    }
    assert evidence_groups(row) == [[("Telemundo", 0)], [("Telemundo", 1)]]


def test_a_multi_key_group_is_multi_hop():
    row = {"evidence": [[[1, 2, "Telemundo", 4], [1, 2, "Hispanic_and_Latino_Americans", 0]]]}
    groups = evidence_groups(row)
    assert len(groups) == 1 and len(groups[0]) == 2


def test_partially_null_group_keeps_only_real_keys():
    row = {"evidence": [[[1, 2, "Page", 3], [1, 2, None, None]]]}
    assert evidence_groups(row) == [[("Page", 3)]]


def test_sample_claims_is_stratified_and_deterministic():
    import random

    rows = (
        [{"id": i, "label": "SUPPORTS"} for i in range(100)]
        + [{"id": 100 + i, "label": "REFUTES"} for i in range(100)]
        + [{"id": 200 + i, "label": "NOT ENOUGH INFO"} for i in range(100)]
    )
    a = sample_claims(rows, 30, random.Random(1337))
    b = sample_claims(rows, 30, random.Random(1337))
    assert [r["id"] for r in a] == [r["id"] for r in b]        # seeded
    counts = {}
    for r in a:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    assert counts == {"SUPPORTS": 10, "REFUTES": 10, "NOT ENOUGH INFO": 10}


def test_sample_claims_does_not_depend_on_input_order():
    """Row order in the file must not change which claims get picked."""
    import random

    rows = [{"id": i, "label": "SUPPORTS"} for i in range(50)]
    shuffled = list(reversed(rows))
    a = sample_claims(rows, 9, random.Random(7))
    b = sample_claims(shuffled, 9, random.Random(7))
    assert [r["id"] for r in a] == [r["id"] for r in b]


# --------------------------------------------------------------------------- #
# end-to-end on a realistic fixture
# --------------------------------------------------------------------------- #

FIXTURE = (
    "0\tMarie Curie was a Polish physicist.\tMarie Curie\tMarie_Curie\n"
    "1\t\n"
    "2\tShe won the Nobel Prize in 1903 and 1911.\tNobel Prize\tNobel_Prize\n"
    "3\t\n"
    "4\tShe died in 1934."
)


def test_realistic_page_parses_to_the_right_shape():
    out = parse_lines_field(FIXTURE)
    assert len(out) == 5
    assert out[0].startswith("Marie Curie was a Polish")
    assert out[1] == ""
    assert out[2].startswith("She won the Nobel Prize")
    assert out[3] == ""
    assert out[4] == "She died in 1934."


def test_gold_key_four_resolves_to_the_fifth_sentence_not_the_third():
    """The concrete failure: with empties filtered, index 4 would not exist at all,
    and index 2 would be 'She died in 1934.'"""
    out = parse_lines_field(FIXTURE)
    assert out[4] == "She died in 1934."
    assert out[2] != "She died in 1934."


def test_parsed_page_is_addressable_through_corpus():
    """The parse feeds Corpus, and Corpus's row<->key mapping must agree with it."""
    from src.data.corpus import Corpus, Document

    corpus = Corpus([Document("Marie_Curie", "Marie Curie", parse_lines_field(FIXTURE))])
    assert corpus.get("Marie_Curie", 4).text == "She died in 1934."
    assert corpus.at(corpus.row("Marie_Curie", 2)).text.startswith("She won the Nobel")


# --------------------------------------------------------------------------- #
# zip member filtering
# --------------------------------------------------------------------------- #

def test_macos_resource_forks_are_excluded():
    """wiki-pages.zip ships an AppleDouble fork beside every data file. They end in
    .jsonl, they are binary, and '._' sorts BEFORE 'w' -- so a naive endswith filter
    reads one first and dies with JSONDecodeError at char 0. 218 members, 109 real."""
    from scripts.build_debug_corpus import is_real_wiki_member

    assert is_real_wiki_member("wiki-pages/wiki-001.jsonl")
    assert not is_real_wiki_member("__MACOSX/wiki-pages/._wiki-001.jsonl")
    assert not is_real_wiki_member("wiki-pages/._wiki-001.jsonl")
    assert not is_real_wiki_member("._wiki-001.jsonl")
    assert not is_real_wiki_member("wiki-pages/README.txt")


def test_real_members_sort_ahead_of_nothing_surprising():
    from scripts.build_debug_corpus import is_real_wiki_member

    names = ["__MACOSX/wiki-pages/._wiki-001.jsonl", "wiki-pages/wiki-002.jsonl",
             "wiki-pages/wiki-001.jsonl"]
    assert sorted(n for n in names if is_real_wiki_member(n)) == [
        "wiki-pages/wiki-001.jsonl", "wiki-pages/wiki-002.jsonl",
    ]


def test_sample_claims_returns_exactly_n_distributing_the_remainder():
    """200 over 3 labels must give 200 (67/67/66), not 198."""
    import random
    from collections import Counter

    rows = [
        {"id": i, "label": label}
        for i, (label, _) in enumerate(
            (L, j) for L in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO") for j in range(500)
        )
    ]
    got = sample_claims(rows, 200, random.Random(1337))
    assert len(got) == 200
    assert sorted(Counter(r["label"] for r in got).values()) == [66, 67, 67]


# --------------------------------------------------------------------------- #
# Unicode normalisation — a MEASURED bug in FEVER, not a hypothetical
# --------------------------------------------------------------------------- #

def test_dev_and_wiki_page_titles_need_nfc_normalisation():
    """The two FEVER files disagree about normalisation for the same title.

    Measured over all 2,892 dev gold pages: 32 are non-NFC — 1.1% overall, but
    72.7% of the 44 non-ASCII gold pages. The aggregate hides a near-total loss on
    an identifiable slice.
    """
    from scripts.build_debug_corpus import normalize_page_id

    dev_form = "Cléopâtre"      # NFD, as it appears in shared_task_dev.jsonl
    wiki_form = "Cl\xe9op\xe2tre"           # NFC, as it appears in wiki-pages.zip

    assert dev_form != wiki_form                                  # the bug
    assert normalize_page_id(dev_form) == normalize_page_id(wiki_form)   # the fix
    assert normalize_page_id(dev_form) == wiki_form               # NFC is the target form


def test_normalisation_leaves_ascii_titles_untouched():
    from scripts.build_debug_corpus import normalize_page_id

    for title in ("Marie_Curie", "Soul_Food_-LRB-film-RRB-", "1986_NBA_Finals"):
        assert normalize_page_id(title) == title


def test_evidence_groups_normalise_page_ids():
    """The dev-side boundary. Without this, gold keys never match corpus doc_ids."""
    row = {"evidence": [[[1, 2, "Cléopâtre", 3]]]}
    assert evidence_groups(row) == [[("Cl\xe9op\xe2tre", 3)]]


def test_normalisation_is_idempotent():
    from scripts.build_debug_corpus import normalize_page_id

    once = normalize_page_id("Björk")
    assert normalize_page_id(once) == once == "Bj\xf6rk"
