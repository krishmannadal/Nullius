"""Contract tests for src/pipeline.py and src/core/trace_io.py.

Runs entirely on stub components — no models, no downloads, milliseconds. The point is
the wiring, not the quality: that stages are called in order, that contracts are checked
at every boundary, that the oracle substitutes evidence and nothing else, and that a
run directory is self-describing enough to be read six weeks later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core import registry
from src.core.interfaces import (
    Aggregator,
    ClaimExtractor,
    ContractError,
    Reranker,
    Retriever,
    Verifier,
)
from src.core.trace_io import (
    TraceWriter,
    load_failure_cases,
    read_traces,
    run_dir_for,
    save_failure_case,
    write_run,
)
from src.core.types import (
    Claim,
    ClaimVerdict,
    Evidence,
    EvidenceVerdict,
    Label,
    SourceSpan,
    Trace,
    evidence_id,
)
from src.data.corpus import Corpus, Document
from src.data.examples import Example
from src.data.metrics import summarise_run
from src.pipeline import RunContext, analyze, analyze_example, gold_evidence_for

RESPONSE = "Marie Curie was born in Warsaw. She won two Nobel Prizes."

CORPUS = Corpus([
    Document("Marie_Curie", "Marie Curie",
             ("Marie Curie was a physicist.", "She was born in Warsaw.", "She won two Nobel Prizes.")),
    Document("Paris", "Paris", ("Paris is the capital of France.",)),
])


# --------------------------------------------------------------------------- #
# stub components
# --------------------------------------------------------------------------- #

class _Ext(ClaimExtractor):
    def extract(self, response: str) -> list[Claim]:
        out, cursor = [], 0
        for piece in response.split(". "):
            text = piece.strip().rstrip(".") + "."
            start = response.find(piece, cursor)
            cursor = start + len(piece)
            out.append(Claim.new("resp", text, self.name,
                                 source_span=SourceSpan(start, start + len(text))))
        return out


class _Ret(Retriever):
    """Returns the whole corpus, ranked by row, truncated to k."""

    def __init__(self, corpus_path: str = "") -> None:
        super().__init__()
        self.corpus = CORPUS

    def retrieve(self, claim: Claim, k: int) -> list[Evidence]:
        return [
            Evidence.new(s.doc_id, s.sent_id, s.text, score=1.0 / (i + 1),
                         retriever_name=self.name, rank=i + 1)
            for i, s in enumerate(self.corpus.sentences[:k])
        ]


class _BadRet(Retriever):
    """Violates the rank contract: 0-based ranks."""

    def retrieve(self, claim: Claim, k: int) -> list[Evidence]:
        return [Evidence.new("Marie_Curie", 0, "x", 1.0, self.name, rank=1),
                Evidence.new("Marie_Curie", 1, "y", 1.0, self.name, rank=3)]


class _Rer(Reranker):
    def rerank(self, claim: Claim, ev: list[Evidence], k: int) -> list[Evidence]:
        return [e.reranked(rank=i, score=e.score, retriever_name=e.retriever_name)
                for i, e in enumerate(ev[:k], start=1)]


class _InventingRer(Reranker):
    def rerank(self, claim: Claim, ev: list[Evidence], k: int) -> list[Evidence]:
        return [Evidence.new("Invented", 0, "not from the input", 1.0, self.name, 1)]


class _Ver(Verifier):
    """Entails iff the evidence shares a rare word with the claim."""

    def score(self, claim: Claim, ev: Evidence) -> EvidenceVerdict:
        overlap = {w.lower().strip(".") for w in claim.text.split() if len(w) > 4} & \
                  {w.lower().strip(".") for w in ev.text.split() if len(w) > 4}
        p = 0.9 if overlap else 0.02
        return EvidenceVerdict(claim.id, ev.id, p, 0.01, round(1 - p - 0.01, 6), None,
                               self.name, 1.0, ev.rank, ev.score)


class _LyingVer(Verifier):
    """Returns a verdict for the wrong evidence."""

    def score(self, claim: Claim, ev: Evidence) -> EvidenceVerdict:
        return EvidenceVerdict(claim.id, evidence_id("Elsewhere", 9), 0.5, 0.3, 0.2, None,
                               self.name, 1.0)


class _Agg(Aggregator):
    def aggregate(self, claim: Claim, v: list[EvidenceVerdict]) -> ClaimVerdict:
        best = max(v, key=lambda x: x.p_entail) if v else None
        return ClaimVerdict(
            claim.id,
            Label.SUPPORTED if best and best.p_entail > 0.5 else Label.INSUFFICIENT,
            float(best.p_entail) if best else 0.0,
            False, tuple(v), self.name,
            {"rule": "stub", "explanation": "stub explanation for the test suite",
             "decisive_evidence_ids": [best.evidence_id] if best else []},
        )


@pytest.fixture()
def stub_registry():
    saved = {kind: dict(reg) for kind, reg in registry._REGISTRY.items()}
    registry.register("extractor", "t_ext")(_Ext)
    registry.register("retriever", "t_ret")(_Ret)
    registry.register("retriever", "t_badret")(_BadRet)
    registry.register("reranker", "t_rer")(_Rer)
    registry.register("reranker", "t_inventing")(_InventingRer)
    registry.register("verifier", "t_ver")(_Ver)
    registry.register("verifier", "t_lying")(_LyingVer)
    registry.register("aggregator", "t_agg")(_Agg)
    yield
    for kind, reg in saved.items():
        registry._REGISTRY[kind] = reg


BASE_CFG = {
    "seed": 1,
    "pipeline": {"k": 3},
    "components": {"extractor": "t_ext", "retriever": "t_ret",
                   "reranker": "t_rer", "verifier": "t_ver", "aggregator": "t_agg"},
}


def build(cfg=None):
    return registry.build_pipeline(cfg or BASE_CFG)


def ctx_for(pipe, cfg=None):
    return RunContext.create(dict(cfg or BASE_CFG), pipe)


EXAMPLE = Example(
    id="ex-1", text=RESPONSE, gold_label=Label.SUPPORTED,
    gold_evidence=(("Marie_Curie", 1),), dataset="test",
)
NEI_EXAMPLE = Example(
    id="ex-nei", text="Curie's favourite colour was blue.", gold_label=Label.INSUFFICIENT,
    gold_evidence=(), dataset="test",
)


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #

def test_analyze_produces_a_complete_trace(stub_registry):
    pipe = build()
    trace = analyze(pipe, RESPONSE, ctx_for(pipe))
    assert len(trace.claims) == 2
    assert len(trace.verdicts) == 2
    assert set(trace.evidence_by_claim) == {c.id for c in trace.claims}
    assert trace.mode == "retrieved"


def test_every_stage_is_timed(stub_registry):
    pipe = build()
    t = analyze(pipe, RESPONSE, ctx_for(pipe))
    for stage in ("extract_ms", "retrieve_ms", "rerank_ms", "verify_ms", "aggregate_ms", "total_ms"):
        assert stage in t.timings
    assert t.timings["total_ms"] == pytest.approx(
        sum(v for k, v in t.timings.items() if k != "total_ms"), abs=0.01
    )


def test_trace_survives_a_json_round_trip(stub_registry):
    pipe = build()
    original = analyze(pipe, RESPONSE, ctx_for(pipe))
    assert Trace.from_json_line(original.to_json_line()) == original


def test_provenance_is_carried_into_the_trace(stub_registry):
    pipe = build()
    ctx = ctx_for(pipe)
    t = analyze(pipe, RESPONSE, ctx)
    assert t.run_id == ctx.run_id
    assert t.config_hash == ctx.config_hash
    assert t.resolved_config["components"]["verifier"]["name"] == "t_ver"
    assert t.resolved_config["k"] == 3


def test_k_is_respected(stub_registry):
    pipe = build({**BASE_CFG, "pipeline": {"k": 2}})
    t = analyze(pipe, RESPONSE, ctx_for(pipe))
    for evs in t.evidence_by_claim.values():
        assert len(evs) == 2


def test_gold_marking_happens_when_an_example_is_supplied(stub_registry):
    pipe = build()
    t = analyze(pipe, RESPONSE, ctx_for(pipe), example=EXAMPLE, corpus=CORPUS)
    flags = {e.is_gold for evs in t.evidence_by_claim.values() for e in evs}
    assert flags == {True, False}          # real booleans, not None


def test_no_example_means_is_gold_stays_unknown(stub_registry):
    pipe = build()
    t = analyze(pipe, RESPONSE, ctx_for(pipe))
    assert all(e.is_gold is None for evs in t.evidence_by_claim.values() for e in evs)


# --------------------------------------------------------------------------- #
# contract enforcement at the boundaries
# --------------------------------------------------------------------------- #

def test_a_retriever_violating_the_rank_contract_is_caught(stub_registry):
    pipe = build({**BASE_CFG, "components": {**BASE_CFG["components"], "retriever": "t_badret"}})
    with pytest.raises(ContractError, match="ranks must be 1-based and contiguous"):
        analyze(pipe, RESPONSE, ctx_for(pipe))


def test_a_retriever_returning_more_than_k_is_caught(stub_registry):
    class _OverlongRet(Retriever):
        def retrieve(self, claim: Claim, k: int) -> list[Evidence]:
            # Return k + 1 items
            return [
                Evidence.new("Marie_Curie", i, str(i), 1.0, self.name, rank=i+1)
                for i in range(k + 1)
            ]
    registry.register("retriever", "t_overlong")(_OverlongRet)
    pipe = build({**BASE_CFG, "components": {**BASE_CFG["components"], "retriever": "t_overlong"}})
    with pytest.raises(ContractError, match="returned .* items for k="):
        analyze(pipe, RESPONSE, ctx_for(pipe))


def test_a_reranker_inventing_evidence_is_caught(stub_registry):
    pipe = build({**BASE_CFG, "components": {**BASE_CFG["components"], "reranker": "t_inventing"}})
    with pytest.raises(ContractError, match="introduced evidence not in its input"):
        analyze(pipe, RESPONSE, ctx_for(pipe))


def test_a_verifier_answering_about_the_wrong_evidence_is_caught(stub_registry):
    pipe = build({**BASE_CFG, "components": {**BASE_CFG["components"], "verifier": "t_lying"}})
    with pytest.raises(ContractError, match="when asked about"):
        analyze(pipe, RESPONSE, ctx_for(pipe))


def test_checks_can_be_disabled_and_then_the_trace_catches_it_anyway(stub_registry):
    """Defence in depth: even with boundary checks off, Trace.__post_init__ refuses a
    verdict that scores evidence the claim never saw."""
    pipe = build({**BASE_CFG, "components": {**BASE_CFG["components"], "verifier": "t_lying"}})
    with pytest.raises(ValueError, match="index-alignment bug|duplicate pairwise"):
        analyze(pipe, RESPONSE, ctx_for(pipe), check_contracts=False)


def test_a_claim_with_wrong_span_is_caught(stub_registry):
    class _BadSpanExt(ClaimExtractor):
        def extract(self, response: str) -> list[Claim]:
            return [
                Claim.new(
                    "resp",
                    "Marie Curie was born in Paris.",
                    self.name,
                    source_span=SourceSpan(0, 10), # "Marie Curi" -> does not match claim text
                    extractor_meta={"decomposed": False},
                )
            ]
    registry.register("extractor", "t_badspan")(_BadSpanExt)
    pipe = build({**BASE_CFG, "components": {**BASE_CFG["components"], "extractor": "t_badspan"}})
    with pytest.raises(ContractError, match="whose text does not match the claim text"):
        analyze(pipe, RESPONSE, ctx_for(pipe))


# --------------------------------------------------------------------------- #
# the oracle path
# --------------------------------------------------------------------------- #

def test_oracle_substitutes_gold_evidence_only(stub_registry):
    pipe = build()
    t = analyze(pipe, RESPONSE, ctx_for(pipe), example=EXAMPLE, corpus=CORPUS, mode="oracle")
    assert t.mode == "oracle"
    for evs in t.evidence_by_claim.values():
        assert [e.id for e in evs] == [evidence_id("Marie_Curie", 1)]
        assert all(e.retriever_name == "oracle" for e in evs)
        assert all(e.is_gold for e in evs)


def test_standard_inference_cannot_access_gold_data(stub_registry):
    """Even if an Example with gold labels is provided, 'retrieved' mode must not leak it."""
    pipe = build()
    t = analyze(pipe, RESPONSE, ctx_for(pipe), example=EXAMPLE, corpus=CORPUS, mode="retrieved")
    assert t.mode == "retrieved"
    # Ensure retriever_name is not oracle
    for evs in t.evidence_by_claim.values():
        assert all(e.retriever_name != "oracle" for e in evs)
        # In retrieved mode, gold flags are added at the end (via mark_gold), 
        # but the retrieval set itself must be purely from the retriever,
        # which means it might contain non-gold items.
        assert not all(e.is_gold for e in evs)



def test_oracle_and_retrieved_differ_only_in_evidence(stub_registry):
    """The comparison is only meaningful if nothing else changed."""
    pipe = build()
    ctx = ctx_for(pipe)
    r = analyze(pipe, RESPONSE, ctx, example=EXAMPLE, corpus=CORPUS, mode="retrieved")
    o = analyze(pipe, RESPONSE, ctx, example=EXAMPLE, corpus=CORPUS, mode="oracle")
    assert [c.id for c in r.claims] == [c.id for c in o.claims]
    assert r.config_hash == o.config_hash
    assert r.resolved_config["components"] == o.resolved_config["components"]
    assert r.mode != o.mode


def test_oracle_refuses_a_gold_key_missing_from_the_corpus(stub_registry):
    """An incomplete oracle biases the central measurement invisibly."""
    bad = Example("ex-bad", RESPONSE, Label.SUPPORTED, (("Nonexistent_Page", 4),), "test")
    pipe = build()
    with pytest.raises(KeyError, match="not in this corpus"):
        analyze(pipe, RESPONSE, ctx_for(pipe), example=bad, corpus=CORPUS, mode="oracle")


def test_oracle_mode_requires_example_and_corpus(stub_registry):
    pipe = build()
    with pytest.raises(ValueError, match="oracle mode needs both"):
        analyze(pipe, RESPONSE, ctx_for(pipe), mode="oracle")


def test_analyze_example_skips_the_oracle_when_there_is_no_gold(stub_registry):
    """'No oracle exists' must stay distinguishable from 'the oracle found nothing'."""
    pipe = build()
    r, o = analyze_example(pipe, NEI_EXAMPLE, CORPUS, ctx_for(pipe))
    assert r.mode == "retrieved"
    assert o is None


def test_analyze_example_runs_both_conditions_when_gold_exists(stub_registry):
    pipe = build()
    r, o = analyze_example(pipe, EXAMPLE, CORPUS, ctx_for(pipe))
    assert (r.mode, o.mode) == ("retrieved", "oracle")


def test_gold_evidence_ranks_are_flagged_as_annotation_order():
    """Rank on oracle evidence is annotation order, not a retrieval ranking. A
    rank-weighted aggregator running on it is weighting by nothing meaningful."""
    ex = Example("e", "t", Label.SUPPORTED, (("Marie_Curie", 2), ("Marie_Curie", 1)), "test")
    evs = gold_evidence_for(ex, CORPUS)
    assert [e.rank for e in evs] == [1, 2]
    assert [e.sent_id for e in evs] == [2, 1]          # annotation order preserved
    assert all(e.retriever_meta["rank_is_annotation_order"] for e in evs)


# --------------------------------------------------------------------------- #
# trace_io
# --------------------------------------------------------------------------- #

def test_trace_writer_round_trip(tmp_path: Path, stub_registry):
    pipe = build()
    ctx = ctx_for(pipe)
    traces = [analyze(pipe, RESPONSE, ctx), analyze(pipe, "Another response here.", ctx)]
    p = tmp_path / "traces.jsonl"
    with TraceWriter(p) as w:
        for t in traces:
            w.write(t)
        assert w.count == 2
    assert list(read_traces(p)) == traces


def test_traces_are_flushed_per_write_so_a_crash_keeps_them(tmp_path: Path, stub_registry):
    pipe = build()
    p = tmp_path / "t.jsonl"
    w = TraceWriter(p)
    w.write(analyze(pipe, RESPONSE, ctx_for(pipe)))
    assert len(list(read_traces(p))) == 1      # readable BEFORE close()
    w.close()


def test_read_traces_refuses_a_malformed_line(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"not": "a trace"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid trace"):
        list(read_traces(p))


def test_write_run_creates_a_self_describing_directory(tmp_path: Path):
    d = write_run(tmp_path / "run1", resolved_config={"a": 1}, metrics={"n": 2})
    assert (d / "config.json").is_file()
    assert (d / "provenance.json").is_file()
    assert (d / "metrics.json").is_file()
    prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
    assert "environment" in prov and "python" in prov["environment"]


def test_metrics_carry_the_warning_inside_the_artifact(tmp_path: Path):
    """A metrics file read six weeks from now must not be mistakable for a result."""
    d = write_run(tmp_path / "run2", resolved_config={}, metrics={"n": 1})
    payload = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    assert "harness_kind=debug" in payload["_warning"]
    assert "Do not report them" in payload["_warning"]


def test_run_dir_is_named_for_the_run_id():
    assert run_dir_for("20260828T000000Z-abc123", "results").name == "20260828T000000Z-abc123"


# --------------------------------------------------------------------------- #
# failure cases — the error-analysis corpus
# --------------------------------------------------------------------------- #

def test_save_failure_case_embeds_the_whole_trace(tmp_path: Path, stub_registry):
    pipe = build()
    ctx = ctx_for(pipe)
    r = analyze(pipe, RESPONSE, ctx, example=EXAMPLE, corpus=CORPUS)
    o = analyze(pipe, RESPONSE, ctx, example=EXAMPLE, corpus=CORPUS, mode="oracle")
    path = save_failure_case(r, "retrieval missed the obvious sentence",
                             claim_id=r.claims[0].id, oracle_trace=o, out_dir=tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["note"] == "retrieval missed the obvious sentence"
    assert record["oracle_trace"]["mode"] == "oracle"
    # the WHOLE trace, so "why did it decide that" is answerable later
    assert record["trace"]["verdicts"][0]["aggregation_trace"]["explanation"]
    assert Trace.from_dict(record["trace"]) == r


def test_failure_cases_are_one_file_each(tmp_path: Path, stub_registry):
    pipe = build()
    ctx = ctx_for(pipe)
    t = analyze(pipe, RESPONSE, ctx)
    save_failure_case(t, "note one", claim_id="a", out_dir=tmp_path)
    save_failure_case(t, "note two", claim_id="b", out_dir=tmp_path)
    assert len(list(Path(tmp_path).glob("*.json"))) == 2
    assert {c["note"] for c in load_failure_cases(tmp_path)} == {"note one", "note two"}


# --------------------------------------------------------------------------- #
# run summary
# --------------------------------------------------------------------------- #

def test_summarise_computes_the_oracle_gap(stub_registry):
    pipe = build()
    ctx = ctx_for(pipe)
    r, o = analyze_example(pipe, EXAMPLE, CORPUS, ctx)
    m = summarise_run([r], {EXAMPLE.id: o}, {EXAMPLE.id: EXAMPLE})
    gap = m["retrieval_attributable_error"]
    assert gap["n_claims_with_both_conditions"] == 2
    assert gap["retrieved_agreement"] is not None
    assert gap["oracle_agreement"] is not None
    assert gap["gap"] == pytest.approx(gap["oracle_agreement"] - gap["retrieved_agreement"])


def test_summarise_excludes_examples_without_gold_from_recall(stub_registry):
    """NEI examples have no denominator; they must not be counted as 0.0 or 1.0."""
    pipe = build()
    ctx = ctx_for(pipe)
    r, _ = analyze_example(pipe, NEI_EXAMPLE, CORPUS, ctx)
    m = summarise_run([r], {}, {NEI_EXAMPLE.id: NEI_EXAMPLE})
    assert m["n_recall_defined"] == 0
    assert m["recall_at_k_mean"] is None


def test_summarise_carries_its_own_warning(stub_registry):
    pipe = build()
    r = analyze(pipe, RESPONSE, ctx_for(pipe))
    assert "NOT evaluation results" in summarise_run([r], {}, {})["_warning"]


def test_summarise_multi_claim_gap_computation(stub_registry):
    # We create a fake Trace with two claims that have different verdicts.
    c1 = Claim.new("r1", "claim 1", "test", None)
    c2 = Claim.new("r1", "claim 2", "test", None)
    
    def make_trace(v1, v2):
        return Trace(
            run_id="run1", config_hash="h", git_sha="g", resolved_config={"example_id": "e1"},
            mode="retrieved", response_text="text", timestamp="t", claims=[c1, c2], evidence_by_claim={c1.id: [], c2.id: []},
            verdicts=[
                ClaimVerdict(claim_id=c1.id, label=v1, confidence=1.0, abstained=False, per_evidence=(), aggregator_name="agg", aggregation_trace={"rule": "r", "explanation": "e", "decisive_evidence_ids": []}),
                ClaimVerdict(claim_id=c2.id, label=v2, confidence=1.0, abstained=False, per_evidence=(), aggregator_name="agg", aggregation_trace={"rule": "r", "explanation": "e", "decisive_evidence_ids": []})
            ],
            timings={}
        )
    
    # Retrieved gets claim1=Supported, claim2=Contradicted
    r_trace = make_trace(Label.SUPPORTED, Label.CONTRADICTED)
    
    # Oracle gets claim1=Contradicted, claim2=Supported
    o_trace = make_trace(Label.CONTRADICTED, Label.SUPPORTED)
    
    # Gold is Supported for the example
    from src.data.examples import Example
    ex = Example("e1", "text", Label.SUPPORTED, (("d1", 1),), "test")
    
    m = summarise_run([r_trace], {"e1": o_trace}, {"e1": ex})
    gap = m["retrieval_attributable_error"]
    
    # Claim 1: Gold=Supported. Retrieved=Supported (correct), Oracle=Contradicted (wrong) -> broken by oracle
    # Claim 2: Gold=Supported. Retrieved=Contradicted (wrong), Oracle=Supported (correct) -> fixed by oracle
    
    assert gap["n_claims_with_both_conditions"] == 2
    assert gap["n_fixed_by_oracle"] == 1
    assert gap["n_broken_by_oracle"] == 1
    assert gap["retrieved_agreement"] == 0.5  # 1/2 correct
    assert gap["oracle_agreement"] == 0.5     # 1/2 correct

