"""Research state/provenance regression tests without model inference."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.core.interfaces import ContractError, check_claims, check_rerank_is_subset
from src.core.registry import build
from src.core.types import Claim, Evidence, EvidenceVerdict, SourceSpan, Trace
from src.service import NulliusService


def trace_fixture(run_id="fixture-run"):
    claim = Claim.new(
        "r", "A fixture assertion.", "fixture", SourceSpan(0, 20), {"nested": {"value": 1}}
    )
    evidence = Evidence.new("fixture-doc", 0, "Fixture evidence.", 1.0, "fixture", 1)
    verdict = EvidenceVerdict(claim.id, evidence.id, 0.8, 0.1, 0.1, None, "fixture", 0.0, 1, 1.0)
    final = build("aggregator", "max_entailment").aggregate(claim, [verdict])
    return Trace(
        run_id,
        "cfg",
        None,
        "fixture-time",
        claim.text,
        (claim,),
        {claim.id: (evidence,)},
        (final,),
        resolved_config={"data_identity": {"corpus": {"sha256": "original-corpus"}}},
    )


def test_storage_copy_and_reaggregation_preserve_original_inputs(tmp_path, monkeypatch):
    service = NulliusService(failure_dir=tmp_path)
    trace = trace_fixture()
    service._store_run(trace)
    trace.claims[0].extractor_meta["nested"]["value"] = 99
    trace.resolved_config["data_identity"]["corpus"]["sha256"] = "tampered"
    monkeypatch.setattr(
        service,
        "_ensure_loaded",
        lambda *a, **k: pytest.fail("reaggregation must not load inference"),
    )
    result = service.reaggregate(trace.run_id, ["majority", "max_entailment"], {})
    assert set(result["aggregator_configurations"]) == {"majority", "max_entailment"}
    pairs = [v["per_evidence"] for v in result["comparisons"][trace.claims[0].id].values()]
    assert pairs[0] == pairs[1]
    result["comparisons"][trace.claims[0].id]["majority"]["aggregation_trace"]["rule"] = "tampered"
    saved = service.save_failure_case(trace.run_id, trace.claims[0].id, "unclear", "Fixture case")
    data = json.loads(Path(saved["file_path"]).read_text())
    assert data["corpus_fingerprint"] == "original-corpus"
    assert data["trace"]["claims"][0]["extractor_meta"]["nested"]["value"] == 1
    with pytest.raises(ValueError, match="server-held"):
        service.save_failure_case(
            trace.run_id,
            trace.claims[0].id,
            "unclear",
            "Fixture",
            alternative_aggregations={"fake": {"confidence": 1}},
        )


def test_volatile_store_expiry_and_eviction(tmp_path, monkeypatch):
    import src.service as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    service = NulliusService(failure_dir=tmp_path)
    service._STORE_TTL = 1
    service._MAX_STORE_SIZE = 1
    first = trace_fixture("first")
    service._store_run(first)
    now[0] = 100.1
    service._store_run(trace_fixture("second"))
    with pytest.raises(KeyError):
        service.reaggregate("first", ["majority"], {})
    now[0] = 102
    with pytest.raises(KeyError):
        service.reaggregate("second", ["majority"], {})


def test_service_and_retriever_cache_track_content(tmp_path):
    from src.components.retrievers import load_corpus_cached

    source = Path("data/debug/mini/corpus.jsonl").read_text(encoding="utf-8").splitlines()[0]
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(source + "\n", encoding="utf-8")
    b.write_text(source.replace("Marie", "Maria") + "\n", encoding="utf-8")
    service = NulliusService()
    ca = service.get_corpus({"paths": {"corpus": str(a)}})
    cb = service.get_corpus({"paths": {"corpus": str(b)}})
    assert ca is not cb
    original = load_corpus_cached(str(a))
    a.write_text(source.replace("Marie", "Maria") + "\n", encoding="utf-8")
    changed = load_corpus_cached(str(a))
    assert (
        original is not changed and original.content_fingerprint() != changed.content_fingerprint()
    )
    assert service.get_corpus({"paths": {"corpus": str(a)}}) is not ca


def test_approximate_span_still_has_bounds():
    claim = Claim.new("r", "rewritten", "fixture", SourceSpan(100, 200), {"span_is_exact": False})
    with pytest.raises(ContractError, match="outside"):
        check_claims([claim], "short")


def test_reranker_cannot_change_evidence_text():
    evidence = Evidence.new("doc", 0, "Original", 1, "fixture", 1)
    with pytest.raises(ContractError, match="changed evidence"):
        check_rerank_is_subset([evidence], [replace(evidence, text="Changed")])


@pytest.mark.parametrize(
    "change", ["missing_verdict", "missing_evidence", "unscored", "duplicate_decisive"]
)
def test_trace_and_verdict_completeness(change):
    trace = trace_fixture()
    with pytest.raises(ValueError):
        if change == "missing_verdict":
            replace(trace, verdicts=())
        elif change == "missing_evidence":
            replace(trace, evidence_by_claim={})
        elif change == "unscored":
            extra = Evidence.new("other", 0, "Other", 0.5, "fixture", 2)
            replace(
                trace,
                evidence_by_claim={
                    trace.claims[0].id: (*trace.evidence_by_claim[trace.claims[0].id], extra)
                },
            )
        else:
            verdict = trace.verdicts[0]
            replace(
                verdict,
                aggregation_trace={
                    **verdict.aggregation_trace,
                    "decisive_evidence_ids": [verdict.per_evidence[0].evidence_id] * 2,
                },
            )


def test_api_factory_and_annotation_path_isolation(tmp_path):
    from src.api.app import create_app
    from tests.test_api import ASGIClient

    existing = tmp_path / "human-annotations.jsonl"
    existing.write_bytes(b"real pre-existing record must remain identical\n")
    service = NulliusService(
        default_config_path="configs/debug.yaml", annotation_dir=tmp_path / "isolated"
    )
    app = create_app(service=service)
    client = ASGIClient(app)
    before = existing.read_bytes()
    result = client.post(
        "/annotate",
        json_body={"response_text": "Fixture", "claim_text": "Fixture", "human_label": "unclear"},
    )
    assert result.status_code == 200
    assert existing.read_bytes() == before
    assert (tmp_path / "isolated/annotations.jsonl").is_file()
    assert app.state.service.default_config_path == "configs/debug.yaml"
    assert (
        create_app(default_config_path="configs/debug.yaml").state.service.default_config_path
        == "configs/debug.yaml"
    )


def test_api_rejects_client_pairwise_payload():
    from pydantic import ValidationError

    from src.api.schemas import ReaggregateRequest

    with pytest.raises(ValidationError):
        ReaggregateRequest(run_id="r", target_aggregators=["majority"], verifier_outputs=[])


@pytest.mark.parametrize(
    "arguments",
    [
        {"config_path": "../outside.yaml"},
        {"config_path": "data/eval/s1/metadata.json"},
        {"overrides": ["paths.corpus=private.jsonl"]},
        {"overrides": ["components.retriever.params.index_dir=private"]},
        {"overrides": ["pipeline.k=0"]},
    ],
)
def test_http_configuration_rejects_filesystem_control(arguments):
    from pydantic import ValidationError

    from src.api.schemas import AnalyzeRequest

    with pytest.raises(ValidationError):
        AnalyzeRequest(text="Fixture", **arguments)


def test_http_configuration_accepts_local_config_and_bounded_tuning():
    from src.api.schemas import AnalyzeRequest

    request = AnalyzeRequest(
        text="Fixture", config_path="configs/mini.yaml", overrides=["pipeline.k=3", "seed=1337"]
    )
    assert request.config_path.endswith("mini.yaml")
