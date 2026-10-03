"""Exercise real lightweight browser retrieval without loading transformer models."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from src.api.app import create_app
from src.core.registry import build as real_build
from src.service import NulliusService
from tests.test_api import ASGIClient


def test_quick_check_does_not_construct_verifier_and_reuses_components():
    service = NulliusService("configs/extension.yaml")
    built = []

    def record_build(kind, spec):
        assert kind in {"extractor", "retriever"}, "Quick Check must not build NLI or rerankers"
        built.append(kind)
        return real_build(kind, spec)

    with patch("src.service.build", side_effect=record_build):
        first = asyncio.run(service.verify_quick("Water is a liquid.", evidence_floor_score=0.01))
        second = asyncio.run(service.verify_quick("zyxqvkjw bogusword.", evidence_floor_score=0.01))
    assert built == ["extractor", "retriever"]
    assert first.claims
    assert first.counts["total_claims"] == len(first.claims)
    assert second.claims[0].status == "no-evidence-found"
    assert service._cached_pipeline is None
    assert service.queue_depth == 0


def test_extension_api_uses_selected_service_and_returns_schema():
    service = NulliusService("configs/extension.yaml")
    client = ASGIClient(create_app(service=service))
    response = client.post(
        "/verify/quick",
        json_body={
            "text": "Water is a liquid.",
            "config_path": "configs/extension.yaml",
            "evidence_floor_score": 0.01,
            "retrieve_k": 5,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["response_text"] == "Water is a liquid."
    assert data["claims"][0]["claim_text"]
    assert data["counts"]["total_claims"] == len(data["claims"])
    assert service._cached_pipeline is None


def test_full_warmup_is_reused_for_explicit_default_config_path():
    # A sentinel is sufficient to test initialization reuse without fabricating scores.
    service = NulliusService("configs/extension.yaml")
    pipeline = object()
    with patch("src.service.build_pipeline", return_value=pipeline) as builder:
        service._ensure_loaded()
        _, selected = service._ensure_loaded("configs/extension.yaml")
    assert selected is pipeline
    builder.assert_called_once()


def stored_fixture_service():
    """Synthetic trace for wiring tests only; never a model output or S1 annotation."""
    from src.core.types import Claim, Evidence, EvidenceVerdict, Trace, utcnow_iso

    service = NulliusService("configs/extension.yaml")
    claim = Claim.new("fixture-response", "A fixture claim.", "fixture-extractor")
    evidence = tuple(
        Evidence.new(
            "fixture-doc", i, f"Fixture evidence {i}.", 1 / (i + 1), "fixture-retriever", i + 1
        )
        for i in range(3)
    )
    probabilities = ((0.65, 0.05, 0.30), (0.05, 0.75, 0.20), (0.05, 0.10, 0.85))
    pairs = [
        EvidenceVerdict(
            claim_id=claim.id,
            evidence_id=item.id,
            p_entail=probability[0],
            p_contra=probability[1],
            p_neutral=probability[2],
            similarity=None,
            verifier_name="synthetic-fixture",
            latency_ms=0,
            evidence_rank=item.rank,
            evidence_score=item.score,
        )
        for item, probability in zip(evidence, probabilities)
    ]
    original = real_build("aggregator", "threshold_abstain").aggregate(claim, pairs)
    trace = Trace(
        run_id="fixture-run",
        config_hash="fixture-config",
        git_sha=None,
        timestamp=utcnow_iso(),
        response_text=claim.text,
        claims=(claim,),
        evidence_by_claim={claim.id: evidence},
        verdicts=(original,),
    )
    service._store_run(trace)
    return service, trace, pairs


def test_comparison_preserves_exact_scores_without_model_initialization():
    service, trace, pairs = stored_fixture_service()
    original = trace.to_dict()
    client = ASGIClient(create_app(service=service))
    rules = ["max_entailment", "noisy_or", "weighted_by_retrieval", "threshold_abstain", "majority"]
    with patch.object(
        service, "_ensure_loaded", side_effect=AssertionError("No model initialization")
    ):
        response = client.post(
            "/reaggregate",
            json_body={
                "run_id": trace.run_id,
                "target_aggregators": rules,
                "aggregator_configs": {},
            },
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["run_id"] == trace.run_id
    assert data["config_hash"] == trace.config_hash
    for rule in rules:
        assert data["comparisons"][trace.claims[0].id][rule]["per_evidence"] == [
            p.to_dict() for p in pairs
        ]
    assert trace.to_dict() == original


def test_client_scores_and_empty_rule_lists_are_rejected():
    service, trace, _ = stored_fixture_service()
    client = ASGIClient(create_app(service=service))
    response = client.post(
        "/reaggregate",
        json_body={
            "run_id": trace.run_id,
            "target_aggregators": ["majority"],
            "pairwise_scores": [],
        },
    )
    assert response.status_code == 422
    response = client.post(
        "/reaggregate", json_body={"run_id": trace.run_id, "target_aggregators": []}
    )
    assert response.status_code == 422


def test_rule_cannot_mutate_another_rules_inputs_or_stored_trace():
    service, trace, pairs = stored_fixture_service()
    original = trace.to_dict()
    delegate = real_build("aggregator", "majority")

    class ClearingRule:
        def aggregate(self, claim, verdicts):
            verdicts.clear()
            return delegate.aggregate(claim, verdicts)

    def selected_build(kind, spec):
        if spec["name"] == "fixture-clearing-rule":
            return ClearingRule()
        return real_build(kind, spec)

    with patch("src.service.build", side_effect=selected_build):
        result = service.reaggregate(trace.run_id, ["fixture-clearing-rule", "majority"], {})
    assert result["comparisons"][trace.claims[0].id]["majority"]["per_evidence"] == [
        p.to_dict() for p in pairs
    ]
    assert trace.to_dict() == original
    assert isinstance(
        service._execution_store[trace.run_id].claims_inputs[trace.claims[0].id][1], tuple
    )


def test_expired_comparison_returns_recoverable_not_found():
    service, trace, _ = stored_fixture_service()
    service._execution_store[trace.run_id].expires_at = 0
    response = ASGIClient(create_app(service=service)).post(
        "/reaggregate",
        json_body={
            "run_id": trace.run_id,
            "target_aggregators": ["majority"],
        },
    )
    assert response.status_code == 404
