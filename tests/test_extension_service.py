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
