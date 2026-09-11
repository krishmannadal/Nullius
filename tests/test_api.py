"""Contract tests for Nullius FastAPI backend.

Tests all endpoints:
  - GET  /health
  - POST /analyze
  - POST /analyze/oracle
  - POST /verify/quick
  - POST /verify/full (including SSE streaming)
  - POST /annotate
  - Error handling and contract mapping
  - OpenAPI schema completeness

Uses an in-process ASGI test client requiring no external network dependencies.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest

from src.api.app import create_app
from src.core.types import SCHEMA_VERSION


class ASGIResponse:
    def __init__(self, status_code: int, headers: list[tuple[bytes, bytes]], body: bytes):
        self.status_code = status_code
        self._raw_headers = headers
        self.headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in headers}
        self.content = body

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")


class ASGIClient:
    """Zero-dependency in-process ASGI 3 test client."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> ASGIResponse:
        method = method.upper()
        raw_headers: list[tuple[bytes, bytes]] = []

        body_bytes = b""
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode("utf-8")
            raw_headers.append((b"content-type", b"application/json"))

        if headers:
            for k, v in headers.items():
                raw_headers.append((k.lower().encode("latin1"), v.encode("latin1")))

        response_status = 200
        response_headers: list[tuple[bytes, bytes]] = []
        response_chunks: list[bytes] = []

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": raw_headers,
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 12345),
        }

        request_sent = False
        disconnect_event = asyncio.Event()

        async def receive() -> dict[str, Any]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body_bytes,
                    "more_body": False,
                }
            await disconnect_event.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            nonlocal response_status, response_headers
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = message.get("headers", [])
            elif message["type"] == "http.response.body":
                response_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    disconnect_event.set()

        try:
            await self.app(scope, receive, send)
        finally:
            disconnect_event.set()

        return ASGIResponse(response_status, response_headers, b"".join(response_chunks))

    def get(self, path: str, **kwargs: Any) -> ASGIResponse:
        return asyncio.run(self.request("GET", path, **kwargs))

    def post(self, path: str, **kwargs: Any) -> ASGIResponse:
        return asyncio.run(self.request("POST", path, **kwargs))


@pytest.fixture(scope="module")
def client():
    app = create_app(default_config_path="configs/mini.yaml")
    return ASGIClient(app)


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #

def test_health_endpoint(client: ASGIClient):
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["schema_version"] == SCHEMA_VERSION
    assert "config_hash" in data
    assert "components" in data
    assert "extractor" in data["components"]
    assert "device" in data
    assert isinstance(data["queue_depth"], int)


# --------------------------------------------------------------------------- #
# /analyze
# --------------------------------------------------------------------------- #

def test_analyze_valid_input(client: ASGIClient):
    payload = {
        "text": "Marie Curie was born in Warsaw. She was a physicist.",
    }
    resp = client.post("/analyze", json_body=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["mode"] == "retrieved"
    assert len(data["claims"]) >= 1
    assert len(data["verdicts"]) == len(data["claims"])
    assert "run_id" in data
    assert "config_hash" in data
    assert "timings" in data

    # Verify claim-level invariant
    for claim in data["claims"]:
        cid = claim["id"]
        assert cid in data["evidence_by_claim"]
        v = next((v for v in data["verdicts"] if v["claim_id"] == cid), None)
        assert v is not None
        assert "explanation" in v["aggregation_trace"]


def test_analyze_empty_input_fails(client: ASGIClient):
    resp = client.post("/analyze", json_body={"text": ""})
    assert resp.status_code == 422


def test_analyze_whitespace_only_fails(client: ASGIClient):
    resp = client.post("/analyze", json_body={"text": "   "})
    assert resp.status_code in (400, 422)


def test_analyze_repeated_deterministic(client: ASGIClient):
    payload = {"text": "Marie Curie was a physicist."}
    r1 = client.post("/analyze", json_body=payload).json()
    r2 = client.post("/analyze", json_body=payload).json()
    assert [c["id"] for c in r1["claims"]] == [c["id"] for c in r2["claims"]]
    assert [v["label"] for v in r1["verdicts"]] == [v["label"] for v in r2["verdicts"]]


# --------------------------------------------------------------------------- #
# /analyze/oracle
# --------------------------------------------------------------------------- #

def test_analyze_oracle_success(client: ASGIClient):
    # mini-001 exists in data/debug/mini/examples.jsonl
    payload = {"example_id": "mini-001"}
    resp = client.post("/analyze/oracle", json_body=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["mode"] == "oracle"
    assert data["resolved_config"]["example_id"] == "mini-001"
    # Oracle evidence must be marked as gold
    for ev_list in data["evidence_by_claim"].values():
        for ev in ev_list:
            assert ev["is_gold"] is True
            assert ev["retriever_name"] == "oracle"


def test_analyze_oracle_missing_example(client: ASGIClient):
    resp = client.post("/analyze/oracle", json_body={"example_id": "nonexistent-id-999"})
    assert resp.status_code == 404


def test_oracle_isolation_cannot_leak_into_analyze(client: ASGIClient):
    """Calling oracle must not pollute subsequent /analyze calls."""
    # 1. Call oracle
    client.post("/analyze/oracle", json_body={"example_id": "mini-001"})

    # 2. Call standard analyze with the same text
    resp = client.post("/analyze", json_body={"text": "Marie Curie was born in Warsaw."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "retrieved"
    for ev_list in data["evidence_by_claim"].values():
        for ev in ev_list:
            assert ev["retriever_name"] != "oracle"


# --------------------------------------------------------------------------- #
# /verify/quick
# --------------------------------------------------------------------------- #

def test_verify_quick_tier1(client: ASGIClient):
    payload = {
        "text": "Marie Curie was born in Warsaw. An unknown fictitious alien arrived on Mars in 1300.",
    }
    resp = client.post("/verify/quick", json_body=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "claims" in data
    assert len(data["claims"]) >= 2
    assert "counts" in data
    assert "likely_checkable" in data["counts"]
    assert "no_evidence_found" in data["counts"]
    assert data["counts"]["total_claims"] == len(data["claims"])

    # Marie Curie sentence should be checkable in mini corpus
    curie_claim = next(c for c in data["claims"] if "Curie" in c["claim_text"])
    assert curie_claim["status"] == "likely-checkable"
    assert curie_claim["n_evidence_found"] > 0
    assert curie_claim["top_evidence_score"] is not None


# --------------------------------------------------------------------------- #
# /verify/full
# --------------------------------------------------------------------------- #

def test_verify_full_json(client: ASGIClient):
    payload = {
        "text": "Marie Curie was born in Warsaw.",
        "stream": False,
    }
    resp = client.post("/verify/full", json_body=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["mode"] == "retrieved"
    assert len(data["claims"]) == 1
    assert len(data["verdicts"]) == 1


def test_verify_full_sse_streaming(client: ASGIClient):
    payload = {
        "text": "Marie Curie was born in Warsaw.",
        "stream": True,
    }
    resp = client.post("/verify/full", json_body=payload)
    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers.get("content-type", "")

    body = resp.text
    assert "event: start" in body
    assert "event: claim_verdict" in body
    assert "event: complete" in body


# --------------------------------------------------------------------------- #
# /annotate
# --------------------------------------------------------------------------- #

def test_annotate_valid_record(client: ASGIClient):
    from pathlib import Path

    payload = {
        "site": "chatgpt.com",
        "model_name": "gpt-4o",
        "response_text": "Marie Curie won two Nobel prizes.",
        "claim_text": "Marie Curie won two Nobel prizes.",
        "human_label": "agree",
        "human_note": "Verified accurate fact.",
    }
    try:
        resp = client.post("/annotate", json_body=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "saved"
        record = data["record"]
        assert record["site"] == "chatgpt.com"
        assert record["human_label"] == "agree"
        assert record["response_text"] == payload["response_text"]
    finally:
        test_file = Path("data/annotations/annotations.jsonl")
        if test_file.is_file():
            test_file.unlink()


def test_annotate_missing_fields_fails(client: ASGIClient):
    # Missing required 'human_label'
    payload = {
        "response_text": "Marie Curie won two Nobel prizes.",
        "claim_text": "Marie Curie won two Nobel prizes.",
    }
    resp = client.post("/annotate", json_body=payload)
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# OpenAPI Schema
# --------------------------------------------------------------------------- #

def test_openapi_schema(client: ASGIClient):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200, resp.text
    schema = resp.json()
    paths = schema.get("paths", {})
    for ep in ("/health", "/analyze", "/analyze/oracle", "/verify/quick", "/verify/full", "/annotate"):
        assert ep in paths, f"Endpoint {ep} missing from OpenAPI paths"
