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
from pathlib import Path
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
def client(tmp_path_factory):
    from src.service import NulliusService
    directory = tmp_path_factory.mktemp("api-artifacts")
    service = NulliusService(annotation_dir=directory / "annotations", failure_dir=directory / "failures")
    app = create_app(default_config_path="configs/mini.yaml", service=service)
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
    verdict = data["verdicts"][0]
    assert len(verdict["per_evidence"]) > 0
    ev0 = verdict["per_evidence"][0]
    assert "p_entail" in ev0
    assert "p_neutral" in ev0
    assert "p_contra" in ev0
    assert "similarity" in ev0
    assert ev0["p_neutral"] is not None
    assert 0.0 <= ev0["p_neutral"] <= 1.0
    assert ev0["p_entail"] + ev0["p_neutral"] + ev0["p_contra"] == pytest.approx(1.0, abs=1e-4)


def test_extension_popup_contract():
    popup_path = Path("src/extension/popup.js")
    assert popup_path.is_file(), "popup.js must exist"
    content = popup_path.read_text(encoding="utf-8")

    # 1. p_neutral rendered directly from backend response without client-side calculation
    assert "evVerdict.p_neutral" in content, "p_neutral must be read directly from evVerdict"
    assert "Neutral:" in content, "Neutral score label must be rendered"
    assert "1 - evVerdict" not in content and "1 - p_" not in content, "Do not calculate 1 - p_entail - p_contra"

    # 2. Separate endpoints and timeouts
    assert "'http://127.0.0.1:8000/verify/quick'" in content
    assert "5000" in content, "5s timeout for Quick Check"
    assert "'http://127.0.0.1:8000/verify/full'" in content
    assert "60000" in content, "60s timeout for Full Inspection"

    # 3. Security invariants: zero innerHTML, zero eval
    assert ".innerHTML" not in content, "Zero innerHTML property usage"
    assert "eval(" not in content, "Zero eval allowed"

    # 4. Safe DOM methods used
    assert "createElement" in content
    assert "replaceChildren" in content

    # 5. No background polling or mutation observers
    assert "MutationObserver" not in content
    assert "setInterval" not in content

    # 6. Traceability and integrity
    assert "decisive_evidence_ids" in content, "Must read decisive evidence IDs from aggregation_trace"
    assert "badge-decisive" in content, "Must render decisive badge"
    assert "\\u2605 DECISIVE" in content or "\u2605 DECISIVE" in content, "Must render decisive label"


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
    payload = {
        "site": "chatgpt.com",
        "model_name": "gpt-4o",
        "response_text": "Marie Curie won two Nobel prizes.",
        "claim_text": "Marie Curie won two Nobel prizes.",
        "human_label": "agree",
        "human_note": "Verified accurate fact.",
    }
    service = client.app.state.service
    test_file = service.annotation_dir / "annotations.jsonl"
    before = test_file.read_text(encoding="utf-8") if test_file.exists() else ""
    resp = client.post("/annotate", json_body=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "saved"
    record = data["record"]
    assert record["site"] == "chatgpt.com"
    assert record["human_label"] == "agree"
    assert record["response_text"] == payload["response_text"]
    after = test_file.read_text(encoding="utf-8")
    assert len(after) > len(before)
    assert "Marie Curie won two Nobel prizes" in after


def test_annotate_missing_fields_fails(client: ASGIClient):
    # Missing required 'human_label'
    payload = {
        "response_text": "Marie Curie won two Nobel prizes.",
        "claim_text": "Marie Curie won two Nobel prizes.",
    }
    resp = client.post("/annotate", json_body=payload)
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Reaggregate
# --------------------------------------------------------------------------- #

def test_reaggregate_success_from_analyze(client: ASGIClient):
    # 1. Run analyze to populate the execution store
    payload = {"text": "Marie Curie was a physicist."}
    resp1 = client.post("/analyze", json_body=payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    run_id = data1["run_id"]
    claim_id = data1["claims"][0]["id"]

    # 2. Call reaggregate
    reagg_payload = {
        "run_id": run_id,
        "target_aggregators": ["majority", "max_entailment"],
        "aggregator_configs": {}
    }
    resp2 = client.post("/reaggregate", json_body=reagg_payload)
    assert resp2.status_code == 200
    data2 = resp2.json()

    assert data2["run_id"] == run_id
    assert claim_id in data2["comparisons"]
    comparisons = data2["comparisons"][claim_id]
    
    assert "majority" in comparisons
    assert "max_entailment" in comparisons
    assert comparisons["majority"]["aggregator_name"] == "majority"
    assert comparisons["max_entailment"]["aggregator_name"] == "max_entailment"


def test_reaggregate_missing_run_id(client: ASGIClient):
    payload = {
        "run_id": "nonexistent_run_id_123",
        "target_aggregators": ["majority"],
    }
    resp = client.post("/reaggregate", json_body=payload)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_reaggregate_invalid_aggregator(client: ASGIClient):
    # Need a valid run_id first
    resp1 = client.post("/analyze", json_body={"text": "Marie Curie."})
    run_id = resp1.json()["run_id"]

    payload = {
        "run_id": run_id,
        "target_aggregators": ["unknown_aggregator_999"],
    }
    resp2 = client.post("/reaggregate", json_body=payload)
    assert resp2.status_code == 400
    assert "no aggregator named" in resp2.json()["detail"].lower()


def test_reaggregate_validation_error(client: ASGIClient):
    # Missing required field
    payload = {
        "run_id": "123",
        # missing target_aggregators
    }
    resp = client.post("/reaggregate", json_body=payload)
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /failure-cases
# --------------------------------------------------------------------------- #

def test_save_failure_case_success(client: ASGIClient):
    # 1. Run analyze
    payload = {"text": "Marie Curie was born in Warsaw."}
    resp1 = client.post("/analyze", json_body=payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    run_id = data1["run_id"]
    claim_id = data1["claims"][0]["id"]
    system_label = data1["verdicts"][0]["label"]

    # 2. Save failure case
    fc_payload = {
        "run_id": run_id,
        "claim_id": claim_id,
        "failure_category": "verifier_error",
        "researcher_note": "Model was overconfident on distractor hit",
        "human_label": "Contradicted",
    }
    resp2 = client.post("/failure-cases", json_body=fc_payload)
    assert resp2.status_code == 200, resp2.text
    data2 = resp2.json()
    assert data2["status"] == "saved"
    assert data2["failure_case_id"].startswith("fc_")
    file_path = Path(data2["file_path"])
    assert file_path.is_file()

    # 3. Verify saved content
    saved = json.loads(file_path.read_text(encoding="utf-8"))
    assert saved["failure_case_id"] == data2["failure_case_id"]
    assert saved["claim_id"] == claim_id
    assert saved["researcher_annotation"]["failure_category"] == "verifier_error"
    assert saved["researcher_annotation"]["researcher_note"] == "Model was overconfident on distractor hit"
    assert saved["researcher_annotation"]["human_label"] == "Contradicted"
    assert saved["trace"]["run_id"] == run_id
    assert saved["config_hash"] == data1["config_hash"]
    # System verdict is preserved and not overwritten by human label
    assert saved["trace"]["verdicts"][0]["label"] == system_label


def test_save_failure_case_with_oracle_and_reaggregate(client: ASGIClient):
    text = "Marie Curie was born in Warsaw."
    # 1. Retrieved run
    r1 = client.post("/analyze", json_body={"text": text}).json()
    run_id = r1["run_id"]
    claim_id = r1["claims"][0]["id"]

    # 2. Oracle run
    r_oracle = client.post("/analyze/oracle", json_body={"response_text": text}).json()
    oracle_run_id = r_oracle["run_id"]

    # 3. Reaggregate
    reagg_payload = {
        "run_id": run_id,
        "target_aggregators": ["majority", "max_entailment"],
    }
    r_reagg = client.post("/reaggregate", json_body=reagg_payload).json()
    alt_aggs = r_reagg["comparisons"][claim_id]

    # 4. Save failure case with oracle and alternative aggregations
    fc_payload = {
        "run_id": run_id,
        "claim_id": claim_id,
        "failure_category": "oracle_disagreement",
        "researcher_note": "Disagreement under investigation",
        "oracle_run_id": oracle_run_id,
        "alternative_aggregations": alt_aggs,
    }
    resp = client.post("/failure-cases", json_body=fc_payload)
    assert resp.status_code == 200
    saved_file = Path(resp.json()["file_path"])
    assert saved_file.is_file()

    saved = json.loads(saved_file.read_text(encoding="utf-8"))
    assert saved["oracle_trace"] is not None
    assert saved["oracle_trace"]["mode"] == "oracle"
    assert saved["oracle_trace"]["run_id"] == oracle_run_id
    assert saved["alternative_aggregations"] is not None
    assert "majority" in saved["alternative_aggregations"]
    assert "max_entailment" in saved["alternative_aggregations"]


def test_save_failure_case_unknown_run(client: ASGIClient):
    fc_payload = {
        "run_id": "nonexistent_run_999",
        "claim_id": "clm_fake",
        "failure_category": "unclear",
        "researcher_note": "Test note",
    }
    resp = client.post("/failure-cases", json_body=fc_payload)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_save_failure_case_unknown_claim(client: ASGIClient):
    r1 = client.post("/analyze", json_body={"text": "Marie Curie was a physicist."}).json()
    run_id = r1["run_id"]

    fc_payload = {
        "run_id": run_id,
        "claim_id": "clm_nonexistent",
        "failure_category": "unclear",
        "researcher_note": "Test note",
    }
    resp = client.post("/failure-cases", json_body=fc_payload)
    assert resp.status_code == 404
    assert "claim id" in resp.json()["detail"].lower()


def test_save_failure_case_unknown_oracle_run(client: ASGIClient):
    r1 = client.post("/analyze", json_body={"text": "Marie Curie was a physicist."}).json()
    run_id = r1["run_id"]
    claim_id = r1["claims"][0]["id"]

    fc_payload = {
        "run_id": run_id,
        "claim_id": claim_id,
        "failure_category": "oracle_disagreement",
        "researcher_note": "Test note",
        "oracle_run_id": "oracle_nonexistent",
    }
    resp = client.post("/failure-cases", json_body=fc_payload)
    assert resp.status_code == 404
    assert "oracle run id" in resp.json()["detail"].lower()


def test_save_failure_case_invalid_category(client: ASGIClient):
    r1 = client.post("/analyze", json_body={"text": "Marie Curie was a physicist."}).json()
    run_id = r1["run_id"]
    claim_id = r1["claims"][0]["id"]

    fc_payload = {
        "run_id": run_id,
        "claim_id": claim_id,
        "failure_category": "hallucination_detected",  # invalid category
        "researcher_note": "Test note",
    }
    resp = client.post("/failure-cases", json_body=fc_payload)
    assert resp.status_code == 422


def test_save_failure_case_mismatched_oracle_run(client: ASGIClient):
    # Run 1: Marie Curie
    r1 = client.post("/analyze", json_body={"text": "Marie Curie was born in Warsaw."}).json()
    run_id = r1["run_id"]
    claim_id = r1["claims"][0]["id"]

    # Oracle run 2: Polonium (mini-004)
    r_oracle = client.post("/analyze/oracle", json_body={"example_id": "mini-004"}).json()
    oracle_run_id = r_oracle["run_id"]

    fc_payload = {
        "run_id": run_id,
        "claim_id": claim_id,
        "failure_category": "oracle_disagreement",
        "researcher_note": "Mismatched test",
        "oracle_run_id": oracle_run_id,
    }
    resp = client.post("/failure-cases", json_body=fc_payload)
    assert resp.status_code == 400
    assert "mismatched" in resp.json()["detail"].lower()


def test_save_failure_case_duplicate_saves_create_distinct_files(client: ASGIClient):
    r1 = client.post("/analyze", json_body={"text": "Marie Curie was born in Warsaw."}).json()
    run_id = r1["run_id"]
    claim_id = r1["claims"][0]["id"]

    fc_payload = {
        "run_id": run_id,
        "claim_id": claim_id,
        "failure_category": "unclear",
        "researcher_note": "Duplicate test save",
    }
    resp1 = client.post("/failure-cases", json_body=fc_payload)
    resp2 = client.post("/failure-cases", json_body=fc_payload)
    assert resp1.status_code == 200
    assert resp2.status_code == 200

    f1 = resp1.json()["file_path"]
    f2 = resp2.json()["file_path"]
    assert f1 != f2
    assert Path(f1).is_file()
    assert Path(f2).is_file()


def test_save_failure_case_round_trip_load(client: ASGIClient):
    from src.core.trace_io import load_failure_cases

    r1 = client.post("/analyze", json_body={"text": "Marie Curie was born in Warsaw."}).json()
    run_id = r1["run_id"]
    claim_id = r1["claims"][0]["id"]

    unique_note = f"Boundary split issue {run_id}"
    fc_payload = {
        "run_id": run_id,
        "claim_id": claim_id,
        "failure_category": "extraction_error",
        "researcher_note": unique_note,
    }
    resp = client.post("/failure-cases", json_body=fc_payload)
    assert resp.status_code == 200

    cases = load_failure_cases(client.app.state.service.failure_dir)
    matched = [
        c for c in cases
        if c.get("researcher_annotation", {}).get("researcher_note") == unique_note
    ]
    assert len(matched) == 1
    case = matched[0]
    assert case["claim_id"] == claim_id
    assert case["researcher_annotation"]["failure_category"] == "extraction_error"
    assert case["trace"]["run_id"] == run_id


# --------------------------------------------------------------------------- #
# OpenAPI Schema
# --------------------------------------------------------------------------- #

def test_openapi_schema(client: ASGIClient):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200, resp.text
    schema = resp.json()
    paths = schema.get("paths", {})
    for ep in ("/health", "/analyze", "/analyze/oracle", "/verify/quick", "/verify/full", "/annotate", "/reaggregate", "/failure-cases"):
        assert ep in paths, f"Endpoint {ep} missing from OpenAPI paths"
