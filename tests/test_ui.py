"""Unit and integration tests for Nullius Streamlit UI (Step E slice).

Tests verify:
  1. Initial render makes no HTTP request and displays persistent notice.
  2. Blank submission makes no HTTP request and displays local validation error.
  3. Valid submission makes exactly one request with expected endpoint, payload, and timeouts.
  4. Claim selection and explicit reruns make no additional HTTP requests.
  5. Edited input leaves prior results explicitly tied to their submitted snapshot.
  6. A failed subsequent submission does not display stale successful results.
  7. Robust handling of timeout, connection error, non-2xx JSON/non-JSON, invalid JSON, and malformed payload.
  8. Empty claims, missing optional fields, zero scores, and identifier-based joins.
  9. Separate UI sessions do not share state or results.

Uses synthetic schema-consistent fixtures and unittest.mock with streamlit.testing.v1.AppTest.
Requires NO live backend, NO model downloads, and NO GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import requests
from streamlit.testing.v1 import AppTest

from src.ui.app import (
    DEFAULT_BACKEND_URL,
    DEFAULT_TIMEOUT,
    build_endpoint_url,
    format_http_error,
    send_oracle_request,
    send_save_failure_case_request,
    validate_response_payload,
)

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "ui" / "app.py")


def make_synthetic_payload(
    claims: list[dict[str, Any]] | None = None,
    evidence_by_claim: dict[str, list[dict[str, Any]]] | None = None,
    verdicts: list[dict[str, Any]] | None = None,
    timings: dict[str, float] | None = None,
    run_id: str = "20260912T000000Z-test01",
    config_hash: str = "cfg_hash_test",
    git_sha: str | None = "9d01afd",
    response_text: str = "Marie Curie was a physicist.",
    mode: str = "retrieved",
    schema_version: str = "1.0.0",
    resolved_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a valid, schema-consistent synthetic AnalyzeResponse dictionary."""
    if claims is None:
        claims = [
            {
                "id": "clm_01",
                "response_id": "resp_01",
                "text": "Marie Curie was a physicist.",
                "source_span": {"start": 0, "end": 28},
                "extractor_name": "spacy_sentence",
                "extractor_meta": {},
            }
        ]
    if evidence_by_claim is None:
        evidence_by_claim = {
            "clm_01": [
                {
                    "id": "ev::curie::0",
                    "doc_id": "curie",
                    "sent_id": 0,
                    "text": "Marie Curie was a Polish and naturalised-French physicist.",
                    "score": 0.895,
                    "retriever_name": "bm25",
                    "rank": 1,
                    "is_gold": True,
                    "retriever_meta": {},
                }
            ]
        }
    if verdicts is None:
        verdicts = [
            {
                "claim_id": "clm_01",
                "label": "Supported",
                "confidence": 0.942,
                "abstained": False,
                "per_evidence": [
                    {
                        "claim_id": "clm_01",
                        "evidence_id": "ev::curie::0",
                        "p_entail": 0.95,
                        "p_contra": 0.01,
                        "p_neutral": 0.04,
                        "similarity": 0.88,
                        "verifier_name": "nli_deberta",
                        "latency_ms": 42.5,
                        "evidence_rank": 1,
                        "evidence_score": 0.895,
                    }
                ],
                "aggregator_name": "threshold_aggregator",
                "aggregation_trace": {
                    "rule": "max_entailment_above_threshold",
                    "explanation": "p_entail exceeds 0.7 threshold",
                    "decisive_evidence_ids": ["ev::curie::0"],
                },
            }
        ]
    if timings is None:
        timings = {
            "extract_ms": 12.3,
            "retrieve_ms": 45.6,
            "verify_ms": 78.9,
            "aggregate_ms": 2.1,
            "total_ms": 138.9,
        }
    return {
        "run_id": run_id,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "timestamp": "2026-09-12T00:00:00Z",
        "response_text": response_text,
        "mode": mode,
        "claims": claims,
        "evidence_by_claim": evidence_by_claim,
        "verdicts": verdicts,
        "timings": timings,
        "resolved_config": resolved_config or {"k": 3},
        "schema_version": schema_version,
    }


# =========================================================================== #
# Helper Unit Tests
# =========================================================================== #

def test_build_endpoint_url():
    assert build_endpoint_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/analyze"
    assert build_endpoint_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000/analyze"
    assert build_endpoint_url("http://localhost:8000/analyze") == "http://localhost:8000/analyze"
    assert build_endpoint_url("http://example.com/api/") == "http://example.com/api/analyze"


def test_validate_response_payload():
    valid = make_synthetic_payload()
    is_valid, err = validate_response_payload(valid)
    assert is_valid is True
    assert err is None

    # Missing required key
    invalid = dict(valid)
    del invalid["claims"]
    is_valid, err = validate_response_payload(invalid)
    assert is_valid is False
    assert "claims" in str(err)

    # Invalid types
    invalid = dict(valid, claims="not a list")
    is_valid, err = validate_response_payload(invalid)
    assert is_valid is False
    assert "claims" in str(err)


def test_format_http_error_custom_error_response():
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 400
    resp.json.return_value = {
        "error": "ContractError",
        "detail": "Violated check_claims boundary",
        "error_type": "PipelineContractViolation",
    }
    msg = format_http_error(resp)
    assert "HTTP 400" in msg
    assert "ContractError" in msg
    assert "Violated check_claims boundary" in msg
    assert "PipelineContractViolation" in msg


def test_format_http_error_fastapi_validation_list():
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 422
    resp.json.return_value = {
        "detail": [
            {"loc": ["body", "text"], "msg": "Field required", "type": "missing"}
        ]
    }
    msg = format_http_error(resp)
    assert "HTTP 422" in msg
    assert "body -> text" in msg
    assert "Field required" in msg


def test_format_http_error_non_json():
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 502
    resp.json.side_effect = ValueError("No JSON")
    resp.text = "Bad Gateway from reverse proxy"
    msg = format_http_error(resp)
    assert "HTTP 502: Bad Gateway from reverse proxy" in msg


# =========================================================================== #
# AppTest Integration Tests
# =========================================================================== #

def test_1_initial_render_makes_no_http_request():
    """1. Initial render must make zero requests and show persistent notice."""
    with patch("requests.post") as mock_post:
        at = AppTest.from_file(APP_PATH).run()
        assert mock_post.call_count == 0
        assert len(at.error) == 0
        # Persistent notice should be rendered
        assert any("Research instrument" in w.value for w in at.warning)


def test_2_blank_submission_makes_no_request():
    """2. Blank submission makes no request and rejects locally."""
    with patch("requests.post") as mock_post:
        at = AppTest.from_file(APP_PATH).run()
        # Set text area to whitespace only
        at.text_area[0].input("   \n\t  ").run()
        at.button[0].click().run()

        assert mock_post.call_count == 0
        assert len(at.error) > 0
        assert any("cannot be blank" in e.value for e in at.error)


def test_3_valid_submission_makes_requests_correctly():
    """3. Valid submission issues one /analyze request and reaggregates."""
    payload = make_synthetic_payload(response_text="Marie Curie was a physicist.")

    with patch("requests.post") as mock_post:
        import streamlit as st
        st.cache_data.clear()
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        mock_post.return_value = mock_resp

        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Marie Curie was a physicist.").run()
        at.button[0].click().run()

        assert mock_post.call_count == 2
        args1, kwargs1 = mock_post.call_args_list[0]
        assert args1[0] == f"{DEFAULT_BACKEND_URL}/analyze"
        assert kwargs1["json"] == {"text": "Marie Curie was a physicist."}
        assert kwargs1["timeout"] == DEFAULT_TIMEOUT

        args2, kwargs2 = mock_post.call_args_list[1]
        assert args2[0] == f"{DEFAULT_BACKEND_URL}/reaggregate"
        assert kwargs2["json"]["run_id"] == "20260912T000000Z-test01"
        assert kwargs2["json"]["target_aggregators"] == ["max_entailment", "noisy_or", "majority", "threshold_abstain", "weighted_by_retrieval"]

        # Results should be rendered without errors
        assert len(at.error) == 0
        assert len(at.selectbox) >= 1
        assert at.selectbox[0].value == "clm_01"


def test_4_claim_selection_and_rerun_make_no_extra_requests():
    """4. Claim selection and explicit reruns make no additional HTTP requests."""
    claims = [
        {"id": "c1", "text": "Claim One", "response_id": "r1", "source_span": None, "extractor_name": "ext"},
        {"id": "c2", "text": "Claim Two", "response_id": "r1", "source_span": None, "extractor_name": "ext"},
    ]
    evidence_by_claim = {"c1": [], "c2": []}
    verdicts = [
        {"claim_id": "c1", "label": "Supported", "confidence": 0.9, "abstained": False, "per_evidence": [], "aggregator_name": "agg", "aggregation_trace": {}},
        {"claim_id": "c2", "label": "Contradicted", "confidence": 0.8, "abstained": False, "per_evidence": [], "aggregator_name": "agg", "aggregation_trace": {}},
    ]
    payload = make_synthetic_payload(claims=claims, evidence_by_claim=evidence_by_claim, verdicts=verdicts)

    with patch("requests.post") as mock_post:
        import streamlit as st
        st.cache_data.clear()
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        mock_post.return_value = mock_resp

        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Input text").run()
        at.button[0].click().run()
        
        # 1 analyze + 1 batched reaggregate
        assert mock_post.call_count == 2

        # Select second claim in selectbox
        at.selectbox[0].select("c2").run()
        assert at.selectbox[0].value == "c2"
        # Reaggregation is cached by run_id (claim_id is no longer a cache key)
        # So no new HTTP calls should happen!
        assert mock_post.call_count == 2

        # Explicit rerun without click
        at.run()
        # Should be cached, so no more calls
        assert mock_post.call_count == 2


def test_5_edited_input_leaves_prior_results_tied_to_snapshot():
    """5. Edited input leaves prior results explicitly tied to their submitted snapshot."""
    payload = make_synthetic_payload(response_text="Original text.")

    with patch("requests.post") as mock_post:
        import streamlit as st
        st.cache_data.clear()
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        mock_post.return_value = mock_resp

        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Original text.").run()
        at.button[0].click().run()
        
        assert mock_post.call_count == 2

        # Edit text without clicking Run
        at.text_area[0].input("Edited text.").run()
        # Cache hits, no additional requests
        assert mock_post.call_count == 2

        # Should display warning indicating input text differs from snapshot
        warning_texts = [w.value for w in at.warning]
        assert any("modified since last run" in w for w in warning_texts)
        assert any("Original text." in c.value for c in at.caption)


def test_6_failed_subsequent_submission_clears_stale_results():
    """6. A failed subsequent submission does not display stale successful results."""
    payload = make_synthetic_payload(response_text="Valid initial run.")

    with patch("requests.post") as mock_post:
        # First call succeeds
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = payload
        
        call_tracker = {"count": 0}

        def mock_post_side_effect(*args, **kwargs):
            call_tracker["count"] += 1
            url = args[0]
            if url.endswith("/analyze"):
                if call_tracker["count"] == 1:
                    return resp1
                else:
                    raise requests.exceptions.ConnectionError("Connection refused")
            
            # For reaggregate
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"comparisons": {}}
            return resp

        mock_post.side_effect = mock_post_side_effect

        import streamlit as st
        st.cache_data.clear()

        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Initial text").run()
        at.button[0].click().run()
        
        # 1 analyze + 1 reaggregate
        assert mock_post.call_count == 2
        assert len(at.selectbox) >= 1

        # Now submit again, which triggers failure (the second analyze call)
        at.button[0].click().run()
        # One more call to /analyze (which fails, stopping UI from making more reaggregate calls)
        assert mock_post.call_count == 3
        # Error must be displayed
        assert len(at.error) > 0
        assert any("Failed to connect" in e.value for e in at.error)
        # Previous successful result controls (like claim selector) must be cleared
        assert len(at.selectbox) == 0


def test_7_error_handling():
    """7. Handle timeout, connection error, non-2xx JSON/non-JSON, invalid JSON, malformed payload."""
    # 7a. Timeout (with explicit statement that backend is not cancelled)
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("Read timeout")
        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Text").run()
        at.button[0].click().run()
        assert len(at.error) > 0
        err_msg = at.error[0].value
        assert "timed out" in err_msg
        assert "does not imply that the backend has cancelled its execution" in err_msg

    # 7b. Connection error
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Refused")
        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Text").run()
        at.button[0].click().run()
        assert len(at.error) > 0
        assert "Failed to connect to backend" in at.error[0].value

    # 7c. Non-2xx with JSON detail
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {
            "error": "ContractError",
            "detail": "Failed contract check",
            "error_type": "PipelineContractViolation",
        }
        mock_post.return_value = mock_resp
        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Text").run()
        at.button[0].click().run()
        assert len(at.error) > 0
        assert "HTTP 400" in at.error[0].value
        assert "Failed contract check" in at.error[0].value

    # 7d. Non-2xx with non-JSON text
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.side_effect = ValueError("Not JSON")
        mock_resp.text = "Bad Gateway"
        mock_post.return_value = mock_resp
        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Text").run()
        at.button[0].click().run()
        assert len(at.error) > 0
        assert "HTTP 502: Bad Gateway" in at.error[0].value

    # 7e. 200 with invalid JSON
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON in body")
        mock_post.return_value = mock_resp
        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Text").run()
        at.button[0].click().run()
        assert len(at.error) > 0
        assert "not valid JSON" in at.error[0].value

    # 7f. 200 with malformed payload structure (missing required keys)
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"random_key": 123}
        mock_post.return_value = mock_resp
        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Text").run()
        at.button[0].click().run()
        assert len(at.error) > 0
        assert "Malformed successful response" in at.error[0].value


def test_8_empty_claims_missing_optionals_zero_scores_and_identifier_joins():
    """8. Handle empty claims, missing optional fields, zero scores, and identifier-based joins."""
    # 8a. Empty claims is valid empty result
    empty_payload = make_synthetic_payload(claims=[], evidence_by_claim={}, verdicts=[])
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = empty_payload
        mock_post.return_value = mock_resp

        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("No claims here.").run()
        at.button[0].click().run()
        assert len(at.error) == 0
        assert any("No claims extracted" in info.value for info in at.info)

    # 8b & 8c & 8d. Missing optional fields, zero scores, identifier joins
    claims = [
        {"id": "clm_A", "text": "Claim A", "response_id": "r1", "source_span": None, "extractor_name": "ext"},
        {"id": "clm_B", "text": "Claim B", "response_id": "r1", "source_span": None, "extractor_name": "ext"},
    ]
    # Intentionally reverse list order of verdicts to verify ID join
    verdicts = [
        {
            "claim_id": "clm_B",
            "label": "Contradicted",
            "confidence": 0.0,  # Zero score preserved
            "abstained": False,
            "per_evidence": [
                {
                    "claim_id": "clm_B",
                    "evidence_id": "ev::b::1",
                    "p_entail": 0.0,  # Zero score preserved
                    "p_contra": 1.0,
                    "p_neutral": 0.0,  # Zero score preserved
                    "similarity": 0.0,  # Zero score preserved
                    "verifier_name": "nli",
                    "latency_ms": 0.0,
                    "evidence_rank": None,  # Optional field None
                    "evidence_score": None,  # Optional field None
                }
            ],
            "aggregator_name": "agg",
            "aggregation_trace": {
                "rule": "strict_contra",
                "explanation": "Decisive contradiction found",
                "decisive_evidence_ids": ["ev::b::1"],
            },
        },
        {
            "claim_id": "clm_A",
            "label": "Supported",
            "confidence": 0.99,
            "abstained": False,
            "per_evidence": [],
            "aggregator_name": "agg",
            "aggregation_trace": {"rule": "all_pass", "explanation": "ok", "decisive_evidence_ids": []},
        },
    ]
    evidence_by_claim = {
        "clm_A": [],
        "clm_B": [
            {
                "id": "ev::b::1",
                "doc_id": "doc_b",
                "sent_id": 1,
                "text": "Sentence B contradicts.",
                "score": 0.0,  # Zero score preserved
                "retriever_name": "bm25",
                "rank": 1,
                "is_gold": None,  # Optional field None
                "retriever_meta": {},
            }
        ],
    }
    payload = make_synthetic_payload(
        claims=claims,
        verdicts=verdicts,
        evidence_by_claim=evidence_by_claim,
        git_sha=None,  # Optional field None
    )

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        mock_post.return_value = mock_resp

        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Claim A. Claim B.").run()
        at.button[0].click().run()

        # Select clm_B
        at.selectbox[0].select("clm_B").run()

        # Verify clm_B matched with Contradicted verdict by ID (not clm_A)
        metric_values = [m.value for m in at.metric]
        assert "Contradicted" in metric_values
        # Verify 0.0000 score is preserved in metrics
        assert "0.0000" in metric_values


def test_9_separate_ui_sessions_do_not_share_results():
    """9. Separate UI sessions do not share state or results."""
    payload = make_synthetic_payload(response_text="Session 1 text.")

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        mock_post.return_value = mock_resp

        # Session 1 runs and populates results
        at1 = AppTest.from_file(APP_PATH).run()
        at1.text_area[0].input("Session 1 text.").run()
        at1.button[0].click().run()
        assert len(at1.selectbox) >= 1
        assert at1.session_state.result is not None

        # Session 2 initializes independently
        at2 = AppTest.from_file(APP_PATH).run()
        assert at2.session_state.result is None
        assert at2.session_state.submitted_text is None
        assert len(at2.selectbox) == 0


def test_10_reaggregate_comparison():
    """10. Compare aggregators sends /reaggregate requests efficiently."""
    payload = make_synthetic_payload(response_text="Text.")

    def mock_post_side_effect(*args, **kwargs):
        url = args[0]
        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/analyze"):
            resp.json.return_value = payload
        elif url.endswith("/reaggregate"):
            comparisons = {}
            for agg in kwargs["json"]["target_aggregators"]:
                comparisons[agg] = {
                    "claim_id": "clm_01",
                    "label": "Supported",
                    "confidence": 0.99,
                    "abstained": False,
                    "per_evidence": [],
                    "aggregator_name": agg,
                    "aggregation_trace": {"rule": "test rule", "decisive_evidence_ids": ["ev_1"]}
                }
            resp.json.return_value = {
                "run_id": kwargs["json"]["run_id"],
                "comparisons": {
                    "clm_01": comparisons
                }
            }
        return resp

    with patch("requests.post") as mock_post:
        import streamlit as st
        st.cache_data.clear()

        mock_post.side_effect = mock_post_side_effect
        
        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Text.").run()
        at.button[0].click().run()
        
        # 1 call to /analyze, and 1 call to /reaggregate
        # Because we expand the expander or it's run as part of the script
        assert mock_post.call_count == 2
        
        # Rerunning the app (e.g. changing claim selection) should not trigger /reaggregate again due to cache
        at.run()
        assert mock_post.call_count == 2
        
        # Verify the dataframe renders
        assert len(at.dataframe) > 0
        df = at.dataframe[0].value
        assert len(df) == 5  # 5 baseline aggregators
        assert "Aggregator" in df.columns
        assert "Label" in df.columns


def test_send_oracle_request_helpers():
    """Test send_oracle_request client function under success, HTTP error, and connection failure."""
    # 1. Success
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"mode": "oracle", "run_id": "oracle_123"}
        mock_post.return_value = mock_resp

        data, err = send_oracle_request("http://127.0.0.1:8000", "Sample text")
        assert err is None
        assert data["mode"] == "oracle"
        assert mock_post.call_args[1]["json"] == {"response_text": "Sample text"}

    # 2. HTTP 404
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"detail": "Example not found"}
        mock_post.return_value = mock_resp

        data, err = send_oracle_request("http://127.0.0.1:8000", "Unknown text")
        assert data is None
        assert "HTTP 404" in err

    # 3. ConnectionError
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        data, err = send_oracle_request("http://127.0.0.1:8000", "Text")
        assert data is None
        assert "Oracle request failed" in err


def test_send_save_failure_case_request_helpers():
    """Test send_save_failure_case_request client function under success, HTTP error, and connection failure."""
    payload = {
        "run_id": "run_123",
        "claim_id": "clm_01",
        "failure_category": "extraction_error",
        "researcher_note": "Boundary split issue",
    }
    # 1. Success
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "failure_case_id": "fc_test",
            "file_path": "results/failure_cases/test.json",
            "status": "saved",
        }
        mock_post.return_value = mock_resp

        data, err = send_save_failure_case_request("http://127.0.0.1:8000", payload)
        assert err is None
        assert data["status"] == "saved"
        assert mock_post.call_args[1]["json"] == payload

    # 2. HTTP 422
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {"detail": "Validation error"}
        mock_post.return_value = mock_resp

        data, err = send_save_failure_case_request("http://127.0.0.1:8000", payload)
        assert data is None
        assert "HTTP 422" in err

    # 3. ConnectionError
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        data, err = send_save_failure_case_request("http://127.0.0.1:8000", payload)
        assert data is None
        assert "Save failure case request failed" in err


def test_11_failure_case_ui_flow():
    """11. UI failure case recording and single-request invariant."""
    payload = make_synthetic_payload(
        response_text="Marie Curie was a physicist.",
        resolved_config={"has_gold_evidence": True, "example_id": "mini-001", "gold_label": "Supported"}
    )

    fc_calls = []

    def mock_post_side_effect(*args, **kwargs):
        url = args[0]
        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/analyze"):
            resp.json.return_value = payload
        elif url.endswith("/reaggregate"):
            resp.json.return_value = {
                "run_id": kwargs["json"]["run_id"],
                "comparisons": {"clm_01": {}}
            }
        elif url.endswith("/failure-cases"):
            fc_calls.append(kwargs["json"])
            resp.json.return_value = {
                "failure_case_id": "fc_test_01",
                "file_path": "results/failure_cases/20260912-clm_01.json",
                "status": "saved",
            }
        return resp

    with patch("requests.post") as mock_post:
        import streamlit as st
        st.cache_data.clear()

        mock_post.side_effect = mock_post_side_effect

        at = AppTest.from_file(APP_PATH).run()
        at.text_area[0].input("Marie Curie was a physicist.").run()
        at.button[0].click().run()

        # Find Save Failure Case button
        save_buttons = [b for b in at.button if "Save Failure Case" in b.label]
        assert len(save_buttons) == 1
        save_btn = save_buttons[0]

        # Disabled initially because note is blank
        assert save_btn.disabled is True

        # Enter researcher note in the note text area (second text area in page)
        note_areas = [ta for ta in at.text_area if "Researcher Note" in ta.label]
        assert len(note_areas) == 1
        note_areas[0].input("Observed overconfidence on distractor hit").run()

        # Save button is now enabled
        save_buttons = [b for b in at.button if "Save Failure Case" in b.label]
        assert save_buttons[0].disabled is False

        # Click save
        save_buttons[0].click().run()

        # Check that POST /failure-cases was called exactly once
        assert len(fc_calls) == 1
        assert fc_calls[0]["run_id"] == payload["run_id"]
        assert fc_calls[0]["claim_id"] == "clm_01"
        assert fc_calls[0]["failure_category"] == "unclear"
        assert fc_calls[0]["researcher_note"] == "Observed overconfidence on distractor hit"

        # Rerun UI - must issue ZERO additional requests to /failure-cases
        at.run()
        assert len(fc_calls) == 1

        # Test edited input guard: changing response text in editor must disable save
        at.text_area[0].input("Edited text that was not analyzed").run()
        save_buttons = [b for b in at.button if "Save Failure Case" in b.label]
        assert save_buttons[0].disabled is True

