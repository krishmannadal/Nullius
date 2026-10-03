"""Nullius Streamlit Inspection Harness (Step E: First Bounded Slice).

A thin, decoupled HTTP client for the existing POST /analyze endpoint.
Provides white-box inspection of claims, evidence, pairwise scores,
aggregation rationale, timings, provenance, and raw traces.

Contracts & Constraints:
  - Consumes existing FastAPI POST /analyze over HTTP; does NOT import
    pipeline, service, or model components.
  - Exactly one HTTP request per explicit user submission.
  - Zero server-side inference on claim selection or rendering-only reruns.
  - Session-isolated state; never caches user traces globally.
  - Strictly preserves uncalibrated score semantics and avoids response-level truth claims.
"""

from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st

DEFAULT_BACKEND_URL: str = "http://127.0.0.1:8000"
DEFAULT_CONNECT_TIMEOUT: float = 5.0
DEFAULT_READ_TIMEOUT: float = 120.0
DEFAULT_TIMEOUT: tuple[float, float] = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)

REQUIRED_PAYLOAD_KEYS = ("claims", "evidence_by_claim", "verdicts")


def build_endpoint_url(base_url: str) -> str:
    """Consistently construct the /analyze endpoint URL from a base URL."""
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/analyze"):
        cleaned = cleaned[:-len("/analyze")].rstrip("/")
    return f"{cleaned}/analyze"


def format_http_error(resp: requests.Response) -> str:
    """Extract descriptive error message from HTTP response without crashing."""
    status_code = resp.status_code
    try:
        data = resp.json()
    except Exception:
        text = resp.text.strip()
        body_snippet = text[:300] if text else "Empty response body"
        return f"HTTP {status_code}: {body_snippet}"

    if isinstance(data, dict):
        error_name = data.get("error")
        error_type = data.get("error_type")
        detail = data.get("detail")

        parts: list[str] = []
        if error_name:
            parts.append(f"Error: {error_name}")
        if error_type:
            parts.append(f"Type: {error_type}")

        if isinstance(detail, list):
            detail_strings = []
            for item in detail:
                if isinstance(item, dict):
                    loc = " -> ".join(str(x) for x in item.get("loc", []))
                    msg = item.get("msg", "")
                    detail_strings.append(f"[{loc}] {msg}" if loc else msg)
                else:
                    detail_strings.append(str(item))
            parts.append("Detail: " + "; ".join(detail_strings))
        elif detail is not None:
            parts.append(f"Detail: {detail}")

        if parts:
            return f"HTTP {status_code} — " + " | ".join(parts)
        return f"HTTP {status_code}: {json.dumps(data)}"

    return f"HTTP {status_code}: {data}"


def validate_response_payload(data: Any) -> tuple[bool, str | None]:
    """Validate that successful response adheres to required schema structure."""
    if not isinstance(data, dict):
        return False, "Response payload is not a JSON object."
    missing = [k for k in REQUIRED_PAYLOAD_KEYS if k not in data]
    if missing:
        return False, f"Missing required fields in payload: {missing}"
    if not isinstance(data["claims"], list):
        return False, "Field 'claims' must be a list."
    if not isinstance(data["evidence_by_claim"], dict):
        return False, "Field 'evidence_by_claim' must be an object (dict)."
    if not isinstance(data["verdicts"], list):
        return False, "Field 'verdicts' must be a list."
    return True, None


def send_analyze_request(
    base_url: str,
    text: str,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> tuple[dict[str, Any] | None, str | None]:
    """Send exactly one POST /analyze request with bounded timeouts."""
    endpoint_url = build_endpoint_url(base_url)
    try:
        resp = requests.post(
            endpoint_url,
            json={"text": text},
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        return None, (
            f"Request to {endpoint_url} timed out (connect={timeout[0]}s, read={timeout[1]}s). "
            "Note: A timeout does not imply that the backend has cancelled its execution; "
            "the pipeline may still be running in the background."
        )
    except requests.exceptions.ConnectionError:
        return None, (
            f"Failed to connect to backend at {endpoint_url}. "
            "Please ensure the FastAPI service is running."
        )
    except requests.exceptions.RequestException as exc:
        return None, f"HTTP request failed: {exc}"

    if resp.status_code != 200:
        return None, format_http_error(resp)

    try:
        data = resp.json()
    except Exception as exc:
        return None, f"Backend returned HTTP 200, but response body is not valid JSON: {exc}"

    is_valid, validation_err = validate_response_payload(data)
    if not is_valid:
        return None, f"Malformed successful response: {validation_err}"

    return data, None


@st.cache_data(show_spinner=False, ttl=3600)
def send_reaggregate_request(
    base_url: str,
    run_id: str,
    target_aggregators: list[str],
    timeout: float = 10.0,
) -> tuple[dict[str, Any] | None, str | None]:
    """Hit the backend /reaggregate endpoint to run multiple pure aggregation steps."""
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/analyze"):
        cleaned = cleaned[:-len("/analyze")].rstrip("/")
    endpoint_url = f"{cleaned}/reaggregate"
    
    payload = {
        "run_id": run_id,
        "target_aggregators": target_aggregators,
        "aggregator_configs": {},
    }
    try:
        resp = requests.post(endpoint_url, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        return None, f"Reaggregation request failed: {exc}"

    if resp.status_code != 200:
        return None, format_http_error(resp)

    try:
        data = resp.json()
        return data.get("comparisons"), None
    except Exception as exc:
        return None, f"Invalid JSON response: {exc}"


def send_oracle_request(
    base_url: str,
    response_text: str,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> tuple[dict[str, Any] | None, str | None]:
    """Hit the backend /analyze/oracle endpoint with response text."""
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/analyze"):
        cleaned = cleaned[:-len("/analyze")].rstrip("/")
    endpoint_url = f"{cleaned}/analyze/oracle"

    try:
        resp = requests.post(
            endpoint_url,
            json={"response_text": response_text},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        return None, f"Oracle request failed: {exc}"

    if resp.status_code != 200:
        return None, format_http_error(resp)

    try:
        data = resp.json()
        return data, None
    except ValueError as exc:
        return None, f"Invalid JSON response from oracle endpoint: {exc}"


def send_save_failure_case_request(
    base_url: str,
    payload: dict[str, Any],
    timeout: float = 10.0,
) -> tuple[dict[str, Any] | None, str | None]:
    """Submit a structured failure case to POST /failure-cases."""
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/analyze"):
        cleaned = cleaned[:-len("/analyze")].rstrip("/")
    endpoint_url = f"{cleaned}/failure-cases"

    try:
        resp = requests.post(
            endpoint_url,
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        return None, f"Save failure case request failed: {exc}"

    if resp.status_code != 200:
        return None, format_http_error(resp)

    try:
        data = resp.json()
        return data, None
    except ValueError as exc:
        return None, f"Invalid JSON response from failure-cases endpoint: {exc}"


def main() -> None:
    st.set_page_config(page_title="Nullius Inspection Harness", layout="wide")

    # Persistent research instrument warning notice
    st.warning(
        "⚠️ **Research instrument**: debug corpora, untuned defaults, no scientific validation. "
        "Nullius is an inspection harness for claim-level verification analysis, "
        "not a validated production hallucination detector."
    )

    st.title("Nullius — White-Box Inspection Harness")

    # Sidebar: Backend configuration
    st.sidebar.header("Backend Configuration")
    backend_base_url = st.sidebar.text_input(
        "Backend Base URL",
        value=DEFAULT_BACKEND_URL,
        help="Base URL of the FastAPI backend service (e.g. http://127.0.0.1:8000)",
    )

    # Session state initialization
    if "result" not in st.session_state:
        st.session_state.result = None
    if "submitted_text" not in st.session_state:
        st.session_state.submitted_text = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "oracle_result" not in st.session_state:
        st.session_state.oracle_result = None
    if "oracle_error" not in st.session_state:
        st.session_state.oracle_error = None
    if "last_save_success" not in st.session_state:
        st.session_state.last_save_success = None
    if "last_save_error" not in st.session_state:
        st.session_state.last_save_error = None
    if "reaggregate_comparisons" not in st.session_state:
        st.session_state.reaggregate_comparisons = {}

    # Input section
    input_text = st.text_area(
        "Response Text",
        value="",
        height=140,
        placeholder="Enter model response text to extract claims and verify...",
        help="Enter the response text to inspect.",
        key="response_text_input",
    )

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        run_clicked = st.button("Run Analysis", type="primary")

    # Submission handling
    if run_clicked:
        # Clear previous results and errors on new submission attempt
        st.session_state.result = None
        st.session_state.submitted_text = None
        st.session_state.last_error = None
        st.session_state.oracle_result = None
        st.session_state.oracle_error = None
        st.session_state.last_save_success = None
        st.session_state.last_save_error = None
        st.session_state.reaggregate_comparisons = {}

        cleaned_text = input_text.strip()
        if not cleaned_text:
            st.session_state.last_error = "Response text cannot be blank."
        else:
            with st.spinner("Executing pipeline via FastAPI backend..."):
                data, err = send_analyze_request(backend_base_url, input_text)
                if err:
                    st.session_state.last_error = err
                else:
                    st.session_state.result = data
                    st.session_state.submitted_text = input_text

    # Error presentation
    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    # Result presentation
    result = st.session_state.result
    submitted_snapshot = st.session_state.submitted_text

    if result is not None and submitted_snapshot is not None:
        st.markdown("---")
        # Explicit input association check
        if input_text != submitted_snapshot:
            st.warning(
                "⚠️ **Input text in editor has been modified since last run.** "
                "The results displayed below correspond strictly to the submitted snapshot."
            )
            st.caption(f"Submitted Snapshot: *\"{submitted_snapshot}\"*")
        else:
            st.caption(f"Results for submitted snapshot: *\"{submitted_snapshot}\"*")

        claims = result.get("claims", [])
        evidence_by_claim = result.get("evidence_by_claim", {})
        verdicts = result.get("verdicts", [])

        # Build identifier-based lookup maps (never rely on list positions)
        claim_by_id = {
            c["id"]: c for c in claims if isinstance(c, dict) and "id" in c
        }
        verdict_by_claim_id = {
            v["claim_id"]: v for v in verdicts if isinstance(v, dict) and "claim_id" in v
        }

        st.subheader("Claim-Level Inspection")

        if not claims:
            st.info("No claims extracted from the submitted response text.")
        else:
            claim_ids = [c["id"] for c in claims if isinstance(c, dict) and "id" in c]
            selected_claim_id = st.selectbox(
                "Select Claim to Inspect",
                options=claim_ids,
                format_func=lambda cid: f"[{cid}] {claim_by_id[cid].get('text', '')}",
                key="claim_selector",
            )

            if selected_claim_id and selected_claim_id in claim_by_id:
                claim = claim_by_id[selected_claim_id]
                claim_text = claim.get("text", "")
                extractor_name = claim.get("extractor_name", "unknown")
                source_span = claim.get("source_span")

                st.markdown(f"#### Claim: *\"{claim_text}\"*")
                span_info = "Source Span: None"
                if isinstance(source_span, dict) and "start" in source_span and "end" in source_span:
                    s_start = source_span.get("start")
                    s_end = source_span.get("end")
                    if isinstance(s_start, int) and isinstance(s_end, int):
                        span_info = f"Source Span: character offsets `[{s_start}, {s_end})` in submitted text"
                        if 0 <= s_start <= s_end <= len(submitted_snapshot):
                            span_text = submitted_snapshot[s_start:s_end]
                            st.caption(f"{span_info} (Text: *\"{span_text}\"*)")
                        else:
                            st.caption(span_info)
                else:
                    st.caption(span_info)

                st.caption(f"Claim ID: `{selected_claim_id}` | Extractor: `{extractor_name}`")

                # Claim Verdict
                verdict = verdict_by_claim_id.get(selected_claim_id)
                if verdict is None:
                    st.info("No verdict available for this claim.")
                else:
                    label = verdict.get("label", "Unknown")
                    confidence = verdict.get("confidence")
                    abstained = verdict.get("abstained", False)
                    aggregator_name = verdict.get("aggregator_name", "unknown")

                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric("Claim Verdict", label)
                    with m_col2:
                        conf_val = (
                            f"{float(confidence):.4f}"
                            if isinstance(confidence, (int, float))
                            else "N/A"
                        )
                        st.metric("Aggregator Score (Uncalibrated)", conf_val)
                    with m_col3:
                        st.metric("Abstained", "Yes" if abstained else "No")

                    st.caption(f"Aggregator: `{aggregator_name}`")

                    # Aggregation Rationale (actual locations in schema)
                    agg_trace = verdict.get("aggregation_trace")
                    if isinstance(agg_trace, dict):
                        rule = agg_trace.get("rule")
                        explanation = agg_trace.get("explanation")
                        decisive_ids = agg_trace.get("decisive_evidence_ids")

                        st.markdown("**Aggregation Rationale:**")
                        if rule is not None:
                            st.markdown(f"- **Rule:** `{rule}`")
                        if explanation is not None:
                            st.markdown(f"- **Explanation:** {explanation}")
                        if decisive_ids is not None:
                            st.markdown(f"- **Decisive Evidence IDs:** `{decisive_ids}`")

                        # Fallback / additional details
                        other_keys = {
                            k: v
                            for k, v in agg_trace.items()
                            if k not in ("rule", "explanation", "decisive_evidence_ids")
                        }
                        if other_keys or (rule is None and explanation is None and decisive_ids is None):
                            with st.expander("Raw Aggregation Trace Details"):
                                st.json(agg_trace)

                    # Evidence & Pairwise Verification
                    evidence_list = evidence_by_claim.get(selected_claim_id, [])
                    per_evidence_list = verdict.get("per_evidence", [])
                    per_evidence_by_id = {
                        pe["evidence_id"]: pe
                        for pe in per_evidence_list
                        if isinstance(pe, dict) and "evidence_id" in pe
                    }

                    st.markdown("##### Retrieved Evidence & Pairwise Verification")
                    if not evidence_list:
                        st.info("No evidence retrieved for this claim.")
                    else:
                        for ev in evidence_list:
                            if not isinstance(ev, dict):
                                continue
                            ev_id = ev.get("id", "unknown")
                            ev_text = ev.get("text", "")
                            ev_rank = ev.get("rank")
                            ev_score = ev.get("score")
                            retriever_name = ev.get("retriever_name", "unknown")
                            is_gold = ev.get("is_gold")

                            pv = per_evidence_by_id.get(ev_id)
                            score_str = (
                                f"{float(ev_score):.4f}"
                                if isinstance(ev_score, (int, float))
                                else "N/A"
                            )
                            rank_str = f"#{ev_rank}" if ev_rank is not None else ""

                            with st.expander(f"Evidence {rank_str} `{ev_id}` (Score: {score_str})", expanded=False):
                                st.markdown(f"> {ev_text}")
                                meta_c1, meta_c2, meta_c3 = st.columns(3)
                                with meta_c1:
                                    st.write(f"**Doc ID:** `{ev.get('doc_id')}`")
                                    st.write(f"**Sent ID:** `{ev.get('sent_id')}`")
                                with meta_c2:
                                    st.write(f"**Retriever / Reranker:** `{retriever_name}`")
                                    st.write(f"**Evidence Score:** `{score_str}` *(raw or reranked)*")
                                with meta_c3:
                                    if is_gold is not None:
                                        st.write(f"**Gold Annotation:** `{'Yes' if is_gold else 'No'}`")

                                # Pairwise results
                                if pv is not None:
                                    st.markdown("---")
                                    st.markdown("**Pairwise Verification Scores:**")
                                    p_entail = pv.get("p_entail")
                                    p_contra = pv.get("p_contra")
                                    p_neutral = pv.get("p_neutral")
                                    similarity = pv.get("similarity")

                                    pv_cols = st.columns(4)
                                    # Preserve zero values explicitly (p_val is not None)
                                    if p_entail is not None:
                                        with pv_cols[0]:
                                            st.metric("p(entail)", f"{float(p_entail):.4f}")
                                    if p_contra is not None:
                                        with pv_cols[1]:
                                            st.metric("p(contra)", f"{float(p_contra):.4f}")
                                    if p_neutral is not None:
                                        with pv_cols[2]:
                                            st.metric("p(neutral)", f"{float(p_neutral):.4f}")
                                    if similarity is not None:
                                        with pv_cols[3]:
                                            st.metric("Similarity", f"{float(similarity):.4f}")

                                    pv_verifier = pv.get("verifier_name", "unknown")
                                    pv_lat = pv.get("latency_ms")
                                    lat_str = (
                                        f" | Latency: `{float(pv_lat):.2f} ms`"
                                        if isinstance(pv_lat, (int, float))
                                        else ""
                                    )
                                    st.caption(f"Verifier: `{pv_verifier}`{lat_str}")
                                else:
                                    st.caption("No pairwise verdict recorded for this evidence item.")
                                    
                    # Compare Aggregators
                    with st.expander("Compare Aggregators (Stateless Reaggregation)", expanded=False):
                        st.markdown("Comparing alternative aggregation rules over identical materialized pairwise verdicts.")
                        
                        baseline_aggregators = ["max_entailment", "noisy_or", "majority", "threshold_abstain", "weighted_by_retrieval"]
                        run_id = result.get("run_id")
                        comp_data = []
                        
                        if not run_id:
                            st.error("No run_id available for reaggregation.")
                        else:
                            # Use a spinner while fetching reaggregation
                            with st.spinner("Reaggregating..."):
                                comparisons_dict, reagg_err = send_reaggregate_request(
                                    backend_base_url,
                                    run_id=run_id,
                                    target_aggregators=baseline_aggregators,
                                )
                                
                                if reagg_err:
                                    st.error(reagg_err)
                                elif comparisons_dict and selected_claim_id in comparisons_dict:
                                    st.session_state.reaggregate_comparisons = comparisons_dict
                                    claim_comparisons = comparisons_dict[selected_claim_id]
                                    for agg_name in baseline_aggregators:
                                        reagg_verdict = claim_comparisons.get(agg_name)
                                        if not reagg_verdict:
                                            comp_data.append({
                                                "Aggregator": agg_name,
                                                "Label": "Error",
                                                "Score": "N/A",
                                                "Rule": "Missing from response",
                                                "Decisive IDs": "",
                                            })
                                        else:
                                            comp_data.append({
                                                "Aggregator": agg_name,
                                                "Label": reagg_verdict.get("label", "Unknown"),
                                                "Score": f"{float(reagg_verdict.get('confidence', 0.0)):.4f}",
                                                "Rule": reagg_verdict.get("aggregation_trace", {}).get("rule", ""),
                                                "Decisive IDs": ", ".join(reagg_verdict.get("aggregation_trace", {}).get("decisive_evidence_ids", [])),
                                            })
                        
                            if comp_data:
                                st.dataframe(comp_data, use_container_width=True)

                    # Oracle Comparison (available when response is associated with an example that has gold evidence)
                    resolved_cfg = result.get("resolved_config", {})
                    has_gold = resolved_cfg.get("has_gold_evidence", False)
                    example_id = resolved_cfg.get("example_id")

                    with st.expander("Oracle Comparison (Gold Evidence Substitution)", expanded=False):
                        if not has_gold:
                            st.info(
                                "Oracle comparison is available only when the response is associated with an annotated dataset example with gold evidence. "
                                "The current response does not match a gold example."
                            )
                        else:
                            st.markdown(
                                f"**Dataset Example:** `{example_id}` | **Annotated Gold Label:** `{resolved_cfg.get('gold_label', 'Unknown')}`"
                            )
                            st.markdown(
                                "Runs verification with retrieved evidence replaced by the example's annotated gold evidence. "
                                "Isolates retrieval failure from verifier failure."
                            )

                            run_oracle_btn = st.button("Run Oracle Verification", key=f"run_oracle_{selected_claim_id}")
                            if run_oracle_btn:
                                with st.spinner("Executing oracle verification..."):
                                    o_data, o_err = send_oracle_request(backend_base_url, submitted_snapshot)
                                    if o_err:
                                        st.session_state.oracle_error = o_err
                                        st.session_state.oracle_result = None
                                    else:
                                        st.session_state.oracle_result = o_data
                                        st.session_state.oracle_error = None

                            if st.session_state.get("oracle_error"):
                                st.error(st.session_state.oracle_error)

                            oracle_res = st.session_state.get("oracle_result")
                            if oracle_res:
                                o_verdicts = oracle_res.get("verdicts", [])
                                o_verdict = next((v for v in o_verdicts if v.get("claim_id") == selected_claim_id), None)

                                if o_verdict:
                                    o_col1, o_col2 = st.columns(2)
                                    with o_col1:
                                        st.metric("Retrieved Condition Verdict", verdict.get("label", "Unknown"))
                                    with o_col2:
                                        st.metric("Oracle Condition Verdict", o_verdict.get("label", "Unknown"))

                                    retrieved_lbl = verdict.get("label")
                                    oracle_lbl = o_verdict.get("label")
                                    if retrieved_lbl != oracle_lbl:
                                        st.warning(
                                            f"⚠️ **Verdict Disagreement**: Retrieved = `{retrieved_lbl}`, Oracle = `{oracle_lbl}`. "
                                            "Note: A verdict difference does not automatically imply a retrieval failure; "
                                            "check whether gold evidence was retrieved or how the verifier scored it."
                                        )
                                    else:
                                        st.success(f"✓ Retrieved and Oracle verdicts agree: `{retrieved_lbl}`.")

                                    o_evidence_by_claim = oracle_res.get("evidence_by_claim", {})
                                    o_evidence = o_evidence_by_claim.get(selected_claim_id, [])
                                    if o_evidence:
                                        st.markdown("**Gold Evidence Sentences (Oracle Pool):**")
                                        for gev in o_evidence:
                                            st.markdown(f"- `{gev.get('id')}`: {gev.get('text', '')}")

                    # Record Failure Case
                    with st.expander("Record Failure Case (results/failure_cases/)", expanded=False):
                        st.markdown("Persist this claim inspection as a reproducible failure case in `results/failure_cases/`.")

                        is_editor_modified = (input_text != submitted_snapshot)
                        if is_editor_modified:
                            st.warning(
                                "⚠️ **Input text in editor has changed since last analysis.** "
                                "To save a failure case for modified text, rerun analysis first."
                            )

                        category_options = [
                            "extraction_error",
                            "retrieval_miss",
                            "verifier_error",
                            "aggregation_error",
                            "corpus_insufficient",
                            "oracle_disagreement",
                            "unclear",
                        ]
                        selected_cat = st.selectbox(
                            "Failure Category (Researcher Classification)",
                            options=category_options,
                            index=6,  # default "unclear"
                            help="Classification of the observed failure mode. This is researcher annotation, not model truth.",
                            key=f"fail_cat_{selected_claim_id}",
                        )

                        guardrail_passed = True
                        if selected_cat == "retrieval_miss":
                            has_oracle_evidence = False
                            if st.session_state.get("oracle_result"):
                                has_oracle_evidence = True

                            confirm_corpus_has = st.checkbox(
                                "I have verified that settling evidence exists in the corpus that the retriever missed.",
                                value=has_oracle_evidence,
                                key=f"confirm_retrieval_miss_{selected_claim_id}",
                            )
                            if not confirm_corpus_has:
                                st.caption("⚠️ Guardrail: 'retrieval_miss' requires oracle evidence or explicit confirmation that the corpus contains settling evidence.")
                                guardrail_passed = False

                        elif selected_cat == "oracle_disagreement":
                            st.caption(
                                "ℹ️ Note: Oracle disagreement indicates retrieved != oracle verdict; "
                                "it does not automatically imply retrieval caused the error."
                            )

                        human_label_options = ["None", "Supported", "Contradicted", "Insufficient", "Unclear"]
                        selected_human_label = st.selectbox(
                            "Human Ground-Truth Label (Optional)",
                            options=human_label_options,
                            index=0,
                            key=f"human_label_{selected_claim_id}",
                        )

                        note_text = st.text_area(
                            "Researcher Note",
                            value="",
                            placeholder="Explain the observed failure, discrepancy, or research context...",
                            help="Researcher commentary saved alongside the case.",
                            key=f"note_{selected_claim_id}",
                        )

                        can_save = (
                            not is_editor_modified
                            and guardrail_passed
                            and bool(note_text.strip())
                        )

                        save_btn = st.button(
                            "Save Failure Case",
                            type="primary",
                            disabled=not can_save,
                            key=f"save_btn_{selected_claim_id}",
                        )

                        if save_btn:
                            oracle_rid = (
                                st.session_state.oracle_result.get("run_id")
                                if st.session_state.get("oracle_result")
                                else None
                            )
                            reagg_map = st.session_state.get("reaggregate_comparisons", {})
                            alt_aggs = reagg_map.get(selected_claim_id)

                            fc_payload = {
                                "run_id": result.get("run_id"),
                                "claim_id": selected_claim_id,
                                "failure_category": selected_cat,
                                "researcher_note": note_text.strip(),
                                "human_label": None if selected_human_label == "None" else selected_human_label,
                                "oracle_run_id": oracle_rid,
                                "alternative_aggregations": alt_aggs,
                            }
                            with st.spinner("Saving failure case to results/failure_cases/..."):
                                fc_data, fc_err = send_save_failure_case_request(backend_base_url, fc_payload)
                                if fc_err:
                                    st.session_state.last_save_error = fc_err
                                    st.session_state.last_save_success = None
                                else:
                                    st.session_state.last_save_success = fc_data
                                    st.session_state.last_save_error = None

                        if st.session_state.get("last_save_error"):
                            st.error(st.session_state.last_save_error)

                        if st.session_state.get("last_save_success"):
                            saved_info = st.session_state.last_save_success
                            st.success(f"✓ Saved failure case: `{saved_info.get('file_path')}`")


        # Timings & Provenance
        st.markdown("---")
        with st.expander("Pipeline Timings & Run Provenance", expanded=False):
            timings = result.get("timings", {})
            if timings and isinstance(timings, dict):
                st.markdown("**Stage Timings (milliseconds):**")
                t_cols = st.columns(min(len(timings), 4) or 1)
                for idx, (stage_name, ms_val) in enumerate(sorted(timings.items())):
                    col = t_cols[idx % len(t_cols)]
                    with col:
                        val_str = f"{float(ms_val):.2f} ms" if isinstance(ms_val, (int, float)) else str(ms_val)
                        st.metric(stage_name, val_str)

            st.markdown("**Provenance:**")
            prov = {
                "run_id": result.get("run_id"),
                "config_hash": result.get("config_hash"),
                "git_sha": result.get("git_sha"),
                "timestamp": result.get("timestamp"),
                "mode": result.get("mode"),
                "schema_version": result.get("schema_version"),
            }
            st.json(prov)

            if result.get("resolved_config"):
                st.markdown("**Resolved Configuration:**")
                st.json(result.get("resolved_config"))

        # Raw Trace
        with st.expander("Raw Trace (Complete JSON)", expanded=False):
            st.json(result)


if __name__ == "__main__" or "streamlit" in st.__name__:
    main()
