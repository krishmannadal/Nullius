"""Independent, session-local human annotation UI; no inference or gold display."""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path

import streamlit as st

from src.eval.s1_annotations import (
    AnnotationError,
    canonical_sha256,
    export_annotations,
    load_responses,
    make_review,
    read_jsonl,
    validate_annotations,
    validate_review,
)

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "eval" / "s1"


def main() -> None:
    st.set_page_config(page_title="Nullius · S1 annotation", layout="wide")
    st.title("S1 · Independent human annotation")
    st.caption("Annotate factual claims from source text. No extractors or model outputs run here.")
    st.info(
        "Work independently. Do not import another annotator's work or gold labels. "
        "This session is kept in memory; download your work before closing or reloading."
    )
    benchmark_dir = BENCHMARK_DIR
    metadata = json.loads((benchmark_dir / "metadata.json").read_text(encoding="utf-8"))
    guidelines_data = (benchmark_dir / "ANNOTATION_GUIDELINES.md").read_bytes()
    guideline_hash = canonical_sha256(guidelines_data)
    if metadata.get("guidelines_sha256", guideline_hash) != guideline_hash:
        st.error("Frozen guideline hash differs. Stop and resolve the protocol version.")
        return
    with st.sidebar:
        st.header("Your session")
        identity = st.text_input("Annotator ID")
        guideline = st.text_input("Guideline version", value=metadata.get("guideline_version", "practice-v2"), disabled=True)
        expected_hash = st.text_input(
            "Frozen benchmark SHA-256", value=metadata["responses_hash_sha256"], disabled=True
        )
        uploaded = st.file_uploader("Alternative response benchmark (optional)", type=["jsonl"])
    with st.expander("Annotation guidelines"):
        st.markdown((benchmark_dir / "ANNOTATION_GUIDELINES.md").read_text(encoding="utf-8"))
    if not identity.strip() or not guideline.strip() or not expected_hash.strip():
        st.warning("Enter your annotator ID, guideline version and frozen benchmark hash first.")
        return
    responses_data = (
        uploaded.getvalue() if uploaded else (benchmark_dir / "responses.jsonl").read_bytes()
    )
    try:
        responses = load_responses(responses_data, expected_hash)
    except AnnotationError as exc:
        st.error(str(exc))
        return
    context = (canonical_sha256(responses_data), identity, guideline_hash)
    if st.session_state.get("s1_context") != context:
        if st.session_state.get("s1_records") or st.session_state.get("s1_completed"):
            st.error(
                "Session identity, guidelines or benchmark changed. Restore them and export your work first."
            )
            return
        st.session_state.s1_context = context
        st.session_state.s1_records = []
        st.session_state.s1_completed = []
    records = st.session_state.s1_records
    completed = st.session_state.s1_completed
    st.caption(f"Benchmark SHA-256: {context[0]}")
    st.progress(
        len(completed) / len(responses),
        text=f"{len(completed)} / {len(responses)} responses reviewed",
    )
    with st.expander("Resume your own exported work"):
        annotation_upload = st.file_uploader("Your annotations.jsonl", type=["jsonl"])
        review_upload = st.file_uploader("Your review.json", type=["json"])
        if st.button("Resume export"):
            if records or completed:
                st.error("Resume requires an empty session to avoid overwriting work.")
            elif annotation_upload is None or review_upload is None:
                st.error("Upload both files from your own export.")
            else:
                try:
                    restored = read_jsonl(annotation_upload.getvalue(), "annotations")
                    review = json.loads(review_upload.getvalue())
                    if not isinstance(review, dict):
                        raise AnnotationError("review must be a JSON object")
                    validate_annotations(restored, responses, identity)
                    validate_review(
                        review, responses_data, responses, annotation_upload.getvalue(), restored,
                        guideline_sha256=guideline_hash
                    )
                    if (
                        review["annotator_id"] != identity
                        or review["guideline_version"] != guideline
                    ):
                        raise AnnotationError(
                            "review identity or guideline version differs from this session"
                        )
                    st.session_state.s1_records = restored
                    st.session_state.s1_completed = review["completed_response_ids"]
                    st.rerun()
                except (AnnotationError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                    st.error(str(exc))
    response_id = st.selectbox("Response", list(responses))
    text = responses[response_id]
    st.subheader("Source text")
    st.text(text)
    st.caption(
        "Offsets are zero-based Unicode characters; end is exclusive. No automatic claim suggestions."
    )
    start = st.number_input(
        "Start character", min_value=0, max_value=len(text) - 1, key=f"start_{response_id}"
    )
    end = st.number_input(
        "End character (exclusive)",
        min_value=1,
        max_value=len(text),
        value=len(text),
        key=f"end_{response_id}",
    )
    st.write("Selected source slice")
    st.text(text[start:end])
    with st.form(f"claim_{response_id}", clear_on_submit=True):
        claim_text = st.text_area("Atomic factual claim")
        decomposed = st.checkbox("Split from a compound assertion")
        operations = st.multiselect("Other operations", ["rewrite", "coreference"])
        compound_id = st.text_input("Compound group ID (required for decomposition)")
        hedged = st.checkbox("Hedged assertion")
        notes = st.text_area("Notes (qualifiers, shared span, coreference or uncertainty)")
        extra_spans = st.text_area("Additional source spans as JSON list (optional)", value="[]")
        submitted = st.form_submit_button("Add human annotation")
    if submitted:
        if decomposed:
            operations = ["decomposition", *operations]
        record = {
            "schema_version": "s1-claim-v2",
            "response_id": response_id,
            "claim_id": f"{identity}-{response_id}-{uuid.uuid4().hex[:12]}",
            "claim_text": claim_text,
            "source_span": {"start": start, "end": end},
            "decomposed": decomposed,
            "hedged": hedged,
            "annotator_id": identity,
            "notes": notes,
            "operations": operations,
            "compound_id": compound_id.strip() or None,
        }
        try:
            record["source_spans"] = [record["source_span"], *json.loads(extra_spans)]
            validate_annotations([record], responses, identity)
            records.append(record)
            if response_id in completed:
                completed.remove(response_id)
            st.rerun()
        except (AnnotationError, ValueError, TypeError) as exc:
            st.error(str(exc))
    selected = [r for r in records if r["response_id"] == response_id]
    st.subheader(f"Your annotations ({len(selected)})")
    for record in selected:
        with st.expander(f"{record['claim_id']} · {record['claim_text']}"):
            st.json(record)
            if st.button("Remove annotation", key=f"remove_{record['claim_id']}"):
                records.remove(record)
                if response_id in completed:
                    completed.remove(response_id)
                st.rerun()
    if response_id in completed:
        st.success("You marked this response reviewed.")
        if st.button("Reopen response"):
            completed.remove(response_id)
            st.rerun()
    elif st.button("Mark response reviewed (including zero claims)"):
        completed.append(response_id)
        st.rerun()
    result = validate_annotations(records, responses, identity)
    for warning in result.warnings:
        st.warning(warning)
    annotations_data = export_annotations(records)
    independent = st.checkbox("I worked independently without system outputs or another annotator's work")
    review = make_review(responses_data, annotations_data, identity, guideline, completed,
                         guideline_hash, independent)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annotations.jsonl", annotations_data)
        archive.writestr("review.json", json.dumps(review, ensure_ascii=False, indent=2))
    st.download_button(
        "Download annotations and review record",
        buffer.getvalue(),
        file_name="s1-annotations.zip",
        mime="application/zip",
    )
    st.caption(
        "An export is your draft, not adjudicated gold. A file with zero claims can still contain completed reviews."
    )


if __name__ == "__main__":
    main()
