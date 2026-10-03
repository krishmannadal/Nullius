"""Validate human S1 annotations without running or judging extractors.

Offsets are Python Unicode character offsets, not UTF-8 bytes or UTF-16 units.
Human decisions remain separate from validation and descriptive comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


class AnnotationError(ValueError):
    """An input violates the annotation data contract."""


@dataclass(frozen=True)
class ValidationResult:
    annotation_count: int
    annotated_response_ids: tuple[str, ...]
    warnings: tuple[str, ...]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(data: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnnotationError(f"{label}: expected UTF-8") from exc
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise AnnotationError(f"{label}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise AnnotationError(f"{label}:{line_number}: expected a JSON object")
        records.append(record)
    return records


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationError(f"{label}: expected a nonempty string")
    return value


def load_responses(data: bytes, expected_sha256: str | None = None) -> dict[str, str]:
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise AnnotationError("expected_sha256: expected 64 hexadecimal characters")
        if sha256(data) != expected:
            raise AnnotationError("responses: SHA-256 differs from the frozen benchmark")
    responses = {}
    for number, record in enumerate(read_jsonl(data, "responses"), 1):
        # Accommodate either published response-file convention, never ambiguous IDs.
        if "response_id" in record and "id" in record and record["response_id"] != record["id"]:
            raise AnnotationError(f"responses:{number}: conflicting id and response_id")
        response_id = _nonempty_string(
            record.get("response_id", record.get("id")), f"responses:{number}:response_id"
        )
        text = _nonempty_string(record.get("text"), f"responses:{number}:text")
        if response_id in responses:
            raise AnnotationError(f"responses:{number}: duplicate response_id {response_id}")
        responses[response_id] = text
    if not responses:
        raise AnnotationError("responses: benchmark is empty")
    return responses


def validate_annotations(
    records: list[dict[str, Any]],
    responses: dict[str, str],
    annotator_id: str | None = None,
) -> ValidationResult:
    seen = set()
    annotated = set()
    warnings = []
    identities = set()
    required = {
        "response_id",
        "claim_id",
        "claim_text",
        "source_span",
        "decomposed",
        "hedged",
        "annotator_id",
    }
    for number, record in enumerate(records, 1):
        label = f"annotations:{number}"
        missing = required - record.keys()
        if missing:
            raise AnnotationError(f"{label}: missing fields: {', '.join(sorted(missing))}")
        response_id = _nonempty_string(record["response_id"], f"{label}:response_id")
        claim_id = _nonempty_string(record["claim_id"], f"{label}:claim_id")
        claim_text = _nonempty_string(record["claim_text"], f"{label}:claim_text")
        identity = _nonempty_string(record["annotator_id"], f"{label}:annotator_id")
        identities.add(identity)
        if annotator_id is not None and identity != annotator_id:
            raise AnnotationError(f"{label}: annotator_id does not match this session")
        if response_id not in responses:
            raise AnnotationError(f"{label}: unknown response_id {response_id}")
        key = claim_id
        if key in seen:
            raise AnnotationError(f"{label}: duplicate claim_id {claim_id}")
        seen.add(key)
        annotated.add(response_id)
        for field in ("decomposed", "hedged"):
            if type(record[field]) is not bool:
                raise AnnotationError(f"{label}:{field}: expected a boolean")
        if not isinstance(record.get("notes", ""), str):
            raise AnnotationError(f"{label}:notes: expected a string")
        span = record["source_span"]
        if not isinstance(span, dict) or set(span) != {"start", "end"}:
            raise AnnotationError(f"{label}:source_span: expected start and end only")
        start, end = span["start"], span["end"]
        if type(start) is not int or type(end) is not int:
            raise AnnotationError(f"{label}:source_span: offsets must be integers, not booleans")
        text = responses[response_id]
        if not 0 <= start < end <= len(text):
            raise AnnotationError(f"{label}:source_span: outside [0, {len(text)}] or empty")
        if not text[start:end].strip():
            raise AnnotationError(f"{label}:source_span: contains only whitespace")
        if not record["decomposed"] and text[start:end] != claim_text:
            warnings.append(
                f"{label}: non-decomposed claim differs from source slice; human review required"
            )
    if len(identities) > 1:
        raise AnnotationError("annotations: multiple annotator identities in an independent file")
    return ValidationResult(len(records), tuple(sorted(annotated)), tuple(warnings))


def validate_review(
    review: dict[str, Any],
    responses_data: bytes,
    responses: dict[str, str],
    annotations_data: bytes,
    records: list[dict[str, Any]],
    require_complete: bool = False,
) -> None:
    """Verify explicit response completion, including responses with zero claims."""
    if review.get("schema_version") != "s1-annotation-review-v1":
        raise AnnotationError("review: unsupported schema_version")
    if review.get("responses_sha256") != sha256(responses_data):
        raise AnnotationError("review: response hash differs from this benchmark")
    if review.get("annotations_sha256") != sha256(annotations_data):
        raise AnnotationError("review: annotation hash differs from this export")
    identity = _nonempty_string(review.get("annotator_id"), "review:annotator_id")
    _nonempty_string(review.get("guideline_version"), "review:guideline_version")
    validate_annotations(records, responses, identity)
    completed = review.get("completed_response_ids")
    if not isinstance(completed, list) or any(not isinstance(x, str) for x in completed):
        raise AnnotationError("review: completed_response_ids must be a list of strings")
    if len(set(completed)) != len(completed) or set(completed) - responses.keys():
        raise AnnotationError("review: duplicate or unknown completed response IDs")
    if require_complete and set(completed) != responses.keys():
        missing = sorted(responses.keys() - set(completed))
        raise AnnotationError(f"review: responses not completed: {', '.join(missing)}")


def export_annotations(records: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


def make_review(
    responses_data: bytes,
    annotations_data: bytes,
    annotator_id: str,
    guideline_version: str,
    completed_response_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "s1-annotation-review-v1",
        "responses_sha256": sha256(responses_data),
        "annotations_sha256": sha256(annotations_data),
        "annotator_id": annotator_id,
        "guideline_version": guideline_version,
        "completed_response_ids": sorted(completed_response_ids),
    }
