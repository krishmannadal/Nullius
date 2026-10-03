"""Deterministic mechanical validator for Nullius S1 Human Annotation files.

Validates annotations_A.jsonl and annotations_B.jsonl against the approved
S1 protocol and responses.jsonl.

SCIENTIFIC INTEGRITY CONSTRAINTS:
- MUST NOT evaluate or judge the scientific correctness of claims.
- MUST NOT determine whether a claim should or should not exist.
- MUST NOT automatically correct, rewrite, or alter annotations.
- MUST NOT compare Annotator A against Annotator B.
- MUST NOT generate gold annotations.
- MUST NOT load, invoke, or reference any extractor or model predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.s1_annotations import (
    load_responses as load_response_bytes,
)
from src.eval.s1_annotations import (
    read_jsonl,
    validate_review,
)


class AnnotationValidationError:
    def __init__(self, line_no: int, claim_id: str | None, message: str) -> None:
        self.line_no = line_no
        self.claim_id = claim_id
        self.message = message

    def __str__(self) -> str:
        cid_str = f" [claim_id: {self.claim_id}]" if self.claim_id else ""
        return f"Line {self.line_no}{cid_str}: {self.message}"


class AnnotationValidationWarning:
    def __init__(self, line_no: int, claim_id: str | None, message: str) -> None:
        self.line_no = line_no
        self.claim_id = claim_id
        self.message = message

    def __str__(self) -> str:
        cid_str = f" [claim_id: {self.claim_id}]" if self.claim_id else ""
        return f"Line {self.line_no}{cid_str}: {self.message}"


def load_responses(responses_path: Path) -> dict[str, str]:
    """Load response_id -> text mapping from responses.jsonl."""
    if not responses_path.exists():
        raise FileNotFoundError(f"Responses file not found: {responses_path}")

    return load_response_bytes(responses_path.read_bytes())


def validate_annotations(
    annotation_path: Path,
    responses: dict[str, str],
    expected_annotator_id: str | None = None,
    show_spans: bool = False,
) -> tuple[list[AnnotationValidationError], list[AnnotationValidationWarning], dict[str, Any]]:
    """Perform deterministic structural validation of an annotation file."""
    errors: list[AnnotationValidationError] = []
    warnings: list[AnnotationValidationWarning] = []

    stats: dict[str, Any] = {
        "total_records": 0,
        "valid_records": 0,
        "unique_claims": 0,
        "responses_covered": set(),
        "decomposed_count": 0,
        "hedged_count": 0,
        "inspected_records": [],
    }

    if not annotation_path.exists():
        errors.append(AnnotationValidationError(0, None, f"File not found: {annotation_path}"))
        return errors, warnings, stats

    seen_claim_ids: set[str] = set()

    with open(annotation_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines or all(not line.strip() for line in lines):
        # Empty template or blank file
        return errors, warnings, stats

    for line_no, raw_line in enumerate(lines, 1):
        line_str = raw_line.strip()
        if not line_str:
            warnings.append(
                AnnotationValidationWarning(line_no, None, "Empty/whitespace line ignored.")
            )
            continue

        stats["total_records"] += 1
        previous_error_count = len(errors)

        try:
            record = json.loads(line_str)
        except json.JSONDecodeError as err:
            errors.append(AnnotationValidationError(line_no, None, f"Invalid JSON syntax: {err}"))
            continue

        if not isinstance(record, dict):
            errors.append(
                AnnotationValidationError(
                    line_no, None, f"Record must be a JSON object, got {type(record).__name__}"
                )
            )
            continue

        claim_id = record.get("claim_id")
        claim_id_str = str(claim_id) if claim_id is not None else None

        # 1. Validate required fields existence
        required_fields = [
            "response_id",
            "claim_id",
            "claim_text",
            "source_span",
            "decomposed",
            "hedged",
            "annotator_id",
        ]
        missing_fields = [field for field in required_fields if field not in record]
        if missing_fields:
            errors.append(
                AnnotationValidationError(
                    line_no, claim_id_str, f"Missing required fields: {missing_fields}"
                )
            )
            continue

        # 2. Validate claim_id uniqueness
        if not isinstance(claim_id, str) or not claim_id.strip():
            errors.append(
                AnnotationValidationError(
                    line_no, None, "Field 'claim_id' must be a non-empty string."
                )
            )
        elif claim_id in seen_claim_ids:
            errors.append(
                AnnotationValidationError(
                    line_no, claim_id, f"Duplicate claim_id '{claim_id}' detected."
                )
            )
        else:
            seen_claim_ids.add(claim_id)

        # 3. Validate annotator_id if present or expected
        annotator_id = record.get("annotator_id")
        if not isinstance(annotator_id, str) or not annotator_id.strip():
            errors.append(
                AnnotationValidationError(
                    line_no,
                    claim_id_str,
                    f"Field 'annotator_id' must be a string, got {type(annotator_id).__name__}",
                )
            )
        elif expected_annotator_id and annotator_id != expected_annotator_id:
            errors.append(
                AnnotationValidationError(
                    line_no,
                    claim_id_str,
                    f"Mismatched annotator_id: expected '{expected_annotator_id}', found '{annotator_id}'",
                )
            )

        if not isinstance(record.get("notes", ""), str):
            errors.append(
                AnnotationValidationError(line_no, claim_id_str, "Field 'notes' must be a string.")
            )

        # 4. Validate response_id
        response_id = record.get("response_id")
        if not isinstance(response_id, str) or not response_id.strip():
            errors.append(
                AnnotationValidationError(
                    line_no, claim_id_str, "Field 'response_id' must be a non-empty string."
                )
            )
            continue

        if response_id not in responses:
            errors.append(
                AnnotationValidationError(
                    line_no,
                    claim_id_str,
                    f"Unknown response_id '{response_id}'. Must exist in responses.jsonl.",
                )
            )
            continue

        response_text = responses[response_id]

        # 5. Validate claim_text
        claim_text = record.get("claim_text")
        if not isinstance(claim_text, str) or not claim_text.strip():
            errors.append(
                AnnotationValidationError(
                    line_no, claim_id_str, "Field 'claim_text' must be a non-empty string."
                )
            )

        # 6. Validate booleans strictly
        decomposed = record.get("decomposed")
        if not isinstance(decomposed, bool):
            errors.append(
                AnnotationValidationError(
                    line_no,
                    claim_id_str,
                    f"Field 'decomposed' must be a boolean, got {type(decomposed).__name__}",
                )
            )

        hedged = record.get("hedged")
        if not isinstance(hedged, bool):
            errors.append(
                AnnotationValidationError(
                    line_no,
                    claim_id_str,
                    f"Field 'hedged' must be a boolean, got {type(hedged).__name__}",
                )
            )

        if isinstance(decomposed, bool) and decomposed:
            stats["decomposed_count"] += 1
        if isinstance(hedged, bool) and hedged:
            stats["hedged_count"] += 1

        # 7. Validate source_span
        source_span = record.get("source_span")
        span_valid = True
        if not isinstance(source_span, dict):
            errors.append(
                AnnotationValidationError(
                    line_no,
                    claim_id_str,
                    f"Field 'source_span' must be a dict, got {type(source_span).__name__}",
                )
            )
            span_valid = False
        else:
            start = source_span.get("start")
            end = source_span.get("end")

            # Check exact integer type without bools (since isinstance(True, int) is True in Python)
            if not isinstance(start, int) or isinstance(start, bool):
                errors.append(
                    AnnotationValidationError(
                        line_no,
                        claim_id_str,
                        f"source_span.start must be an integer, got {start!r}",
                    )
                )
                span_valid = False
            if not isinstance(end, int) or isinstance(end, bool):
                errors.append(
                    AnnotationValidationError(
                        line_no, claim_id_str, f"source_span.end must be an integer, got {end!r}"
                    )
                )
                span_valid = False

            if span_valid:
                resp_len = len(response_text)
                if start < 0:
                    errors.append(
                        AnnotationValidationError(
                            line_no, claim_id_str, f"source_span.start must be >= 0, got {start}"
                        )
                    )
                    span_valid = False
                if end <= start:
                    errors.append(
                        AnnotationValidationError(
                            line_no,
                            claim_id_str,
                            f"source_span.end ({end}) must be strictly greater than start ({start})",
                        )
                    )
                    span_valid = False
                if end > resp_len:
                    errors.append(
                        AnnotationValidationError(
                            line_no,
                            claim_id_str,
                            f"source_span.end ({end}) exceeds response length ({resp_len}) for response '{response_id}'",
                        )
                    )
                    span_valid = False

        # 8. Source Span Integrity and Extraction Verification
        if span_valid and isinstance(claim_text, str) and isinstance(decomposed, bool):
            extracted_slice = response_text[source_span["start"] : source_span["end"]]

            if show_spans:
                stats["inspected_records"].append(
                    {
                        "line_no": line_no,
                        "response_id": response_id,
                        "claim_id": claim_id,
                        "span": (source_span["start"], source_span["end"]),
                        "extracted_slice": extracted_slice,
                        "claim_text": claim_text,
                        "decomposed": decomposed,
                        "hedged": hedged,
                    }
                )

            # Check verbatim vs decomposed consistency
            if not decomposed and claim_text != extracted_slice:
                warnings.append(
                    AnnotationValidationWarning(
                        line_no,
                        claim_id_str,
                        f"Claim has decomposed=false, but claim_text != response_text[start:end].\n"
                        f"  claim_text:    {claim_text!r}\n"
                        f"  response_span: {extracted_slice!r}\n"
                        f"  (If the claim was edited, trimmed, or had pronouns resolved, set decomposed=true)",
                    )
                )

        if len(errors) == previous_error_count:
            stats["valid_records"] += 1
            stats["responses_covered"].add(response_id)

    stats["unique_claims"] = len(seen_claim_ids)
    return errors, warnings, stats


def format_report(
    annotation_path: Path,
    responses_path: Path,
    errors: list[AnnotationValidationError],
    warnings: list[AnnotationValidationWarning],
    stats: dict[str, Any],
    total_expected_responses: int,
    show_spans: bool,
) -> str:
    """Format human-readable validation report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("NULLIUS S1 ANNOTATION VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Annotation file: {annotation_path}")
    lines.append(f"Responses file:  {responses_path}")
    lines.append("-" * 70)

    if stats["total_records"] == 0 and not errors:
        lines.append("STATUS: EMPTY FILE (0 records found)")
        lines.append("This file is currently empty or contains only whitespace.")
        lines.append("=" * 70)
        return "\n".join(lines)

    lines.append(f"Total records parsed:    {stats['total_records']}")
    lines.append(f"Structurally valid records: {stats['valid_records']}")
    lines.append(f"Unique claim IDs:        {stats['unique_claims']}")
    lines.append(
        f"Responses covered:       {len(stats['responses_covered'])} / {total_expected_responses}"
    )
    lines.append(f"Decomposed claims:       {stats['decomposed_count']}")
    lines.append(f"Hedged claims:           {stats['hedged_count']}")
    lines.append(f"Errors detected:         {len(errors)}")
    lines.append(f"Warnings detected:       {len(warnings)}")
    lines.append("-" * 70)

    missing_responses = total_expected_responses - len(stats["responses_covered"])
    if missing_responses > 0:
        lines.append(
            f"NOTE: {missing_responses} responses in responses.jsonl do not have annotations in this file yet."
        )

    if show_spans and stats["inspected_records"]:
        lines.append("")
        lines.append("--- SOURCE SPAN INTEGRITY INSPECTION ---")
        for rec in stats["inspected_records"]:
            lines.append(
                f"[L{rec['line_no']}] Claim: {rec['claim_id']} ({rec['response_id']}) | "
                f"Span: [{rec['span'][0]}, {rec['span'][1]}) | Decomp: {rec['decomposed']} | Hedged: {rec['hedged']}"
            )
            lines.append(f"  Source Span Slice: {rec['extracted_slice']!r}")
            lines.append(f"  Final Claim Text:  {rec['claim_text']!r}")
            lines.append("")
        lines.append("--- END SOURCE SPAN INSPECTION ---")
        lines.append("")

    if warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in warnings:
            lines.append(f"  [WARN] {w}")

    if errors:
        lines.append("")
        lines.append("ERRORS:")
        for err in errors:
            lines.append(f"  [ERROR] {err}")
        lines.append("")
        lines.append("RESULT: VALIDATION FAILED")
    else:
        lines.append("")
        lines.append("RESULT: VALIDATION PASSED (All mechanical checks satisfied)")

    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mechanical validator for Nullius S1 independent human annotation files."
    )
    parser.add_argument(
        "annotation_file",
        type=Path,
        help="Path to annotation JSONL file (e.g. data/eval/s1/annotations_A.jsonl)",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/eval/s1/responses.jsonl"),
        help="Path to responses.jsonl (default: data/eval/s1/responses.jsonl)",
    )
    parser.add_argument(
        "--annotator-id",
        type=str,
        default=None,
        help="Expected annotator ID ('A' or 'B')",
    )
    parser.add_argument(
        "--show-spans",
        action="store_true",
        help="Display exact response_text[start:end] slice alongside claim_text for human inspection",
    )
    parser.add_argument("--expected-sha256", help="Frozen benchmark byte SHA-256")
    parser.add_argument(
        "--review", type=Path, help="Completion/provenance JSON exported by annotation UI"
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require every response to be explicitly reviewed",
    )

    args = parser.parse_args()

    try:
        responses = load_responses(args.responses)
        if args.expected_sha256:
            load_response_bytes(args.responses.read_bytes(), args.expected_sha256)
    except (OSError, ValueError) as err:
        print(f"Error loading responses file: {err}", file=sys.stderr)
        return 1

    errors, warnings, stats = validate_annotations(
        annotation_path=args.annotation_file,
        responses=responses,
        expected_annotator_id=args.annotator_id,
        show_spans=args.show_spans,
    )

    try:
        if args.require_complete and args.review is None:
            raise ValueError(
                "--require-complete needs --review; claim coverage cannot prove zero-claim reviews"
            )
        if args.review is not None:
            review = json.loads(args.review.read_text(encoding="utf-8"))
            if not isinstance(review, dict):
                raise ValueError("review must be a JSON object")
            data = args.annotation_file.read_bytes()
            validate_review(
                review,
                args.responses.read_bytes(),
                responses,
                data,
                read_jsonl(data, "annotations"),
                args.require_complete,
            )
            if args.annotator_id and review["annotator_id"] != args.annotator_id:
                raise ValueError("review annotator differs from --annotator-id")
    except (OSError, ValueError) as err:
        errors.append(AnnotationValidationError(0, None, str(err)))

    report = format_report(
        annotation_path=args.annotation_file,
        responses_path=args.responses,
        errors=errors,
        warnings=warnings,
        stats=stats,
        total_expected_responses=len(responses),
        show_spans=args.show_spans,
    )
    print(report)
    if args.require_complete and not errors:
        print(
            "Completion verified: all response IDs explicitly reviewed, including zero-claim responses."
        )

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
