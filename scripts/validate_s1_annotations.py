"""CLI adapter for the authoritative deterministic annotation validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.s1_annotations import AnnotationError, load_responses, read_jsonl, validate_review
from src.eval.s1_annotations import validate_annotations as validate_records
from src.eval.s1_workflow import ROOT, benchmark


def validate_annotations(path, responses, annotator_id=None, formal=False):
    errors, warnings, valid = [], [], 0
    try:
        records = read_jsonl(Path(path).read_bytes(), "annotations")
        for index, record in enumerate(records, 1):
            try:
                result = validate_records([record], responses, annotator_id, formal=formal)
                warnings.extend(result.warnings)
                valid += 1
            except AnnotationError as exc:
                errors.append(f"record {index}: {exc}")
        if not errors:
            validate_records(records, responses, annotator_id, formal=formal)
    except (OSError, AnnotationError) as exc:
        errors.append(str(exc))
    return errors, warnings, {"valid_records": valid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--responses", type=Path, default=ROOT / "responses.jsonl")
    parser.add_argument("--annotator-id")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--show-spans", action="store_true")
    args = parser.parse_args()
    try:
        if args.require_complete and args.review is None:
            raise AnnotationError("--require-complete needs --review")
        if args.formal:
            metadata, data, responses = benchmark(args.responses.parent)
            guideline = metadata["guidelines_sha256"]
        else:
            data = args.responses.read_bytes()
            metadata_path = args.responses.parent / "metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            responses = load_responses(
                data, args.expected_sha256 or metadata.get("responses_hash_sha256")
            )
            guideline = None
        errors, warnings, stats = validate_annotations(
            args.annotations, responses, args.annotator_id, args.formal
        )
        if errors:
            raise AnnotationError("; ".join(errors))
        raw = args.annotations.read_bytes()
        records = read_jsonl(raw, "annotations")
        if args.review:
            review = json.loads(args.review.read_text(encoding="utf-8"))
            validate_review(review, data, responses, raw, records, args.require_complete, guideline)
            if args.annotator_id and review["annotator_id"] != args.annotator_id:
                raise AnnotationError("wrong annotator review")
        if args.formal and (not args.require_complete or not args.review):
            raise AnnotationError("formal submissions require --require-complete and --review")
        print(
            f"Validated {stats['valid_records']} records. "
            + (
                "Completion verified."
                if args.require_complete
                else "Draft only; completion not verified."
            )
        )
        for warning in warnings:
            print("Warning:", warning)
        if args.show_spans:
            for record in records:
                span = record["source_span"]
                print(
                    record["claim_id"],
                    repr(responses[record["response_id"]][span["start"] : span["end"]]),
                )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Blocked: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
