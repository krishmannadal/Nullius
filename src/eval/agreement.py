"""Prepare human agreement review after two independent, completed annotations.

Lexical candidates are suggestions, never semantic agreement or adjudicated gold.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.eval.extraction_eval import compute_token_f1, normalize_text
from src.eval.s1_annotations import (
    AnnotationError,
    load_responses,
    read_jsonl,
    sha256,
    validate_review,
)


def count_alpha(a: list[int], b: list[int]) -> float | None:
    """Krippendorff interval alpha on paired per-response annotation counts.

    All responses (including explicitly reviewed zero-claim responses) are units.
    With two ratings per unit, observed disagreement is mean squared pair distance;
    expected disagreement is twice the sample variance of the pooled ratings.
    No variation means alpha is undefined, not perfect agreement.
    """
    if len(a) != len(b) or not a:
        raise ValueError("Count ratings must be nonempty paired lists")
    pooled = np.asarray(a + b, dtype=float)
    expected = 2 * float(np.var(pooled, ddof=1))
    if expected == 0:
        return None
    observed = float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
    return 1 - observed / expected


def prepare_agreement(
    responses_data: bytes,
    annotations_a: bytes,
    review_a: dict[str, Any],
    annotations_b: bytes,
    review_b: dict[str, Any],
    expected_sha256: str,
) -> dict[str, Any]:
    responses = load_responses(responses_data, expected_sha256)
    a = read_jsonl(annotations_a, "annotator A")
    b = read_jsonl(annotations_b, "annotator B")
    validate_review(review_a, responses_data, responses, annotations_a, a, require_complete=True)
    validate_review(review_b, responses_data, responses, annotations_b, b, require_complete=True)
    if review_a["annotator_id"] == review_b["annotator_id"]:
        raise AnnotationError("Agreement needs two distinct annotator identities")
    if review_a["guideline_version"] != review_b["guideline_version"]:
        raise AnnotationError(
            "Annotators used different guideline versions; resolve before comparison"
        )
    grouped_a = defaultdict(list)
    grouped_b = defaultdict(list)
    for record in a:
        grouped_a[record["response_id"]].append(record)
    for record in b:
        grouped_b[record["response_id"]].append(record)
    units = []
    for response_id, text in responses.items():
        claims_a, claims_b = grouped_a[response_id], grouped_b[response_id]
        weights = np.zeros((len(claims_a), len(claims_b)))
        candidates = []
        for i, ca in enumerate(claims_a):
            for j, cb in enumerate(claims_b):
                score = compute_token_f1(ca["claim_text"], cb["claim_text"])
                exact = normalize_text(ca["claim_text"]) == normalize_text(cb["claim_text"])
                if exact or score >= 0.8:
                    weights[i, j] = score
                    candidates.append(
                        {
                            "claim_a_id": ca["claim_id"],
                            "claim_b_id": cb["claim_id"],
                            "normalized_exact": exact,
                            "token_f1": score,
                            "human_same_claim": None,
                            "human_notes": "",
                        }
                    )
        rows, columns = linear_sum_assignment(-weights)
        suggested = {
            (claims_a[i]["claim_id"], claims_b[j]["claim_id"])
            for i, j in zip(rows, columns)
            if weights[i, j] > 0
        }
        for candidate in candidates:
            candidate["suggested_one_to_one"] = (
                candidate["claim_a_id"],
                candidate["claim_b_id"],
            ) in suggested
        units.append(
            {
                "response_id": response_id,
                "source_text": text,
                "annotations_a": claims_a,
                "annotations_b": claims_b,
                "count_a": len(claims_a),
                "count_b": len(claims_b),
                "lexical_candidates": candidates,
                "human_review_complete": False,
            }
        )
    counts_a = [u["count_a"] for u in units]
    counts_b = [u["count_b"] for u in units]
    return {
        "schema_version": "s1-agreement-review-v1",
        "status": "human_agreement_review_required",
        "responses_sha256": sha256(responses_data),
        "annotations_a_sha256": sha256(annotations_a),
        "annotations_b_sha256": sha256(annotations_b),
        "annotator_a": review_a["annotator_id"],
        "annotator_b": review_b["annotator_id"],
        "guideline_version": review_a["guideline_version"],
        "response_count": len(units),
        "count_agreement": {
            "krippendorff_interval_alpha": count_alpha(counts_a, counts_b),
            "unit": "response",
            "measurement": "atomic claim annotation count",
            "equal_count_responses": sum(x == y for x, y in zip(counts_a, counts_b)),
            "notice": "Equal counts do not establish semantic claim agreement.",
        },
        "claim_presence_kappa": None,
        "kappa_note": (
            "Not calculated: a shared claim-presence unit universe with explicit absence "
            "decisions must be defined and human-coded. Lexical candidates cannot define it."
        ),
        "notice": (
            "All claims, including those without lexical candidates, need human review. "
            "No gold, extractor metrics or automatic adjudication are produced. "
            "Synthetic benchmark source provenance is not verified by this report."
        ),
        "responses": units,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--annotations-a", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--annotations-b", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON review report; never overwrites"
    )
    args = parser.parse_args()
    try:
        review_a = json.loads(args.review_a.read_text(encoding="utf-8"))
        review_b = json.loads(args.review_b.read_text(encoding="utf-8"))
        if not isinstance(review_a, dict) or not isinstance(review_b, dict):
            raise AnnotationError("Review records must be JSON objects")
        report = prepare_agreement(
            args.responses.read_bytes(),
            args.annotations_a.read_bytes(),
            review_a,
            args.annotations_b.read_bytes(),
            review_b,
            args.expected_sha256,
        )
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Blocked: {exc}\n")
    print(f"Wrote human agreement review: {args.output}. No gold was generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
