"""Fail-closed artifact contracts. No inference and no automatic human decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.config import git_is_dirty, git_sha
from src.core.types import utcnow_iso
from src.eval.s1_annotations import (
    AnnotationError,
    canonical_sha256,
    load_responses,
    read_jsonl,
    sha256,
    validate_annotations,
    validate_review,
)

ROOT = Path(__file__).resolve().parents[2] / "data/eval/s1"
CLAIM_LIMITATION = (
    "Curated synthetic pilot / provenance-unverified. Aggregate extraction results only. "
    "Source and length labels are curated, not verified authorship or population strata. "
    "No hallucination-detection, retrieval, verifier, aggregation or calibration validity is established."
)


def require(condition, message):
    if not condition:
        raise AnnotationError(message)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AnnotationError(f"Missing or invalid artifact {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def write_new(path, value):
    """Never overwrite a scientific artifact or prior human decision."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def benchmark(directory=ROOT):
    directory = Path(directory)
    metadata = read_json(directory / "metadata.json")
    data = (directory / "responses.jsonl").read_bytes()
    responses = load_responses(data, metadata["responses_hash_sha256"])
    count = metadata.get("response_count", 30)
    require(len(responses) == count, "benchmark response count mismatch")
    expected_ids = metadata.get("response_ids", [f"s1-resp-{i:02}" for i in range(1, 31)])
    require(
        set(responses) == set(expected_ids) and len(expected_ids) == count,
        "benchmark response IDs mismatch",
    )
    guideline_hash = canonical_sha256((directory / "ANNOTATION_GUIDELINES.md").read_bytes())
    require(guideline_hash == metadata.get("guidelines_sha256"), "stale guideline content hash")
    require(
        metadata.get("provenance_status") == "curated synthetic pilot / provenance-unverified",
        "provenance must be explicitly scoped; verified provenance requires a new reviewed protocol",
    )
    protocol_hash = canonical_sha256((directory / "EVALUATION_PROTOCOL.md").read_bytes())
    require(
        protocol_hash == metadata.get("evaluation_protocol_sha256"), "stale evaluation protocol"
    )
    return metadata, data, responses


def submission(directory, identity):
    metadata, data, responses = benchmark(directory)
    path = Path(directory) / f"annotations_{identity}.jsonl"
    raw = path.read_bytes()
    records = read_jsonl(raw, str(path))
    review = read_json(Path(directory) / f"review_{identity}.json")
    require(review.get("annotator_id") == identity, "wrong annotator identity")
    validate_review(
        review,
        data,
        responses,
        raw,
        records,
        require_complete=True,
        guideline_sha256=metadata["guidelines_sha256"],
    )
    return records, review


def _complete_units(units, responses, label):
    require(isinstance(units, list), f"{label}: response review list required")
    ids = [u.get("response_id") for u in units]
    require(
        len(ids) == len(set(ids)) and set(ids) == set(responses),
        f"{label}: every response must occur exactly once",
    )
    require(
        all(u.get("human_review_complete") is True for u in units),
        f"{label}: human review incomplete",
    )


def validate_gold_inputs(directory):
    directory = Path(directory)
    metadata, data, responses = benchmark(directory)
    a, _ = submission(directory, "A")
    b, _ = submission(directory, "B")
    agreement = read_json(directory / "agreement.json")
    adjudication = read_json(directory / "adjudication.json")
    gold_raw = (directory / "gold_annotations.jsonl").read_bytes()
    gold = read_jsonl(gold_raw, "gold")
    require(bool(gold), "An empty gold file cannot establish this reference standard")
    result = validate_annotations(gold, responses, formal=True)
    require(not result.warnings, "Gold contains unresolved validation warnings")
    for label, artifact in [("agreement", agreement), ("adjudication", adjudication)]:
        require(
            artifact.get("responses_sha256") == metadata["responses_hash_sha256"],
            f"{label}: stale benchmark",
        )
        require(
            artifact.get("guidelines_sha256") == metadata["guidelines_sha256"],
            f"{label}: stale guidelines",
        )
        for identity in ("A", "B"):
            require(
                artifact.get(f"annotations_{identity.lower()}_sha256")
                == sha256((directory / f"annotations_{identity}.jsonl").read_bytes()),
                f"{label}: stale annotation {identity}",
            )
        _complete_units(artifact.get("responses"), responses, label)
    require(bool(adjudication.get("adjudicator_id", "").strip()), "adjudicator identity required")
    require(
        adjudication.get("no_extractor_outputs_used") is True,
        "adjudication needs explicit no-extractor-output attestation",
    )
    require(
        adjudication.get("agreement_sha256") == sha256((directory / "agreement.json").read_bytes()),
        "adjudication must reference finalized agreement",
    )
    require(adjudication.get("gold_sha256") == sha256(gold_raw), "adjudication gold hash mismatch")
    for unit in agreement["responses"]:
        rid = unit["response_id"]
        aa = {r["claim_id"] for r in a if r["response_id"] == rid}
        bb = {r["claim_id"] for r in b if r["response_id"] == rid}
        pairs = unit.get("human_matches")
        require(
            isinstance(pairs, list), "agreement: explicit human_matches required (possibly empty)"
        )
        left, right = [], []
        for pair in pairs:
            require(
                pair.get("claim_a_id") in aa and pair.get("claim_b_id") in bb,
                "agreement: unknown human pair",
            )
            left.append(pair["claim_a_id"])
            right.append(pair["claim_b_id"])
        require(
            len(left) == len(set(left)) and len(right) == len(set(right)),
            "agreement: one-to-one collision",
        )
        require(
            set(unit.get("unmatched_a", [])) == aa - set(left)
            and set(unit.get("unmatched_b", [])) == bb - set(right),
            "agreement: unmatched claims must be explicit",
        )
    gold_ids = {g["claim_id"] for g in gold}
    grouped_ids = set()
    for unit in adjudication["responses"]:
        rid = unit["response_id"]
        ids = [g["claim_id"] for g in gold if g["response_id"] == rid]
        require(
            sorted(unit.get("gold_claim_ids", [])) == sorted(ids),
            "adjudication: gold claim coverage mismatch",
        )
        require(
            type(unit.get("zero_claims")) is bool and unit["zero_claims"] == (not ids),
            "adjudication: explicit zero-claim decision required",
        )
        require(
            isinstance(unit.get("notes"), str) and bool(unit["notes"].strip()),
            "adjudication: rationale required for every response",
        )
        group_names = set()
        for group in unit.get("compound_groups", []):
            members = group.get("gold_claim_ids", [])
            gid = group.get("group_id")
            require(
                isinstance(gid, str) and gid and gid not in group_names, "invalid compound group ID"
            )
            require(
                len(members) >= 2
                and len(members) == len(set(members))
                and set(members) <= set(ids),
                "compound group needs distinct gold claims in its response",
            )
            require(not set(members) & grouped_ids, "gold belongs to multiple compound groups")
            for claim in gold:
                if claim["claim_id"] in members:
                    require(
                        claim.get("compound_id") == gid and claim["decomposed"],
                        "compound relation and gold annotation disagree",
                    )
            grouped_ids.update(members)
            group_names.add(gid)
    require(
        grouped_ids == {g["claim_id"] for g in gold if g["decomposed"]},
        "all decomposed gold claims need explicit compound relations",
    )
    require(grouped_ids <= gold_ids, "unknown compound member")
    names = [
        "annotations_A.jsonl",
        "annotations_B.jsonl",
        "review_A.json",
        "review_B.json",
        "agreement.json",
        "adjudication.json",
        "gold_annotations.jsonl",
    ]
    manifest = {
        "schema_version": "s1-gold-freeze-v2",
        "benchmark_version": metadata["benchmark_version"],
        "responses_sha256": canonical_sha256(data),
        "responses_raw_sha256": sha256(data),
        "response_count": len(responses),
        "response_ids": list(responses),
        "guidelines_sha256": metadata["guidelines_sha256"],
        "evaluation_protocol_sha256": metadata["evaluation_protocol_sha256"],
        "artifacts": {name: sha256((directory / name).read_bytes()) for name in names},
        "response_reviews": adjudication["responses"],
        "adjudicator_id": adjudication["adjudicator_id"],
        "no_extractor_outputs_used": True,
        "limitation": CLAIM_LIMITATION,
    }
    return manifest, gold


def freeze_gold(directory=ROOT):
    manifest, _ = validate_gold_inputs(directory)
    manifest.update(
        frozen_at=utcnow_iso(), software_revision=git_sha(), software_dirty=git_is_dirty()
    )
    write_new(Path(directory) / "gold_manifest.json", manifest)
    return manifest


def load_gold(directory=ROOT):
    directory = Path(directory)
    if not (directory / "gold_annotations.jsonl").exists():
        raise AnnotationError("Human-adjudicated gold is missing. No evaluation was run.")
    require(
        (directory / "gold_manifest.json").exists(),
        "Gold is not frozen: gold_manifest.json required",
    )
    frozen = read_json(directory / "gold_manifest.json")
    current, gold = validate_gold_inputs(directory)
    # Raw transport identity records the original machine, not an EOL-dependent gate.
    for key, value in current.items():
        if key != "responses_raw_sha256":
            require(frozen.get(key) == value, f"Gold freeze mismatch: {key}")
    require(
        bool(frozen.get("frozen_at")) and "software_revision" in frozen,
        "incomplete freeze provenance",
    )
    return frozen, gold


def prepare_adjudication(directory=ROOT):
    """Create an empty decision form, never claims or gold."""
    from src.eval.agreement import prepare_agreement

    directory = Path(directory)
    metadata, data, responses = benchmark(directory)
    _, ra = submission(directory, "A")
    _, rb = submission(directory, "B")
    report = prepare_agreement(
        data,
        (directory / "annotations_A.jsonl").read_bytes(),
        ra,
        (directory / "annotations_B.jsonl").read_bytes(),
        rb,
        metadata["responses_hash_sha256"],
    )
    report["guidelines_sha256"] = metadata["guidelines_sha256"]
    for unit in report["responses"]:
        unit.update(human_matches=[], unmatched_a=[], unmatched_b=[])
    write_new(directory / "agreement.draft.json", report)
    form = {
        k: report[k]
        for k in (
            "responses_sha256",
            "guidelines_sha256",
            "annotations_a_sha256",
            "annotations_b_sha256",
        )
    }
    form.update(
        schema_version="s1-adjudication-v2",
        adjudicator_id="",
        agreement_sha256="",
        gold_sha256="",
        no_extractor_outputs_used=False,
        responses=[
            {
                "response_id": rid,
                "human_review_complete": False,
                "gold_claim_ids": [],
                "zero_claims": None,
                "notes": "",
                "compound_groups": [],
            }
            for rid in responses
        ],
    )
    write_new(directory / "adjudication.draft.json", form)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["validate-benchmark", "prepare-adjudication", "freeze-gold", "validate-gold"],
    )
    parser.add_argument("--s1-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        {
            "validate-benchmark": benchmark,
            "prepare-adjudication": prepare_adjudication,
            "freeze-gold": freeze_gold,
            "validate-gold": load_gold,
        }[args.command](args.s1_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Blocked: {exc}\n")
    print(f"Completed {args.command}. {CLAIM_LIMITATION}")


if __name__ == "__main__":
    main()
