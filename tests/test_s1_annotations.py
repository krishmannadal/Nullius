"""Synthetic annotation fixtures only; never write labels into the S1 benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from scripts.generate_s1_llm_cache import generate_llm_cache
from scripts.validate_s1_annotations import validate_annotations as validate_file
from src.eval.extraction_eval import compute_token_f1, generate_adjudication_csv, load_data
from src.eval.s1_annotations import (
    AnnotationError,
    export_annotations,
    load_responses,
    make_review,
    sha256,
    validate_annotations,
    validate_review,
)


@pytest.fixture
def benchmark(tmp_path):
    data = (json.dumps({"response_id": "fixture-1", "text": "A 🐈 sleeps."}) + "\n").encode()
    (tmp_path / "responses.jsonl").write_bytes(data)
    (tmp_path / "metadata.json").write_text(json.dumps({"responses_hash_sha256": sha256(data)}))
    (tmp_path / "ANNOTATION_GUIDELINES.md").write_text("Synthetic test guidelines only.")
    return tmp_path, data, load_responses(data)


@pytest.fixture
def record():
    return {
        "response_id": "fixture-1",
        "claim_id": "fixture-A-1",
        "claim_text": "A 🐈 sleeps.",
        "source_span": {"start": 0, "end": 11},
        "decomposed": False,
        "hedged": False,
        "annotator_id": "A",
        "notes": "Synthetic fixture",
    }


def test_unicode_offsets_and_optional_notes(benchmark, record):
    record.pop("notes")
    result = validate_annotations([record], benchmark[2], "A")
    assert result.annotation_count == 1
    assert not result.warnings


@pytest.mark.parametrize(
    "field,value",
    [
        ("decomposed", 1),
        ("hedged", "false"),
        ("notes", []),
        ("annotator_id", ""),
        ("claim_text", " "),
        ("response_id", "unknown"),
        ("source_span", {"start": True, "end": 11}),
        ("source_span", {"start": 0, "end": 12}),
        ("source_span", {"start": 5, "end": 5}),
    ],
)
def test_reject_invalid_annotation(benchmark, record, field, value):
    record[field] = value
    with pytest.raises(AnnotationError):
        validate_annotations([record], benchmark[2], "A")


def test_duplicate_ids_rejected(benchmark, record):
    with pytest.raises(AnnotationError, match="duplicate"):
        validate_annotations([record, record], benchmark[2])


def test_rewrite_warning_remains_a_human_decision(benchmark, record):
    record["claim_text"] = "The animal sleeps."
    assert validate_annotations([record], benchmark[2]).warnings
    record["decomposed"] = True
    assert not validate_annotations([record], benchmark[2]).warnings


def test_benchmark_hash_and_duplicate_response_ids(benchmark):
    data = benchmark[1]
    with pytest.raises(AnnotationError, match="SHA-256"):
        load_responses(data + b"\n", sha256(data))
    with pytest.raises(AnnotationError, match="duplicate"):
        load_responses(data + data)
    with pytest.raises(AnnotationError, match="empty"):
        load_responses(b"")
    with pytest.raises(AnnotationError, match="JSON object"):
        load_responses(b"[]\n")


def test_explicit_zero_claim_review_is_complete(benchmark):
    _, data, responses = benchmark
    review = make_review(data, b"", "A", "fixture-v1", ["fixture-1"])
    validate_review(review, data, responses, b"", [], require_complete=True)
    review["completed_response_ids"] = []
    with pytest.raises(AnnotationError, match="not completed"):
        validate_review(review, data, responses, b"", [], require_complete=True)


def test_resume_rejects_modified_work_and_other_identity(benchmark, record):
    _, data, responses = benchmark
    annotations = export_annotations([record])
    review = make_review(data, annotations, "A", "fixture-v1", ["fixture-1"])
    with pytest.raises(AnnotationError, match="annotation hash"):
        validate_review(review, data, responses, annotations + b"\n", [record])
    review["annotator_id"] = "B"
    with pytest.raises(AnnotationError, match="session"):
        validate_review(review, data, responses, annotations, [record])


def test_validator_counts_valid_records_after_an_error(benchmark, record):
    path = benchmark[0] / "annotations.jsonl"
    invalid = dict(record, claim_id="invalid", claim_text="")
    path.write_bytes(export_annotations([invalid, record]))
    errors, _, stats = validate_file(path, benchmark[2], "A")
    assert errors
    assert stats["valid_records"] == 1


def test_cache_stub_fails_without_creating_artifacts(benchmark):
    before = sorted(p.name for p in benchmark[0].iterdir())
    with pytest.raises(NotImplementedError, match="No cache was written"):
        generate_llm_cache(benchmark[0], "fixture-model")
    assert sorted(p.name for p in benchmark[0].iterdir()) == before


def test_evaluator_requires_human_frozen_gold(benchmark, record):
    directory = benchmark[0]
    with pytest.raises(ValueError, match="gold is missing"):
        load_data(directory)
    with pytest.raises(ValueError, match="gold is missing"):
        generate_adjudication_csv([], {}, directory)
    (directory / "gold_annotations.jsonl").write_bytes(export_annotations([record]))
    with pytest.raises(ValueError, match="not frozen"):
        load_data(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["gold_annotations_hash_sha256"] = "0" * 64
    (directory / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Gold hash"):
        load_data(directory)


def test_token_f1_counts_repeated_tokens():
    assert compute_token_f1("cat cat", "cat cat") == 1.0
    assert compute_token_f1("cat cat", "cat") == pytest.approx(2 / 3)


def test_cli_empty_file_requires_explicit_completion(benchmark):
    directory, data, _ = benchmark
    annotations = directory / "annotations.jsonl"
    annotations.write_bytes(b"")
    command = [
        sys.executable,
        "-m",
        "scripts.validate_s1_annotations",
        str(annotations),
        "--responses",
        str(directory / "responses.jsonl"),
        "--require-complete",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "needs --review" in result.stdout
    review_path = directory / "review.json"
    review_path.write_text(json.dumps(make_review(data, b"", "A", "fixture-v1", ["fixture-1"])))
    result = subprocess.run(
        command + ["--review", str(review_path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "Completion verified" in result.stdout


def test_annotation_ui_add_remove_and_completion(benchmark, monkeypatch):
    from src.ui import s1_annotation

    monkeypatch.setattr(s1_annotation, "BENCHMARK_DIR", benchmark[0])
    app = AppTest.from_string("from src.ui.s1_annotation import main\nmain()")
    app.run()
    assert not app.exception
    app.text_input[0].set_value("A").run()
    assert not app.exception
    # Human-entered synthetic fixture, no model calls or benchmark outputs.
    app.text_area[0].set_value("A 🐈 sleeps.")
    next(b for b in app.button if b.label == "Add human annotation").click().run()
    assert not app.exception
    records = app.session_state["s1_records"]
    assert len(records) == 1
    assert not validate_annotations(records, benchmark[2], "A").warnings
    next(b for b in app.button if b.label.startswith("Mark response reviewed")).click().run()
    assert app.session_state["s1_completed"] == ["fixture-1"]
    next(b for b in app.button if b.label == "Remove annotation").click().run()
    assert not app.exception
    assert app.session_state["s1_records"] == []
    assert app.session_state["s1_completed"] == []
    # Completing a zero-claim response is intentional and remains exportable.
    next(b for b in app.button if b.label.startswith("Mark response reviewed")).click().run()
    assert app.session_state["s1_completed"] == ["fixture-1"]


def test_annotation_ui_protects_work_when_identity_changes(benchmark, monkeypatch):
    from src.ui import s1_annotation

    monkeypatch.setattr(s1_annotation, "BENCHMARK_DIR", benchmark[0])
    app = AppTest.from_string("from src.ui.s1_annotation import main\nmain()")
    app.run()
    app.text_input[0].set_value("A").run()
    app.text_area[0].set_value("A 🐈 sleeps.")
    next(b for b in app.button if b.label == "Add human annotation").click().run()
    app.text_input[0].set_value("B").run()
    assert app.error
    assert app.session_state["s1_records"][0]["annotator_id"] == "A"


def test_frozen_benchmark_hash_matches_metadata():
    directory = Path(__file__).resolve().parents[1] / "data" / "eval" / "s1"
    metadata = json.loads((directory / "metadata.json").read_text())
    data = (directory / "responses.jsonl").read_bytes()
    assert len(load_responses(data, metadata["responses_hash_sha256"])) == 30
    assert sha256(data.replace(b"\n", b"\r\n")) == metadata["responses_hash_sha256_original_crlf"]


def test_agreement_keeps_polarity_and_unmatched_cases_for_humans(benchmark, record):
    from src.eval.agreement import prepare_agreement

    _, data, _ = benchmark
    a = dict(record, claim_text="The animal is definitely sleeping in the house.", decomposed=True)
    b = dict(
        record,
        claim_id="fixture-B-1",
        annotator_id="B",
        claim_text="The animal is not definitely sleeping in the house.",
        decomposed=True,
    )
    extra = dict(b, claim_id="fixture-B-2", claim_text="A distinct assertion.")
    annotations_a = export_annotations([a])
    annotations_b = export_annotations([b, extra])
    report = prepare_agreement(
        data,
        annotations_a,
        make_review(data, annotations_a, "A", "fixture-v1", ["fixture-1"]),
        annotations_b,
        make_review(data, annotations_b, "B", "fixture-v1", ["fixture-1"]),
        sha256(data),
    )
    unit = report["responses"][0]
    assert len(unit["annotations_b"]) == 2  # unmatched case remains visible
    assert unit["lexical_candidates"][0]["human_same_claim"] is None
    assert not unit["human_review_complete"]
    assert report["claim_presence_kappa"] is None


def test_agreement_requires_complete_independent_reviews(benchmark):
    from src.eval.agreement import prepare_agreement

    _, data, _ = benchmark
    complete = make_review(data, b"", "A", "fixture-v1", ["fixture-1"])
    incomplete = make_review(data, b"", "B", "fixture-v1", [])
    with pytest.raises(AnnotationError, match="not completed"):
        prepare_agreement(data, b"", complete, b"", incomplete, sha256(data))
    with pytest.raises(AnnotationError, match="distinct"):
        prepare_agreement(data, b"", complete, b"", complete, sha256(data))


def test_count_alpha_has_defined_units_and_handles_no_variation():
    from src.eval.agreement import count_alpha

    assert count_alpha([1, 2, 3], [1, 2, 3]) == 1
    assert count_alpha([1, 1], [1, 1]) is None
    assert count_alpha([0, 2], [2, 0]) == pytest.approx(-0.5)
