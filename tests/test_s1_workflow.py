"""Synthetic unit fixtures ONLY, always in tmp_path. These are not S1 observations."""

import json
from pathlib import Path

import pytest

from src.eval.agreement import prepare_agreement
from src.eval.extraction_eval import (
    bootstrap_ci,
    candidates,
    compute_token_f1,
    normalize_text,
    prf,
    response_metrics,
    span_metrics,
)
from src.eval.s1_annotations import (
    AnnotationError,
    canonical_bytes,
    canonical_sha256,
    export_annotations,
    load_responses,
    make_review,
    sha256,
    validate_annotations,
)
from src.eval.s1_workflow import benchmark, freeze_gold, load_gold, write_new


@pytest.fixture
def complete_human_fixture(tmp_path):
    root = tmp_path / "fixture"
    root.mkdir()
    responses = {
        "fixture-1": "A moth may fly.",
        "fixture-2": "Lena sings and dances.",
        "fixture-3": "Hello!",
    }
    data = export_annotations([{"response_id": k, "text": v} for k, v in responses.items()])
    (root / "responses.jsonl").write_bytes(data)
    (root / "ANNOTATION_GUIDELINES.md").write_text("Fixture guidelines only\n")
    (root / "EVALUATION_PROTOCOL.md").write_text("Fixture protocol only\n")
    metadata = {
        "benchmark_version": "fixture-only",
        "response_count": 3,
        "response_ids": list(responses),
        "responses_hash_sha256": canonical_sha256(data),
        "guidelines_sha256": canonical_sha256((root / "ANNOTATION_GUIDELINES.md").read_bytes()),
        "evaluation_protocol_sha256": canonical_sha256(
            (root / "EVALUATION_PROTOCOL.md").read_bytes()
        ),
        "provenance_status": "curated synthetic pilot / provenance-unverified",
    }
    write_new(root / "metadata.json", metadata)

    def claim(cid, rid, text, spans, operations, group=None, hedged=False):
        return {
            "schema_version": "s1-claim-v2",
            "claim_id": cid,
            "response_id": rid,
            "claim_text": text,
            "source_span": spans[0],
            "source_spans": spans,
            "operations": operations,
            "decomposed": "decomposition" in operations,
            "compound_id": group,
            "hedged": hedged,
            "annotator_id": "adjudicator",
            "notes": "Synthetic fixture",
        }

    gold = [
        claim("g1", "fixture-1", "A moth may fly.", [{"start": 0, "end": 15}], [], hedged=True),
        claim(
            "g2",
            "fixture-2",
            "Lena sings",
            [{"start": 0, "end": 10}],
            ["decomposition"],
            "compound-1",
        ),
        claim(
            "g3",
            "fixture-2",
            "Lena dances",
            [{"start": 0, "end": 4}, {"start": 15, "end": 21}],
            ["decomposition", "rewrite"],
            "compound-1",
        ),
    ]
    # Human decisions here are synthetic test inputs, never written into the real benchmark.
    submissions = {}
    reviews = {}
    for identity in ("A", "B"):
        records = [dict(g, claim_id=identity + g["claim_id"], annotator_id=identity) for g in gold]
        raw = export_annotations(records)
        (root / f"annotations_{identity}.jsonl").write_bytes(raw)
        review = make_review(
            data, raw, identity, "fixture-v2", list(responses), metadata["guidelines_sha256"], True
        )
        write_new(root / f"review_{identity}.json", review)
        submissions[identity], reviews[identity] = raw, review
    agreement = prepare_agreement(
        data, submissions["A"], reviews["A"], submissions["B"], reviews["B"], canonical_sha256(data)
    )
    for unit in agreement["responses"]:
        unit.update(
            human_review_complete=True,
            human_matches=[
                {"claim_a_id": "A" + g["claim_id"], "claim_b_id": "B" + g["claim_id"]}
                for g in gold
                if g["response_id"] == unit["response_id"]
            ],
            unmatched_a=[],
            unmatched_b=[],
        )
    write_new(root / "agreement.json", agreement)
    (root / "gold_annotations.jsonl").write_bytes(export_annotations(gold))
    adjudication = {
        k: agreement[k]
        for k in (
            "responses_sha256",
            "guidelines_sha256",
            "annotations_a_sha256",
            "annotations_b_sha256",
        )
    }
    adjudication.update(
        adjudicator_id="fixture-human",
        no_extractor_outputs_used=True,
        agreement_sha256=sha256((root / "agreement.json").read_bytes()),
        gold_sha256=sha256((root / "gold_annotations.jsonl").read_bytes()),
        responses=[
            {
                "response_id": rid,
                "human_review_complete": True,
                "gold_claim_ids": [g["claim_id"] for g in gold if g["response_id"] == rid],
                "zero_claims": rid == "fixture-3",
                "notes": "Synthetic fixture decision",
                "compound_groups": [{"group_id": "compound-1", "gold_claim_ids": ["g2", "g3"]}]
                if rid == "fixture-2"
                else [],
            }
            for rid in responses
        ],
    )
    write_new(root / "adjudication.json", adjudication)
    return root, responses, gold


def test_real_benchmark_and_all_calibration_offsets():
    root = Path(__file__).resolve().parents[1] / "data/eval/s1"
    metadata, data, responses = benchmark(root)
    assert len(responses) == 30
    assert (
        canonical_sha256(data) == "8ef976d32392724acb468821a0314787b7e3c6c563e3556381e6b1a5e1672356"
    )
    assert metadata["source_subgroup_claims_permitted"] is False
    for example in json.loads((root / "calibration_examples.json").read_text()):
        for span in example["spans"]:
            assert example["text"][span["start"] : span["end"]] == span["text"]


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_transport_identity(newline):
    lf = b'{"response_id":"test","text":"A."}\n'
    raw = lf.replace(b"\n", newline)
    assert canonical_sha256(raw) == sha256(lf)
    assert load_responses(raw, sha256(lf)) == {"test": "A."}
    assert (sha256(raw) == sha256(lf)) == (newline == b"\n")


def test_mixed_newlines_and_lone_cr():
    assert canonical_bytes(b"a\r\nb\n") == b"a\nb\n"
    with pytest.raises(AnnotationError, match="lone CR"):
        canonical_bytes(b"a\rb\n")


def test_complete_gold_freeze_and_transport_portability(complete_human_fixture):
    root, _, gold = complete_human_fixture
    freeze_gold(root)
    assert load_gold(root)[1] == gold
    p = root / "responses.jsonl"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert load_gold(root)[1] == gold
    with pytest.raises(FileExistsError):
        freeze_gold(root)


@pytest.mark.parametrize(
    "change",
    [
        "missing_response",
        "duplicate_response",
        "zero_missing",
        "incomplete",
        "wrong_guidelines",
        "wrong_benchmark",
        "wrong_gold",
        "empty_gold",
        "missing_a",
        "wrong_annotator",
        "contamination",
    ],
)
def test_gold_gates_fail_closed(complete_human_fixture, change):
    root, _, _ = complete_human_fixture
    target = root / "adjudication.json"
    doc = json.loads(target.read_text())
    if change == "missing_response":
        doc["responses"].pop()
    elif change == "duplicate_response":
        doc["responses"].append(doc["responses"][0])
    elif change == "zero_missing":
        doc["responses"][-1].pop("zero_claims")
    elif change == "incomplete":
        doc["responses"][0]["human_review_complete"] = False
    elif change == "wrong_guidelines":
        doc["guidelines_sha256"] = "0" * 64
    elif change == "wrong_benchmark":
        doc["responses_sha256"] = "0" * 64
    elif change == "wrong_gold":
        doc["gold_sha256"] = "0" * 64
    elif change == "empty_gold":
        (root / "gold_annotations.jsonl").write_bytes(b"")
    elif change == "missing_a":
        (root / "annotations_A.jsonl").unlink()
    elif change == "wrong_annotator":
        p = root / "review_A.json"
        r = json.loads(p.read_text())
        r["annotator_id"] = "B"
        p.write_text(json.dumps(r))
    elif change == "contamination":
        p = root / "annotations_A.jsonl"
        records = [json.loads(x) for x in p.read_text().splitlines()]
        records[0]["system_verdict"] = "Supported"
        p.write_bytes(export_annotations(records))
    target.write_text(json.dumps(doc))
    with pytest.raises((AnnotationError, OSError)):
        freeze_gold(root)
    assert not (root / "gold_manifest.json").exists()


@pytest.mark.parametrize(
    "left,right",
    [
        ("x > 3", "x < 3"),
        ("3.14", "314"),
        ("not true", "true"),
        ("2020-01-02", "20200102"),
        ("5 m/s", "5 ms"),
    ],
)
def test_normalization_preserves_semantics(left, right):
    assert normalize_text(left) != normalize_text(right)
    assert compute_token_f1(left, right) < 1


@pytest.mark.parametrize(
    "text",
    [
        "A moth may fly.",
        "Mira said the vessel sank.",
        "If pressure rises, the valve opens.",
        "The key is brass or copper.",
        "The lamp is not lit.",
    ],
)
def test_qualifiers_survive_formal_annotation(text):
    record = {
        "schema_version": "s1-claim-v2",
        "response_id": "fixture",
        "claim_id": "a",
        "claim_text": text,
        "source_span": {"start": 0, "end": len(text)},
        "source_spans": [{"start": 0, "end": len(text)}],
        "operations": [],
        "compound_id": None,
        "decomposed": False,
        "hedged": True,
        "annotator_id": "A",
    }
    assert not validate_annotations([record], {"fixture": text}, formal=True).warnings


def _pred(pid, text="A moth may fly.", span=True):
    return {
        "prediction_id": pid,
        "claim_text": text,
        "source_span": {"start": 0, "end": len(text)} if span else None,
    }


@pytest.mark.parametrize(
    "tp,fp,fn,expected",
    [
        (0, 0, 0, (1, 1, 1)),
        (0, 1, 0, (0, 0, 0)),
        (0, 0, 1, (0, 0, 0)),
        (1, 1, 1, (0.5, 0.5, 0.5)),
        (2, 1, 0, (2 / 3, 1, 0.8)),
    ],
)
def test_actual_claim_values(tp, fp, fn, expected):
    assert tuple(prf(tp, fp, fn).values()) == pytest.approx(expected)


def test_duplicates_remain_distinct_candidates():
    ps = [_pred("p1"), _pred("p2")]
    gs = [
        {"claim_id": "g1", "claim_text": "A moth may fly."},
        {"claim_id": "g2", "claim_text": "A moth may fly."},
    ]
    rows = candidates(ps, gs)
    assert len(rows) == 4
    assert sum(r["suggested_one_to_one"] for r in rows) == 2
    assert all(r["human_decision"] is None for r in rows)


def test_missing_exact_and_approximate_spans():
    gold = {"source_span": {"start": 0, "end": 10}}
    assert span_metrics({"source_span": None}, gold) == {
        "span_exact": 0,
        "span_iou": 0,
        "start_error": None,
        "end_error": None,
        "span_present": 0,
    }
    exact = span_metrics(gold, gold)
    assert (exact["span_exact"], exact["span_iou"], exact["start_error"], exact["end_error"]) == (
        1,
        1,
        0,
        0,
    )
    partial = span_metrics({"source_span": {"start": 2, "end": 8}}, gold)
    assert partial["span_iou"] == 0.6
    assert partial["start_error"] == partial["end_error"] == 2
    multi = {"source_spans": [{"start": 0, "end": 2}, {"start": 8, "end": 10}]}
    assert span_metrics(multi, gold)["span_iou"] == 0.4


def test_decomposition_and_multiple_error_codes():
    ps = [_pred("p1", "Lena sings and dances.")]
    gold = [
        {"claim_id": "g1", "claim_text": "Lena sings", "source_span": {"start": 0, "end": 10}},
        {"claim_id": "g2", "claim_text": "Lena dances", "source_span": {"start": 15, "end": 21}},
    ]
    unit = {
        "response_id": "fixture",
        "candidates": [{"prediction_id": "p1", "gold_id": "g1", "final_match": True}],
        "compound_relations": [
            {"group_id": "c1", "prediction_ids": ["p1"], "under_split": True, "over_split": False}
        ],
        "errors": [{"codes": ["U", "P"]}, {"codes": ["M"]}],
    }
    r = response_metrics(unit, ps, gold, [{"group_id": "c1", "gold_claim_ids": ["g1", "g2"]}])
    assert (r["tp"], r["fp"], r["fn"], r["f1"]) == (1, 0, 1, 2 / 3)
    assert r["compound_claim_recall"] == 0.5 and r["compound_claim_precision"] == 1
    assert r["split_ratio"] == 0.5 and r["under_split_rate"] == 1 and r["over_split_rate"] == 0
    assert r["errors"]["U"] == r["errors"]["P"] == r["errors"]["M"] == 1


def test_bootstrap_actual_values_and_seed():
    assert bootstrap_ci([0.25] * 3) == (0.25, 0.25, 0.25)
    assert bootstrap_ci([]) == (None, None, None)
    assert bootstrap_ci([-0.2, 0, 0.5], seed=4) == bootstrap_ci([-0.2, 0, 0.5], seed=4)
    assert bootstrap_ci([-0.2, 0, 0.5], seed=4)[0] == pytest.approx(0.1)


def test_provider_missing_credentials_creates_no_cache(complete_human_fixture, monkeypatch):
    from scripts.generate_s1_llm_cache import generate_llm_cache

    root, _, _ = complete_human_fixture
    freeze_gold(root)
    monkeypatch.delenv("NULLIUS_LLM_API_KEY", raising=False)
    with pytest.raises(AnnotationError, match="No cache was written"):
        generate_llm_cache(
            root, "fixture-model", provider="fixture", base_url="https://example.invalid/v1"
        )
    assert not (root / "llm_run").exists()


def test_provider_raw_and_duplicates_are_preserved(complete_human_fixture, monkeypatch):
    from scripts.generate_s1_llm_cache import generate_llm_cache
    from src.eval.s1_predictions import load_provider_run

    root, responses, _ = complete_human_fixture
    freeze_gold(root)
    monkeypatch.setenv("NULLIUS_LLM_API_KEY", "fixture-secret-never-persist")

    def fake_provider(url, key, payload):
        text = payload["messages"][-1]["content"].split("\n", 1)[1]
        claim = {
            "claim_text": text,
            "source_spans": [],
            "operations": [],
            "decomposed": False,
            "hedged": False,
        }
        return {
            "model": "fixture-model",
            "choices": [{"message": {"content": json.dumps({"claims": [claim, claim]})}}],
        }

    result = generate_llm_cache(
        root,
        "fixture-model",
        provider="synthetic-test-provider",
        base_url="https://example.invalid/v1",
        complete=fake_provider,
    )
    assert all(len(r["processed_claims"]) == 2 for r in result["responses"])
    assert len(load_provider_run(root / "llm_run", root)["responses"]) == len(responses)
    assert "fixture-secret-never-persist" not in (root / "llm_run/manifest.json").read_text()


def test_write_new_preserves_prior_decision(tmp_path):
    path = tmp_path / "decision.json"
    write_new(path, {"decision": "original"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(path, {"decision": "changed"})
    assert path.read_bytes() == before


@pytest.fixture
def frozen_prediction_fixture(complete_human_fixture, monkeypatch):
    from types import SimpleNamespace

    from scripts.generate_s1_llm_cache import generate_llm_cache
    from src.components import extractors
    from src.core.types import Claim, SourceSpan
    from src.eval.extraction_eval import prepare_matching
    from src.eval.s1_predictions import freeze_predictions, load_predictions

    root, responses, gold = complete_human_fixture
    freeze_gold(root)
    monkeypatch.setenv("NULLIUS_LLM_API_KEY", "fixture-only")

    def fake_provider(url, key, payload):
        text = payload["messages"][-1]["content"].split("\n", 1)[1]
        rid = next(r for r, t in responses.items() if t == text)
        claims = [
            {k: g[k] for k in ("claim_text", "source_spans", "operations", "decomposed", "hedged")}
            for g in gold
            if g["response_id"] == rid
        ]
        return {
            "model": "fixture-only",
            "choices": [{"message": {"content": json.dumps({"claims": claims})}}],
        }

    generate_llm_cache(
        root,
        "fixture-only",
        provider="synthetic-test-interface",
        base_url="https://example.invalid/v1",
        complete=fake_provider,
    )

    class FixtureSentenceExtractor:
        def __init__(self, model):
            self.nlp = SimpleNamespace(to_bytes=lambda: b"synthetic-fixture-no-model")

        def extract(self, text):
            return [Claim.new("fixture", text, "spacy_sentence", SourceSpan(0, len(text)))]

    monkeypatch.setattr(extractors, "SpacySentenceExtractor", FixtureSentenceExtractor)
    freeze_predictions(root)
    manifest, predictions = load_predictions(root)
    form = prepare_matching(root)
    for unit in form["responses"]:
        rid, ext = unit["response_id"], unit["extractor"]
        ps = {
            p["prediction_id"]: p
            for p in predictions
            if p["response_id"] == rid and p["extractor"] == ext
        }
        gs = {g["claim_id"]: g for g in gold if g["response_id"] == rid}
        matched_p, matched_g = set(), set()
        for pair in unit["candidates"]:
            accepted = ps[pair["prediction_id"]]["claim_text"] == gs[pair["gold_id"]]["claim_text"]
            pair.update(
                human_decision="accept" if accepted else "reject",
                final_match=accepted,
                notes="Synthetic fixture judgment",
            )
            if accepted:
                matched_p.add(pair["prediction_id"])
                matched_g.add(pair["gold_id"])
        unit.update(
            human_review_complete=True,
            reviewer_id="fixture-reviewer",
            reviewed_at="2026-01-01T00:00:00Z",
            unmatched_prediction_ids=sorted(set(ps) - matched_p),
            unmatched_gold_ids=sorted(set(gs) - matched_g),
        )
        unit["errors"] = [
            {
                "target_type": "prediction",
                "target_id": p,
                "codes": ["S"],
                "notes": "Fixture unmatched",
            }
            for p in set(ps) - matched_p
        ] + [
            {"target_type": "gold", "target_id": g, "codes": ["M"], "notes": "Fixture missed"}
            for g in set(gs) - matched_g
        ]
        if rid == "fixture-2":
            unit["compound_relations"] = [
                {
                    "group_id": "compound-1",
                    "prediction_ids": list(ps),
                    "under_split": ext == "spacy_sentence",
                    "over_split": False,
                    "notes": "Fixture compound judgment",
                }
            ]
    path = root / "predictions/matching.final.json"
    write_new(path, form)
    return root, manifest, predictions, form


def test_end_to_end_frozen_workflow_and_actual_metrics(frozen_prediction_fixture):
    from src.eval.extraction_eval import freeze_matching, run_evaluation

    root, _, _, _ = frozen_prediction_fixture
    with pytest.raises(AnnotationError, match="Missing or invalid artifact"):
        run_evaluation(root, n_resamples=100)
    freeze_matching(root)
    report = run_evaluation(root, n_resamples=1000)
    assert report["n_responses"] == 3 and report["n_gold_claims"] == 3
    assert report["extractors"]["llm"]["micro"]["f1"] == 1
    assert report["extractors"]["spacy_sentence"]["micro"]["f1"] == pytest.approx(1 / 3)
    assert report["paired"]["f1"]["mean_difference"] == pytest.approx(2 / 3)
    assert report["paired"]["span_iou"]["n_paired_responses"] == 1
    assert report["paired"]["span_iou"]["ci95"] == [0, 0]
    assert report["extractors"]["llm"]["macro_uncertainty"]["f1"]["ci95"] == [1, 1]
    assert (root / "predictions/report.md").exists()
    assert "provenance-unverified" in (root / "predictions/report.md").read_text()


@pytest.mark.parametrize(
    "change",
    [
        "collision",
        "pending",
        "missing_unit",
        "lost_candidate",
        "lost_unmatched",
        "bad_relation",
        "missing_error",
    ],
)
def test_matching_rejects_incomplete_or_conflicting_decisions(frozen_prediction_fixture, change):
    from src.eval.extraction_eval import validate_matching

    root, manifest, predictions, form = frozen_prediction_fixture
    frozen, gold = load_gold(root)
    unit = next(
        u for u in form["responses"] if u["extractor"] == "llm" and u["response_id"] == "fixture-2"
    )
    spacy = next(
        u
        for u in form["responses"]
        if u["extractor"] == "spacy_sentence" and u["response_id"] == "fixture-2"
    )
    if change == "collision":
        for c in unit["candidates"]:
            c.update(human_decision="accept", final_match=True)
    elif change == "pending":
        unit["human_review_complete"] = False
    elif change == "missing_unit":
        form["responses"].pop()
    elif change == "lost_candidate":
        unit["candidates"].pop()
    elif change == "lost_unmatched":
        spacy["unmatched_gold_ids"] = []
    elif change == "bad_relation":
        unit["compound_relations"] = []
    elif change == "missing_error":
        spacy["errors"] = []
    with pytest.raises(AnnotationError):
        validate_matching(form, predictions, gold, frozen, manifest["extractors"])


def test_prediction_and_matching_tamper_detection(frozen_prediction_fixture):
    from src.eval.extraction_eval import freeze_matching, run_evaluation
    from src.eval.s1_predictions import load_predictions

    root, _, _, _ = frozen_prediction_fixture
    freeze_matching(root)
    path = root / "predictions/matching.final.json"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(AnnotationError, match="not frozen or have changed"):
        run_evaluation(root)
    path.write_bytes(original)
    predictions = root / "predictions/s1_predictions.jsonl"
    predictions.write_bytes(predictions.read_bytes() + b"\n")
    with pytest.raises(AnnotationError, match="artifact changed"):
        load_predictions(root)


@pytest.mark.parametrize("code", list("MSUORPBCDN"))
def test_taxonomy_values_are_not_mutually_exclusive(code):
    unit = {
        "response_id": "fixture",
        "candidates": [],
        "compound_relations": [],
        "errors": [{"codes": sorted({"S", code})}],
    }
    values = response_metrics(unit, [_pred("p")], [], [])
    assert values["fp"] == 1 and values["f1"] == 0
    assert values["errors"][code] == 1 and values["errors"]["S"] == 1


def test_duplicate_gold_assertions_fail_formal_validation(complete_human_fixture):
    _, responses, gold = complete_human_fixture
    with pytest.raises(AnnotationError, match="duplicate assertion"):
        validate_annotations(
            [gold[0], dict(gold[0], claim_id="different-id")], responses, formal=True
        )


def test_duplicate_prediction_costs_false_positive():
    ps = [_pred("p1"), _pred("p2")]
    gold = [
        {"claim_id": "g", "claim_text": "A moth may fly.", "source_span": {"start": 0, "end": 15}}
    ]
    unit = {
        "response_id": "fixture",
        "candidates": [{"prediction_id": "p1", "gold_id": "g", "final_match": True}],
        "compound_relations": [],
        "errors": [{"codes": ["S", "D"]}],
    }
    values = response_metrics(unit, ps, gold, [])
    assert (
        values["precision"] == 0.5
        and values["recall"] == 1
        and values["f1"] == pytest.approx(2 / 3)
    )
