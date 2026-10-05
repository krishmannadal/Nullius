"""S1 extraction metrics over frozen predictions and explicit human matching."""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.eval.provenance import software_identity
from src.eval.s1_annotations import read_jsonl, sha256
from src.eval.s1_workflow import (
    CLAIM_LIMITATION,
    ROOT,
    benchmark,
    load_gold,
    read_json,
    require,
    write_new,
)

ERROR_CODES = set("MSUORPBCDN")


def normalize_text(text):
    # Preserve punctuation, numeric structure, operators and negation. NFC only.
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def compute_token_f1(pred, gold):
    tokenize = lambda value: re.findall(r"\w+(?:[.,:/-]\w+)*|[^\w\s]", normalize_text(value))
    a, b = Counter(tokenize(pred)), Counter(tokenize(gold))
    if not a or not b:
        return float(a == b)
    common = sum((a & b).values())
    return 2 * common / (sum(a.values()) + sum(b.values()))


def spans_of(record):
    if record.get("source_spans") is not None:
        return record["source_spans"]
    return [record["source_span"]] if record.get("source_span") else []


def _positions(spans):
    return {i for span in spans for i in range(span["start"], span["end"])}


def span_metrics(pred, gold):
    p, g = spans_of(pred), spans_of(gold)
    a, b = _positions(p), _positions(g)
    return {
        "span_exact": float(p == g) if p else 0.0,
        "span_iou": len(a & b) / len(a | b) if a | b else 0.0,
        "start_error": abs(p[0]["start"] - g[0]["start"]) if p else None,
        "end_error": abs(p[-1]["end"] - g[-1]["end"]) if p else None,
        "span_present": float(bool(p)),
    }


def compute_span_iou(span_sys, span_gold):
    return span_metrics(
        {"source_span": span_sys.to_dict() if span_sys else None}, {"source_span": span_gold}
    )["span_iou"]


def load_data(s1_dir):
    _, gold = load_gold(s1_dir)
    _, data, responses = benchmark(s1_dir)
    return read_jsonl(data, "responses"), {
        rid: [g for g in gold if g["response_id"] == rid] for rid in responses
    }


def candidates(predictions, gold):
    weights = np.zeros((len(predictions), len(gold)))
    rows = []
    for i, p in enumerate(predictions):
        for j, g in enumerate(gold):
            score = compute_token_f1(p["claim_text"], g["claim_text"])
            exact = normalize_text(p["claim_text"]) == normalize_text(g["claim_text"])
            eligible = exact or score >= 0.8
            weights[i, j] = score if eligible else 0
            rows.append(
                {
                    "prediction_id": p["prediction_id"],
                    "gold_id": g["claim_id"],
                    "token_f1": score,
                    "normalized_exact": exact,
                    "candidate_status": "eligible" if eligible else "below_threshold",
                    "human_decision": None,
                    "final_match": False,
                    "notes": "",
                }
            )
    rr, cc = linear_sum_assignment(-weights)
    suggested = {
        (predictions[i]["prediction_id"], gold[j]["claim_id"])
        for i, j in zip(rr, cc)
        if weights[i, j] > 0
    }
    for row in rows:
        row["suggested_one_to_one"] = (row["prediction_id"], row["gold_id"]) in suggested
    return rows


def prepare_matching(s1_dir=ROOT, prediction_dir=None, output=None):
    from src.eval.s1_predictions import load_predictions

    s1_dir = Path(s1_dir)
    prediction_dir = Path(prediction_dir or s1_dir / "predictions")
    manifest, predictions = load_predictions(s1_dir, prediction_dir)
    frozen, gold = load_gold(s1_dir)
    units = []
    for extractor in manifest["extractors"]:
        for rid in frozen["response_ids"]:
            ps = [p for p in predictions if p["response_id"] == rid and p["extractor"] == extractor]
            gs = [g for g in gold if g["response_id"] == rid]
            units.append(
                {
                    "extractor": extractor,
                    "response_id": rid,
                    "candidates": candidates(ps, gs),
                    "human_review_complete": False,
                    "reviewer_id": "",
                    "reviewed_at": "",
                    "unmatched_prediction_ids": [],
                    "unmatched_gold_ids": [],
                    "errors": [],
                    "compound_relations": [],
                }
            )
    form = {
        "schema_version": "s1-matching-v2",
        "prediction_manifest_sha256": sha256((prediction_dir / "manifest.json").read_bytes()),
        "gold_manifest_sha256": sha256((s1_dir / "gold_manifest.json").read_bytes()),
        "previous_matching_sha256": None,
        "responses": units,
        "status": "human_review_required",
    }
    write_new(output or prediction_dir / "matching.draft.json", form)
    return form


def generate_adjudication_csv(responses, gold_by_resp, s1_dir):
    """Compatibility entry point: stable-ID JSON replaces the lossy CSV."""
    return prepare_matching(s1_dir)


def validate_matching(document, predictions, gold, frozen, extractors):
    units = document.get("responses", [])
    keys = [(u.get("extractor"), u.get("response_id")) for u in units]
    expected = {(e, rid) for e in extractors for rid in frozen["response_ids"]}
    require(
        len(keys) == len(set(keys)) and set(keys) == expected,
        "matching: complete response/extractor coverage required",
    )
    for unit in units:
        require(
            unit.get("human_review_complete") is True
            and bool(unit.get("reviewer_id"))
            and bool(unit.get("reviewed_at")),
            "matching: human final review incomplete",
        )
        rid, ext = unit["response_id"], unit["extractor"]
        ps = [p for p in predictions if p["response_id"] == rid and p["extractor"] == ext]
        gs = [g for g in gold if g["response_id"] == rid]
        pids, gids = {p["prediction_id"] for p in ps}, {g["claim_id"] for g in gs}
        original = candidates(ps, gs)
        submitted = unit.get("candidates", [])
        require(len(submitted) == len(original), "matching: candidates cannot be discarded")
        immutable = [
            "prediction_id",
            "gold_id",
            "token_f1",
            "normalized_exact",
            "candidate_status",
            "suggested_one_to_one",
        ]
        pairs = []
        for expected_row, row in zip(original, submitted):
            require(
                all(row.get(k) == expected_row[k] for k in immutable),
                "matching: candidate evidence changed",
            )
            require(
                row.get("human_decision") in {"accept", "reject", "reassign"},
                "matching: undecided candidate",
            )
            require(
                type(row.get("final_match")) is bool
                and row["final_match"] == (row["human_decision"] in {"accept", "reassign"}),
                "matching: inconsistent final decision",
            )
            if row["final_match"]:
                require(
                    bool(row.get("notes", "").strip()),
                    "matching: accepted/reassigned pair needs rationale",
                )
                pairs.append((row["prediction_id"], row["gold_id"]))
        matched_p, matched_g = [p for p, _ in pairs], [g for _, g in pairs]
        require(
            len(set(matched_p)) == len(matched_p) and len(set(matched_g)) == len(matched_g),
            "matching: one-to-one collision",
        )
        for field, ids in [
            ("unmatched_prediction_ids", pids - set(matched_p)),
            ("unmatched_gold_ids", gids - set(matched_g)),
        ]:
            submitted_ids = unit.get(field, [])
            require(
                len(submitted_ids) == len(set(submitted_ids)) and set(submitted_ids) == ids,
                "matching: explicit unmatched coverage required",
            )
        errors = unit.get("errors", [])
        error_keys = set()
        for error in errors:
            kind, target = error.get("target_type"), error.get("target_id")
            require(
                kind in {"prediction", "gold"}
                and target in (pids if kind == "prediction" else gids),
                "matching: invalid error target",
            )
            require((kind, target) not in error_keys, "matching: duplicate error target")
            error_keys.add((kind, target))
            codes = error.get("codes", [])
            require(
                bool(codes) and len(set(codes)) == len(codes) and set(codes) <= ERROR_CODES,
                "matching: invalid taxonomy codes",
            )
            require(bool(error.get("notes", "").strip()), "matching: error rationale required")
        for kind, ids, code in [
            ("prediction", pids - set(matched_p), "S"),
            ("gold", gids - set(matched_g), "M"),
        ]:
            for target in ids:
                require(
                    any(
                        e["target_type"] == kind and e["target_id"] == target and code in e["codes"]
                        for e in errors
                    ),
                    f"matching: unmatched {kind} requires {code} classification",
                )
        source = next(u for u in frozen["response_reviews"] if u["response_id"] == rid)
        groups = source.get("compound_groups", [])
        relations = unit.get("compound_relations", [])
        require(
            sorted(r.get("group_id", "") for r in relations)
            == sorted(g["group_id"] for g in groups),
            "matching: explicit compound relation coverage required",
        )
        used = set()
        for relation in relations:
            related = relation.get("prediction_ids", [])
            require(
                len(set(related)) == len(related)
                and set(related) <= pids
                and not set(related) & used,
                "matching: invalid/duplicate compound prediction relation",
            )
            used.update(related)
            require(
                type(relation.get("under_split")) is bool
                and type(relation.get("over_split")) is bool,
                "matching: human split judgments required",
            )
            require(
                bool(relation.get("notes", "").strip()), "matching: compound rationale required"
            )
    return units


def prf(tp, fp, fn):
    if tp + fp + fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn),
    }


def mean_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def response_metrics(unit, predictions, gold, groups):
    ps = {p["prediction_id"]: p for p in predictions}
    gs = {g["claim_id"]: g for g in gold}
    pairs = [(r["prediction_id"], r["gold_id"]) for r in unit["candidates"] if r["final_match"]]
    tp, fp, fn = len(pairs), len(ps) - len(pairs), len(gs) - len(pairs)
    result = dict(response_id=unit["response_id"], tp=tp, fp=fp, fn=fn, **prf(tp, fp, fn))
    spans = [span_metrics(ps[p], gs[g]) for p, g in pairs]
    for key in ("span_exact", "span_iou", "start_error", "end_error", "span_present"):
        result[key] = mean_defined([s[key] for s in spans])
    result["span_matched_pairs"] = tp
    result["span_boundary_pairs"] = sum(s["start_error"] is not None for s in spans)
    group_ids = {g for group in groups for g in group["gold_claim_ids"]}
    related = {p for rel in unit["compound_relations"] for p in rel["prediction_ids"]}
    membership = {group["group_id"]: set(group["gold_claim_ids"]) for group in groups}
    group_tp = sum(
        p in rel["prediction_ids"] and g in membership[rel["group_id"]]
        for rel in unit["compound_relations"]
        for p, g in pairs
    )
    result.update(
        compound_claim_recall=group_tp / len(group_ids) if group_ids else None,
        compound_claim_precision=group_tp / len(related) if related else None,
        split_ratio=len(related) / len(group_ids) if group_ids else None,
        under_split_rate=mean_defined(
            [float(r["under_split"]) for r in unit["compound_relations"]]
        ),
        over_split_rate=mean_defined([float(r["over_split"]) for r in unit["compound_relations"]]),
    )
    result["errors"] = {
        code: sum(code in e["codes"] for e in unit["errors"]) for code in sorted(ERROR_CODES)
    }
    return result


def bootstrap_ci(data, n_resamples=10000, seed=1337):
    require(n_resamples > 0, "n_resamples must be positive")
    if not data:
        return None, None, None
    values = np.asarray(data, dtype=float)
    require(bool(np.isfinite(values).all()), "bootstrap requires finite defined values")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def evaluate_units(units, predictions, gold, frozen, seed=1337, n_resamples=10000):
    by_extractor = {}
    metric_names = [
        "precision",
        "recall",
        "f1",
        "span_exact",
        "span_iou",
        "start_error",
        "end_error",
        "span_present",
        "compound_claim_recall",
        "compound_claim_precision",
        "split_ratio",
        "under_split_rate",
        "over_split_rate",
    ]
    for ext in ("spacy_sentence", "llm"):
        rows = []
        for unit in units:
            if unit["extractor"] != ext:
                continue
            rid = unit["response_id"]
            groups = next(r for r in frozen["response_reviews"] if r["response_id"] == rid).get(
                "compound_groups", []
            )
            rows.append(
                response_metrics(
                    unit,
                    [p for p in predictions if p["extractor"] == ext and p["response_id"] == rid],
                    [g for g in gold if g["response_id"] == rid],
                    groups,
                )
            )
        totals = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn")}
        by_extractor[ext] = {
            "per_response": rows,
            "micro": {**totals, **prf(**totals)},
            "macro": {k: mean_defined([r[k] for r in rows]) for k in metric_names},
            "errors": {c: sum(r["errors"][c] for r in rows) for c in sorted(ERROR_CODES)},
        }
        by_extractor[ext]["macro_uncertainty"] = {}
        for key in metric_names:
            defined = [r[key] for r in rows if r[key] is not None]
            avg, lo, hi = bootstrap_ci(defined, n_resamples, seed)
            by_extractor[ext]["macro_uncertainty"][key] = {
                "mean": avg,
                "ci95": [lo, hi],
                "n_defined_responses": len(defined),
            }
    a = {r["response_id"]: r for r in by_extractor["spacy_sentence"]["per_response"]}
    b = {r["response_id"]: r for r in by_extractor["llm"]["per_response"]}
    require(set(a) == set(b) == set(frozen["response_ids"]), "paired response sets differ")
    differences, paired = [], {}
    for rid in frozen["response_ids"]:
        differences.append(
            {
                "response_id": rid,
                **{
                    k: b[rid][k] - a[rid][k]
                    if b[rid][k] is not None and a[rid][k] is not None
                    else None
                    for k in metric_names
                },
            }
        )
    for key in metric_names:
        values = [r[key] for r in differences if r[key] is not None]
        avg, lo, hi = bootstrap_ci(values, n_resamples, seed)
        paired[key] = {
            "mean_difference": avg,
            "median_difference": float(np.median(values)) if values else None,
            "ci95": [lo, hi],
            "n_paired_responses": len(values),
        }
    require(n_resamples > 0 and bool(a), "bootstrap needs positive resamples and responses")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(n_resamples, len(a)))
    micro_samples = {}
    for ext, table in [("spacy_sentence", a), ("llm", b)]:
        counts = np.array(
            [[table[rid][k] for k in ("tp", "fp", "fn")] for rid in frozen["response_ids"]]
        )
        sums = counts[indices].sum(axis=1)
        micro_samples[ext] = {
            key: np.array([prf(*row)[key] for row in sums]) for key in ("precision", "recall", "f1")
        }
        by_extractor[ext]["micro_ci95"] = {
            key: np.percentile(values, [2.5, 97.5]).tolist()
            for key, values in micro_samples[ext].items()
        }
    paired_micro = {
        key: {
            "difference": by_extractor["llm"]["micro"][key]
            - by_extractor["spacy_sentence"]["micro"][key],
            "ci95": np.percentile(
                micro_samples["llm"][key] - micro_samples["spacy_sentence"][key], [2.5, 97.5]
            ).tolist(),
            "n_paired_responses": len(a),
        }
        for key in ("precision", "recall", "f1")
    }
    return {
        "schema_version": "s1-report-v2",
        "limitation": CLAIM_LIMITATION,
        "estimand": "macro mean of response-level metrics; paired LLM minus spaCy",
        "undefined_policy": "paired complete cases per metric, with denominators reported",
        "seed": seed,
        "n_resamples": n_resamples,
        "n_responses": len(a),
        "n_gold_claims": len(gold),
        "extractors": by_extractor,
        "paired": paired,
        "paired_micro": paired_micro,
        "per_response_differences": differences,
    }


def _load_matching(s1_dir, prediction_dir, matching_path):
    from src.eval.s1_predictions import load_predictions

    s1_dir = Path(s1_dir)
    prediction_dir = Path(prediction_dir or s1_dir / "predictions")
    frozen, gold = load_gold(s1_dir)
    manifest, predictions = load_predictions(s1_dir, prediction_dir)
    matching_path = Path(matching_path or prediction_dir / "matching.final.json")
    document = read_json(matching_path)
    require(
        document.get("prediction_manifest_sha256")
        == sha256((prediction_dir / "manifest.json").read_bytes()),
        "stale prediction matching",
    )
    require(
        document.get("gold_manifest_sha256")
        == sha256((s1_dir / "gold_manifest.json").read_bytes()),
        "stale gold matching",
    )
    units = validate_matching(document, predictions, gold, frozen, manifest["extractors"])
    return document, units, predictions, gold, frozen, matching_path


def freeze_matching(s1_dir=ROOT, prediction_dir=None, matching_path=None):
    document, _, _, _, _, matching_path = _load_matching(s1_dir, prediction_dir, matching_path)
    from src.core.types import utcnow_iso

    previous = document.get("previous_matching_sha256")
    if previous is not None:
        prior_path = matching_path.parent / document.get("previous_matching_file", "")
        require(
            prior_path.resolve().parent == matching_path.parent.resolve()
            and prior_path.resolve() != matching_path.resolve(),
            "invalid previous decision reference",
        )
        require(
            prior_path.is_file() and sha256(prior_path.read_bytes()) == previous,
            "previous human decisions must be retained unchanged",
        )
    receipt = {
        "schema_version": "s1-matching-freeze-v2",
        "frozen_at": utcnow_iso(),
        "matching_sha256": sha256(matching_path.read_bytes()),
        "gold_manifest_sha256": document["gold_manifest_sha256"],
        "prediction_manifest_sha256": document["prediction_manifest_sha256"],
        "previous_matching_sha256": previous,
    }
    write_new(matching_path.with_suffix(".manifest.json"), receipt)
    return receipt


def run_evaluation(
    s1_dir=ROOT, prediction_dir=None, matching_path=None, output=None, seed=1337, n_resamples=10000
):
    document, units, predictions, gold, frozen, matching_path = _load_matching(
        s1_dir, prediction_dir, matching_path
    )
    receipt = read_json(matching_path.with_suffix(".manifest.json"))
    require(
        receipt.get("schema_version") == "s1-matching-freeze-v2"
        and receipt.get("matching_sha256") == sha256(matching_path.read_bytes()),
        "Matching decisions are not frozen or have changed",
    )
    require(
        all(
            receipt.get(k) == document[k]
            for k in ("gold_manifest_sha256", "prediction_manifest_sha256")
        ),
        "matching freeze input identities differ",
    )
    report = evaluate_units(units, predictions, gold, frozen, seed, n_resamples)
    report["software_identity"] = software_identity()
    report["inputs"] = {
        "matching_sha256": sha256(matching_path.read_bytes()),
        "prediction_manifest_sha256": document["prediction_manifest_sha256"],
        "gold_manifest_sha256": document["gold_manifest_sha256"],
    }
    destination = Path(output or matching_path.parent / "report.json")
    require(
        not destination.exists() and not destination.with_suffix(".md").exists(),
        "Report destination already exists; choose a new output path",
    )
    write_new(destination, report)
    markdown = [
        "# S1 extraction pilot report",
        "",
        CLAIM_LIMITATION,
        "",
        f"Responses: {report['n_responses']}; gold claims: {report['n_gold_claims']}",
        f"Paired bootstrap: {n_resamples} response resamples, seed {seed}.",
        "",
        "| Extractor | Micro precision | Micro recall | Micro F1 | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in report["extractors"].items():
        markdown.append(
            f"| {name} | {values['micro']['precision']:.4f} | {values['micro']['recall']:.4f} | {values['micro']['f1']:.4f} | {values['macro']['f1']:.4f} |"
        )
    markdown += [
        "",
        "All span, decomposition, taxonomy, paired differences and confidence intervals",
        "are retained in the accompanying JSON. Undefined metrics are null, not zero.",
        "These values describe this frozen curated pilot only.",
    ]
    with destination.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(markdown) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-dir", type=Path, default=ROOT)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--matching", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-resamples", type=int, default=10000)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate-csv", "--prepare-matching", action="store_true", dest="prepare")
    group.add_argument("--evaluate", action="store_true")
    group.add_argument("--freeze-matching", action="store_true")
    args = parser.parse_args()
    try:
        if args.prepare:
            prepare_matching(args.s1_dir, args.prediction_dir, args.output)
        elif args.freeze_matching:
            freeze_matching(args.s1_dir, args.prediction_dir, args.matching)
        else:
            run_evaluation(
                args.s1_dir,
                args.prediction_dir,
                args.matching,
                args.output,
                args.seed,
                args.n_resamples,
            )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Blocked: {exc}\n")


if __name__ == "__main__":
    main()
