"""Freeze both extractor outputs once, retaining raw outputs, duplicates and spans."""

from __future__ import annotations

import argparse
import importlib.metadata
import uuid
from pathlib import Path

from scripts.generate_s1_llm_cache import PROCESSING_VERSION, SYSTEM_PROMPT, parse_claims
from src.core.config import git_is_dirty, git_sha
from src.core.types import utcnow_iso
from src.eval.provenance import software_identity
from src.eval.s1_annotations import export_annotations, read_jsonl, sha256
from src.eval.s1_workflow import ROOT, benchmark, load_gold, read_json, require, write_new


def load_provider_run(directory, s1_dir):
    directory, s1_dir = Path(directory), Path(s1_dir)
    manifest = read_json(directory / "manifest.json")
    metadata, _, responses = benchmark(s1_dir)
    require(
        manifest.get("responses_sha256") == metadata["responses_hash_sha256"],
        "provider benchmark identity mismatch",
    )
    require(
        manifest.get("schema_version") == "s1-real-provider-run-v2",
        "real provider manifest required",
    )
    require(
        all(
            isinstance(manifest.get(k), str) and manifest[k].strip()
            for k in ("provider", "requested_model", "endpoint", "generated_at")
        ),
        "provider identity or timestamp missing",
    )
    require(
        manifest.get("gold_manifest_sha256")
        == sha256((s1_dir / "gold_manifest.json").read_bytes()),
        "provider run predates/differs from frozen gold",
    )
    require(
        manifest.get("system_prompt") == SYSTEM_PROMPT
        and manifest.get("system_prompt_sha256") == sha256(SYSTEM_PROMPT.encode()),
        "provider prompt identity mismatch",
    )
    require(
        manifest.get("processing_version") == PROCESSING_VERSION, "processing identity mismatch"
    )
    units = manifest.get("responses", [])
    ids = [u.get("response_id") for u in units]
    require(
        len(ids) == len(set(ids)) and set(ids) == set(responses),
        "provider response coverage incomplete",
    )
    for unit in units:
        path = directory / unit["raw_file"]
        require(
            path.resolve().parent == directory.resolve(),
            "raw reference must remain in provider run",
        )
        require(sha256(path.read_bytes()) == unit["raw_sha256"], "provider raw output changed")
        raw = read_json(path)
        rid = unit["response_id"]
        expected_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Extract from this exact response:\n" + responses[rid]},
        ]
        require(
            raw.get("response_id") == rid and raw.get("messages") == expected_messages,
            "raw request input mismatch",
        )
        require(raw.get("request") == manifest["settings"], "raw settings mismatch")
        require(
            unit.get("user_prompt_sha256") == sha256(expected_messages[-1]["content"].encode()),
            "user prompt identity mismatch",
        )
        require(
            parse_claims(raw["raw_provider_output"], responses[rid]) == unit["processed_claims"],
            "processed claims differ from preserved provider output",
        )
    return manifest


def freeze_predictions(s1_dir=ROOT, llm_dir=None, output_dir=None, spacy_model="en_core_web_sm"):
    from src.components.extractors import SpacySentenceExtractor

    s1_dir = Path(s1_dir)
    load_gold(s1_dir)
    metadata, _, responses = benchmark(s1_dir)
    llm_dir = Path(llm_dir or s1_dir / "llm_run")
    provider = load_provider_run(llm_dir, s1_dir)
    out = Path(output_dir or s1_dir / "predictions")
    out.mkdir(parents=True, exist_ok=False)
    extractor = SpacySentenceExtractor(model=spacy_model)
    run_id = str(uuid.uuid4())
    rows, counts, raw_spacy = [], [], []
    versions = {name: importlib.metadata.version(name) for name in ["spacy", spacy_model]}
    versions["model_bytes_sha256"] = sha256(extractor.nlp.to_bytes())
    versions["effective_settings"] = {
        "model": spacy_model,
        "min_chars": 3,
        "use_sentencizer": False,
    }
    for rid, text in responses.items():
        actual = extractor.extract(text)
        raw = [c.to_dict() for c in actual]
        raw_spacy.append({"response_id": rid, "outputs": raw})
        llm = next(u for u in provider["responses"] if u["response_id"] == rid)
        for ext in ("spacy_sentence", "llm"):
            outputs = (
                [
                    {
                        "claim_text": c.text,
                        "source_span": c.source_span.to_dict() if c.source_span else None,
                        "source_spans": [c.source_span.to_dict()] if c.source_span else [],
                        "span_exactness": "exact" if c.source_span else "missing",
                        "decomposed": False,
                        "hedged": False,
                        "operations": [],
                        "raw_claim_index": i,
                        "processed_claim_index": i,
                    }
                    for i, c in enumerate(actual)
                ]
                if ext == "spacy_sentence"
                else llm["processed_claims"]
            )
            counts.append({"response_id": rid, "extractor": ext, "count": len(outputs)})
            for i, claim in enumerate(outputs):
                identity = f"{run_id}:{ext}:{rid}:{i}"
                rows.append(
                    dict(
                        **claim,
                        prediction_id="pred-" + sha256(identity.encode())[:24],
                        response_id=rid,
                        extractor=ext,
                        extractor_version="s1-v2",
                        model_identity=versions
                        if ext == "spacy_sentence"
                        else {
                            "provider": provider["provider"],
                            "model": provider["requested_model"],
                            "resolved_model": llm["resolved_model"],
                            "revision": provider.get("requested_revision"),
                        },
                        prompt_identity=None
                        if ext == "spacy_sentence"
                        else provider["system_prompt_sha256"],
                        raw_output_reference=f"spacy.raw.json#{rid}/{i}"
                        if ext == "spacy_sentence"
                        else f"provider/{llm['raw_file']}#{i}",
                        postprocessing_version="spacy-sentences-minchars3"
                        if ext == "spacy_sentence"
                        else PROCESSING_VERSION,
                        run_id=run_id,
                    )
                )
    write_new(
        out / "spacy.raw.json", {"model": spacy_model, "versions": versions, "responses": raw_spacy}
    )
    # Copy the exact provider artifacts into the frozen bundle, making it portable.
    (out / "provider").mkdir()
    provider_files = ["manifest.json"] + [u["raw_file"] for u in provider["responses"]]
    for name in provider_files:
        with (out / "provider" / name).open("xb") as stream:
            stream.write((llm_dir / name).read_bytes())
    with (out / "s1_predictions.jsonl").open("xb") as stream:
        stream.write(export_annotations(rows))
    files = ["spacy.raw.json", "s1_predictions.jsonl"] + [
        "provider/" + name for name in provider_files
    ]
    manifest = {
        "schema_version": "s1-prediction-freeze-v2",
        "run_id": run_id,
        "extractors": ["spacy_sentence", "llm"],
        "response_counts": counts,
        "responses_sha256": metadata["responses_hash_sha256"],
        "gold_manifest_sha256": sha256((s1_dir / "gold_manifest.json").read_bytes()),
        "files": {name: sha256((out / name).read_bytes()) for name in files},
        "created_at": utcnow_iso(),
        "software_revision": git_sha(),
        "software_dirty": git_is_dirty(),
        "model_identity": versions,
        "llm_processing": PROCESSING_VERSION,
        "software_identity": software_identity(),
        "note": "No deduplication or heuristic span inference in S1; hedged metadata for spaCy is unpredicted/false and not scored.",
    }
    write_new(out / "manifest.json", manifest)
    load_predictions(s1_dir, out)
    return manifest


def load_predictions(s1_dir=ROOT, directory=None):
    s1_dir = Path(s1_dir)
    directory = Path(directory or s1_dir / "predictions")
    frozen, _ = load_gold(s1_dir)
    _, _, responses = benchmark(s1_dir)
    manifest = read_json(directory / "manifest.json")
    require(
        manifest.get("schema_version") == "s1-prediction-freeze-v2", "prediction freeze required"
    )
    require(manifest.get("extractors") == ["spacy_sentence", "llm"], "both extractors required")
    require(
        manifest.get("gold_manifest_sha256")
        == sha256((s1_dir / "gold_manifest.json").read_bytes()),
        "stale prediction gold identity",
    )
    require(
        manifest.get("responses_sha256") == frozen["responses_sha256"], "stale prediction benchmark"
    )
    files = manifest.get("files", {})
    require(
        {"s1_predictions.jsonl", "spacy.raw.json", "provider/manifest.json"} <= files.keys(),
        "prediction raw artifacts missing",
    )
    for name, digest in files.items():
        path = directory / name
        require(path.resolve().is_relative_to(directory.resolve()), "artifact path escapes run")
        require(sha256(path.read_bytes()) == digest, f"prediction artifact changed: {name}")
    provider = load_provider_run(directory / "provider", s1_dir)
    raw_spacy = read_json(directory / "spacy.raw.json")
    rows = read_jsonl((directory / "s1_predictions.jsonl").read_bytes(), "predictions")
    ids = [r.get("prediction_id") for r in rows]
    require(
        len(set(ids)) == len(ids) and all(isinstance(x, str) and x for x in ids),
        "prediction IDs invalid",
    )
    required = {
        "prediction_id",
        "response_id",
        "extractor",
        "extractor_version",
        "model_identity",
        "prompt_identity",
        "claim_text",
        "source_span",
        "source_spans",
        "span_exactness",
        "decomposed",
        "hedged",
        "raw_claim_index",
        "processed_claim_index",
        "raw_output_reference",
        "postprocessing_version",
        "run_id",
    }
    for row in rows:
        require(required <= row.keys(), "prediction fields missing")
        require(
            row["response_id"] in responses and row["extractor"] in manifest["extractors"],
            "unknown prediction identity",
        )
        require(row["run_id"] == manifest["run_id"], "prediction run identity mismatch")
        require(
            isinstance(row["claim_text"], str) and bool(row["claim_text"].strip()),
            "empty prediction",
        )
        spans = row["source_spans"]
        require(
            isinstance(spans, list) and row["source_span"] == (spans[0] if spans else None),
            "prediction spans inconsistent",
        )
        previous = -1
        for span in spans:
            a, b = span["start"], span["end"]
            require(
                type(a) is int
                and type(b) is int
                and 0 <= a < b <= len(responses[row["response_id"]])
                and a >= previous,
                "prediction span bounds",
            )
            previous = b
        require(
            row["span_exactness"] in {"exact", "approximate", "missing"}, "span exactness invalid"
        )
        require(
            (row["span_exactness"] == "missing") == (not spans), "missing span flag inconsistent"
        )
        rid, index, ext = row["response_id"], row["raw_claim_index"], row["extractor"]
        require(
            type(index) is int and index >= 0 and row["processed_claim_index"] == index,
            "raw/processed indices invalid",
        )
        expected_id = "pred-" + sha256(f"{manifest['run_id']}:{ext}:{rid}:{index}".encode())[:24]
        require(row["prediction_id"] == expected_id, "unstable prediction ID")
        if ext == "llm":
            source_unit = next(u for u in provider["responses"] if u["response_id"] == rid)
            require(index < len(source_unit["processed_claims"]), "raw claim index out of range")
            original = source_unit["processed_claims"][index]
            require(
                all(row.get(k) == v for k, v in original.items()),
                "prediction differs from raw provider processing",
            )
            require(
                row["raw_output_reference"] == f"provider/{source_unit['raw_file']}#{index}"
                and row["postprocessing_version"] == PROCESSING_VERSION
                and row["prompt_identity"] == provider["system_prompt_sha256"],
                "LLM prediction provenance mismatch",
            )
        else:
            source_unit = next(u for u in raw_spacy["responses"] if u["response_id"] == rid)
            require(index < len(source_unit["outputs"]), "spaCy raw index out of range")
            original = source_unit["outputs"][index]
            require(
                row["claim_text"] == original["text"]
                and row["source_span"] == original["source_span"],
                "prediction differs from raw spaCy output",
            )
            require(
                row["raw_output_reference"] == f"spacy.raw.json#{rid}/{index}"
                and row["prompt_identity"] is None,
                "spaCy prediction provenance mismatch",
            )
        if row["span_exactness"] == "exact":
            require(
                len(spans) == 1
                and responses[rid][spans[0]["start"] : spans[0]["end"]] == row["claim_text"],
                "claimed exact span differs from source",
            )
    counts = manifest.get("response_counts", [])
    keys = [(c.get("extractor"), c.get("response_id")) for c in counts]
    require(
        len(keys) == len(set(keys))
        and set(keys) == {(e, r) for e in manifest["extractors"] for r in responses},
        "prediction completion missing",
    )
    for count in counts:
        require(
            count["count"]
            == sum(
                r["extractor"] == count["extractor"] and r["response_id"] == count["response_id"]
                for r in rows
            ),
            "prediction count mismatch",
        )
        subset = [
            r
            for r in rows
            if r["extractor"] == count["extractor"] and r["response_id"] == count["response_id"]
        ]
        require(
            sorted(r["raw_claim_index"] for r in subset) == list(range(count["count"])),
            "prediction indices must preserve every raw output, including duplicates",
        )
        source_units = (
            provider["responses"] if count["extractor"] == "llm" else raw_spacy["responses"]
        )
        source_unit = next(u for u in source_units if u["response_id"] == count["response_id"])
        expected_count = len(
            source_unit["processed_claims"]
            if count["extractor"] == "llm"
            else source_unit["outputs"]
        )
        require(count["count"] == expected_count, "raw predictions were dropped")
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-dir", type=Path, default=ROOT)
    parser.add_argument("--llm-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()
    try:
        freeze_predictions(**vars(args))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Blocked: {exc}\n")


if __name__ == "__main__":
    main()
