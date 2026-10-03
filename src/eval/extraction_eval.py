import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.s1_annotations import load_responses, read_jsonl, validate_annotations


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not common:
        return 0.0
    prec = common / len(pred_tokens)
    rec = common / len(gold_tokens)
    return (2 * prec * rec) / (prec + rec)


def compute_span_iou(span_sys, span_gold):
    if not span_sys or not span_gold:
        return 0.0
    sys_start, sys_end = span_sys.start, span_sys.end
    g_start, g_end = span_gold["start"], span_gold["end"]

    inter_start = max(sys_start, g_start)
    inter_end = min(sys_end, g_end)
    inter = max(0, inter_end - inter_start)

    union = (sys_end - sys_start) + (g_end - g_start) - inter
    return inter / union if union > 0 else 0.0


def load_data(s1_dir: Path):
    responses_file = s1_dir / "responses.jsonl"
    gold_file = s1_dir / "gold_annotations.jsonl"

    metadata = json.loads((s1_dir / "metadata.json").read_text(encoding="utf-8"))
    responses_data = responses_file.read_bytes()
    response_map = load_responses(responses_data, metadata["responses_hash_sha256"])
    if not gold_file.is_file():
        raise ValueError(
            "Human-adjudicated gold is missing. Do not run extractors before annotation."
        )
    gold_hash = metadata.get("gold_annotations_hash_sha256")
    if not gold_hash:
        raise ValueError(
            "Gold is not frozen: record gold_annotations_hash_sha256 after human adjudication."
        )
    gold_data = gold_file.read_bytes()
    if hashlib.sha256(gold_data).hexdigest() != gold_hash:
        raise ValueError("Gold hash differs from the frozen reference. No evaluation was run.")
    gold_records = read_jsonl(gold_data, "gold")
    result = validate_annotations(gold_records, response_map)
    if result.warnings:
        raise ValueError("Gold contains unresolved source/text warnings; human review required.")
    responses = read_jsonl(responses_data, "responses")
    gold_by_resp = defaultdict(list)
    for claim in gold_records:
        gold_by_resp[claim["response_id"]].append(claim)
    return responses, gold_by_resp


def generate_adjudication_csv(responses, gold_by_resp, s1_dir: Path):
    """
    Generate model-to-gold matching suggestions after the human gold gate.
    Suggestions and unmatched cases remain undecided until reviewed by a human.
    """
    # Enforce the gate for direct Python callers as well as the CLI.
    responses, gold_by_resp = load_data(s1_dir)
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.components.extractors import LLMClaimExtractor, SpacySentenceExtractor

    cache_path = s1_dir / "llm_claims_cache.json"
    if not cache_path.exists():
        raise ValueError(
            "Real LLM cache is missing. Both extractor outputs are required for comparison."
        )
    metadata = json.loads((s1_dir / "metadata.json").read_text(encoding="utf-8"))
    expected_cache_hash = metadata.get("llm_cache_hash_sha256")
    if (
        not expected_cache_hash
        or hashlib.sha256(cache_path.read_bytes()).hexdigest() != expected_cache_hash
    ):
        raise ValueError(
            "LLM cache must have a frozen llm_cache_hash_sha256 and real provider provenance."
        )
    for field in ("llm_cache_model", "llm_cache_prompt_version", "llm_cache_frozen_date"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"LLM cache provenance is missing: {field}")
    spacy_ext = SpacySentenceExtractor()
    llm_ext = LLMClaimExtractor(cache_path=str(cache_path), on_miss="error")

    extractors = [("SpacySentenceExtractor", spacy_ext)]
    if llm_ext:
        extractors.append(("LLMClaimExtractor", llm_ext))

    csv_path = s1_dir / "adjudication_candidates.csv"
    with io.StringIO(newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "extractor",
                "response_id",
                "system_claim",
                "candidate_gold_claim",
                "token_f1",
                "human_match_decision",
                "error_taxonomy_code",
                "adjudicator_notes",
            ]
        )

        for ext_name, ext in extractors:
            for r in responses:
                r_id = r["response_id"]
                text = r["text"]
                golds = gold_by_resp[r_id]

                sys_claims = ext.extract(text)

                cost_matrix = np.zeros((len(sys_claims), len(golds)))
                for i, sc in enumerate(sys_claims):
                    for j, gc in enumerate(golds):
                        f1 = compute_token_f1(sc.text, gc["claim_text"])
                        if f1 >= 0.8:
                            cost_matrix[i, j] = f1

                row_ind, col_ind = linear_sum_assignment(-cost_matrix)

                matched_sys = set()
                matched_gold = set()

                # Output candidates
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] >= 0.8:
                        matched_sys.add(i)
                        matched_gold.add(j)
                        writer.writerow(
                            [
                                ext_name,
                                r_id,
                                sys_claims[i].text,
                                golds[j]["claim_text"],
                                cost_matrix[i, j],
                                "",
                                "",
                                "",
                            ]
                        )

                # Output unmatched system claims (Spurious, etc)
                for i, sc in enumerate(sys_claims):
                    if i not in matched_sys:
                        writer.writerow(
                            [
                                ext_name,
                                r_id,
                                sc.text,
                                "[NONE]",
                                0.0,
                                "",
                                "",
                                "Unmatched suggestion; human review required",
                            ]
                        )

                # Output unmatched gold claims (Missed)
                for j, gc in enumerate(golds):
                    if j not in matched_gold:
                        writer.writerow(
                            [
                                ext_name,
                                r_id,
                                "[NONE]",
                                gc["claim_text"],
                                0.0,
                                "",
                                "",
                                "Unmatched suggestion; human review required",
                            ]
                        )
        with open(csv_path, "x", newline="", encoding="utf-8") as output:
            output.write(f.getvalue())

    print(f"Generated {csv_path} for human adjudication.")
    print(
        "Please fill out the 'human_match_decision' (1 or 0) and 'error_taxonomy_code' (S, U, O, R, P, B, C, D, N) columns."
    )


def bootstrap_ci(data: list[float], n_resamples: int = 10000):
    if not data:
        return 0.0, 0.0, 0.0
    data = np.array(data)
    means = np.zeros(n_resamples)
    for i in range(n_resamples):
        indices = np.random.randint(0, len(data), len(data))
        means[i] = np.mean(data[indices])
    return np.mean(data), np.percentile(means, 2.5), np.percentile(means, 97.5)


def run_evaluation(s1_dir: Path):
    """
    This function would load the adjudicated CSV and compute the final bootstrap CIs and taxonomy metrics.
    Currently a stub until the CSV is filled out by a human.
    """
    load_data(s1_dir)
    raise NotImplementedError(
        "Final S1 metrics are not implemented. No metrics were produced. "
        "Human gold and complete human matching decisions are prerequisites."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S1 Extraction Evaluation")
    parser.add_argument(
        "--generate-adjudication",
        action="store_true",
        help="Generate the CSV for human adjudication",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run the evaluation using the adjudicated CSV"
    )
    args = parser.parse_args()

    s1_dir = Path("data/eval/s1")

    try:
        if args.generate_adjudication:
            resp, gold = load_data(s1_dir)
            generate_adjudication_csv(resp, gold, s1_dir)
        elif args.evaluate:
            run_evaluation(s1_dir)
        else:
            parser.print_help()
    except (OSError, ValueError, KeyError, NotImplementedError) as exc:
        parser.exit(1, f"Blocked: {exc}\n")
