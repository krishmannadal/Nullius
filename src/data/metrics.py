"""Retrieval metrics against gold evidence, for one example at a time.

Deliberately small. These are the numbers the Retrieval panel shows for the example
on screen -- not an evaluation harness. Nothing here averages over a dataset, because
averaging is where the two honest ``None``s below get quietly turned into zeros.

The one judgement call: **Recall@k is undefined when an example has no gold
evidence.** FEVER's NEI class has empty evidence by construction, so a third of a
FEVER slice has no denominator. Returning 1.0 (vacuously "found everything") flatters
the retriever; returning 0.0 punishes it for a question that was not asked. Both
corrupt any mean taken over them. So it returns ``None``, and any future aggregation
has to decide what to do with that in the open.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from src.core.types import Evidence


def mark_gold(evidence: Sequence[Evidence], gold_ids: Iterable[str]) -> list[Evidence]:
    """Stamp ``is_gold`` on retrieved evidence.

    After this call ``is_gold`` is True/False -- a real annotation. Before it, the
    field is None, meaning "unknown". The distinction matters: an example with no
    gold annotation at all must not have its evidence marked False, or the UI will
    render "retrieval missed the gold" for an example that never had any.
    """
    gold = frozenset(gold_ids)
    return [replace(e, is_gold=(e.id in gold)) for e in evidence]


def recall_at_k(
    evidence: Sequence[Evidence],
    gold_ids: Iterable[str],
    k: int | None = None,
) -> float | None:
    """|gold ∩ retrieved@k| / |gold|, or None when there is no gold evidence."""
    gold = frozenset(gold_ids)
    if not gold:
        return None
    top = evidence[: k if k is not None else len(evidence)]
    return len(gold & {e.id for e in top}) / len(gold)


def gold_ranks(evidence: Sequence[Evidence], gold_ids: Iterable[str]) -> dict[str, int | None]:
    """gold evidence id -> the rank it was retrieved at, or None if it was missed.

    This is the panel's most useful single output: "the gold sentence exists and the
    retriever put it at rank 34" is a completely different diagnosis from "the gold
    sentence was never returned".
    """
    by_id = {e.id: e.rank for e in evidence}
    return {gid: by_id.get(gid) for gid in gold_ids}


def reciprocal_rank(evidence: Sequence[Evidence], gold_ids: Iterable[str]) -> float | None:
    """1/rank of the first gold hit; 0.0 if gold exists but none was retrieved."""
    gold = frozenset(gold_ids)
    if not gold:
        return None
    for e in evidence:
        if e.id in gold:
            return 1.0 / e.rank
    return 0.0


# --------------------------------------------------------------------------- #
# run-level summary
# --------------------------------------------------------------------------- #
# Everything above is per-example. What follows aggregates over a run, which means it
# has to decide what to do with the Nones above -- and it does so explicitly, in the
# open, exactly as the module docstring demands.

def summarise_run(
    retrieved: Sequence[Any],
    oracle_by_example: Mapping[str, Any],
    examples_by_id: Mapping[str, Any],
) -> dict:
    """Counts and the retrieved-vs-oracle gap over a set of traces.

    THESE ARE NOT EVALUATION NUMBERS. Every corpus in this repo contains the gold
    evidence for its own examples by construction, so accuracy over it describes the
    corpus. The reason to compute them at all is that the *gap* between the retrieved
    and oracle conditions is a within-run comparison -- both conditions see the same
    corpus, the same verifier, the same aggregator -- and that comparison is the point
    of the harness.

    Recall is averaged only over examples that HAVE gold evidence. Examples without it
    contribute None per-example and are excluded here rather than counted as 0 or 1;
    `n_recall_defined` reports the denominator so the exclusion is visible.
    """
    label_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    correct = 0
    scored = 0
    abstained = 0
    recalls: list[float] = []
    per_stage_ms: dict[str, float] = {}

    # retrieved-vs-oracle, over examples where BOTH conditions ran
    gap_rows: list[dict] = []

    for tr in retrieved:
        ex_id = tr.resolved_config.get("example_id")
        ex = examples_by_id.get(ex_id) if ex_id else None
        for stage, ms in tr.timings.items():
            if stage != "total_ms":
                per_stage_ms[stage] = per_stage_ms.get(stage, 0.0) + float(ms)
        per_stage_ms["total_ms"] = per_stage_ms.get("total_ms", 0.0) + float(tr.timings.get("total_ms", 0.0))

        for verdict in tr.verdicts:
            label_counts[verdict.label.value] = label_counts.get(verdict.label.value, 0) + 1
            if verdict.abstained:
                abstained += 1
            if ex is not None and ex.gold_label is not None:
                gold_counts[ex.gold_label.value] = gold_counts.get(ex.gold_label.value, 0) + 1
                scored += 1
                if verdict.label is ex.gold_label:
                    correct += 1

        if ex is not None and ex.has_gold_evidence:
            for evs in tr.evidence_by_claim.values():
                r = recall_at_k(evs, ex.gold_evidence_ids)
                if r is not None:
                    recalls.append(r)

        oracle = oracle_by_example.get(ex_id) if ex_id else None
        if oracle is not None and ex is not None and ex.gold_label is not None:
            r_verdicts = {v.claim_id: v for v in tr.verdicts}
            o_verdicts = {v.claim_id: v for v in oracle.verdicts}
            for claim_id, r_v in r_verdicts.items():
                if claim_id in o_verdicts:
                    o_v = o_verdicts[claim_id]
                    gap_rows.append({
                        "example_id": ex_id,
                        "claim_id": claim_id,
                        "gold": ex.gold_label.value,
                        "retrieved": r_v.label.value,
                        "oracle": o_v.label.value,
                        "retrieved_correct": r_v.label is ex.gold_label,
                        "oracle_correct": o_v.label is ex.gold_label,
                    })

    n_gap = len(gap_rows)
    retrieved_correct = sum(1 for r in gap_rows if r["retrieved_correct"])
    oracle_correct = sum(1 for r in gap_rows if r["oracle_correct"])
    fixed_by_oracle = [r for r in gap_rows if r["oracle_correct"] and not r["retrieved_correct"]]
    broken_by_oracle = [r for r in gap_rows if r["retrieved_correct"] and not r["oracle_correct"]]

    return {
        "n_traces": len(retrieved),
        "n_claims": sum(len(t.claims) for t in retrieved),
        "predicted_labels": label_counts,
        "gold_labels": gold_counts,
        "n_abstained": abstained,
        "n_scored_against_gold": scored,
        "agreement_with_gold": (correct / scored) if scored else None,
        "recall_at_k_mean": (sum(recalls) / len(recalls)) if recalls else None,
        "n_recall_defined": len(recalls),
        "timings_ms_total": {k: round(v, 1) for k, v in sorted(per_stage_ms.items())},
        "retrieval_attributable_error": {
            "n_claims_with_both_conditions": n_gap,
            "retrieved_agreement": (retrieved_correct / n_gap) if n_gap else None,
            "oracle_agreement": (oracle_correct / n_gap) if n_gap else None,
            "gap": ((oracle_correct - retrieved_correct) / n_gap) if n_gap else None,
            "n_fixed_by_oracle": len(fixed_by_oracle),
            "n_broken_by_oracle": len(broken_by_oracle),
            "fixed_by_oracle": fixed_by_oracle[:20],
            "broken_by_oracle": broken_by_oracle[:20],
        },
        "_warning": (
            "Agreement figures are NOT evaluation results: the corpus contains the gold "
            "evidence for every example by construction and nothing is tuned. The gap "
            "between conditions is a within-run comparison and is the useful column."
        ),
    }


__all__ = [
    "gold_ranks",
    "mark_gold",
    "recall_at_k",
    "reciprocal_rank",
    "summarise_run",
]
