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

from dataclasses import replace
from typing import Iterable, Optional, Sequence

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
    k: Optional[int] = None,
) -> Optional[float]:
    """|gold ∩ retrieved@k| / |gold|, or None when there is no gold evidence."""
    gold = frozenset(gold_ids)
    if not gold:
        return None
    top = evidence[: k if k is not None else len(evidence)]
    return len(gold & {e.id for e in top}) / len(gold)


def gold_ranks(evidence: Sequence[Evidence], gold_ids: Iterable[str]) -> dict[str, Optional[int]]:
    """gold evidence id -> the rank it was retrieved at, or None if it was missed.

    This is the panel's most useful single output: "the gold sentence exists and the
    retriever put it at rank 34" is a completely different diagnosis from "the gold
    sentence was never returned".
    """
    by_id = {e.id: e.rank for e in evidence}
    return {gid: by_id.get(gid) for gid in gold_ids}


def reciprocal_rank(evidence: Sequence[Evidence], gold_ids: Iterable[str]) -> Optional[float]:
    """1/rank of the first gold hit; 0.0 if gold exists but none was retrieved."""
    gold = frozenset(gold_ids)
    if not gold:
        return None
    for e in evidence:
        if e.id in gold:
            return 1.0 / e.rank
    return 0.0


__all__ = ["mark_gold", "recall_at_k", "gold_ranks", "reciprocal_rank"]
