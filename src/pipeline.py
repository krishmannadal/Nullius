"""The runner: wires the five components together and produces a Trace.

This is deliberately the least clever file in the repo. It calls five things in order,
times each one, checks the contract at every boundary, and records everything. It makes
no decisions — every decision lives in a component, which is the whole point of the
architecture.

THE ORACLE PATH
---------------
``mode="oracle"`` replaces retrieved evidence with the example's **gold** evidence and
runs everything downstream unchanged. The difference between the two runs is the
**retrieval-attributable error**, and it is the single most useful measurement this
harness produces:

    retrieved run says Insufficient, oracle run says Supported  -> retrieval failure
    both say Insufficient                                       -> genuine insufficiency
                                                                   (or a verifier failure)

Two things make it trustworthy. First, the oracle substitutes evidence *only* — the
same reranker, verifier and aggregator run on it, so nothing else differs. Second,
``Trace.mode`` records which run a trace came from, so the two can never be confused
after the fact.

CONTRACT CHECKING IS ON BY DEFAULT
-----------------------------------
Every stage boundary is checked (``check_claims``, ``check_evidence_list``,
``check_rerank_is_subset``, ``check_pair_verdict``). The cost is O(k) set arithmetic
against a ~70 ms NLI forward pass — unmeasurable. The benefit is that a component
violating its contract fails *at the boundary it violated*, naming itself, rather than
producing a subtly wrong trace that looks fine.
"""

from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.core.config import config_hash, git_is_dirty, git_sha, project_root
from src.core.interfaces import (
    check_claims,
    check_evidence_list,
    check_pair_verdict,
    check_rerank_is_subset,
)
from src.core.registry import Pipeline
from src.core.types import (
    ClaimVerdict,
    Evidence,
    EvidenceVerdict,
    Trace,
    new_run_id,
    utcnow_iso,
)
from src.data.corpus import Corpus
from src.data.examples import Example
from src.data.metrics import mark_gold


class _Timer:
    """Accumulates wall-clock milliseconds per stage across all claims in a response."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}

    def add(self, stage: str, ms: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + ms

    def as_dict(self) -> dict[str, float]:
        out = {k: round(v, 3) for k, v in sorted(self.totals.items())}
        out["total_ms"] = round(sum(self.totals.values()), 3)
        return out


def _timed(timer: _Timer, stage: str):
    class _Ctx:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            timer.add(stage, (time.perf_counter() - self.t0) * 1000.0)
            return False

    return _Ctx()


def gold_evidence_for(example: Example, corpus: Corpus) -> list[Evidence]:
    """Materialise an example's gold keys as Evidence, in annotation order.

    Ranks are 1..n in the order the keys appear. That ordering is an artifact of the
    annotation, not a retrieval ranking, and ``retriever_name="oracle"`` says so — a
    rank-weighted aggregator running on oracle evidence is weighting by annotation
    order, which is meaningless. Recorded rather than hidden.
    """
    out: list[Evidence] = []
    for rank, (doc_id, sent_id) in enumerate(example.gold_evidence, start=1):
        if not corpus.has(doc_id, sent_id):
            raise KeyError(
                f"gold evidence ({doc_id!r}, {sent_id}) for example {example.id!r} is not in "
                "this corpus. The oracle condition would be silently incomplete; refusing."
            )
        sent = corpus.get(doc_id, sent_id)
        out.append(
            Evidence.new(
                doc_id=sent.doc_id,
                sent_id=sent.sent_id,
                text=sent.text,
                score=1.0,
                retriever_name="oracle",
                rank=rank,
                is_gold=True,
                retriever_meta={"source": "gold_annotation", "rank_is_annotation_order": True},
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class RunContext:
    """Everything a run needs that is not a component."""

    run_id: str
    config_hash: str
    git_sha: str | None
    resolved_config: dict[str, Any]

    @classmethod
    def create(cls, cfg: dict[str, Any], pipe: Pipeline) -> RunContext:
        data_identity = {}
        for name in ("corpus", "examples"):
            value = cfg.get("paths", {}).get(name)
            if value:
                path = Path(value)
                path = path if path.is_absolute() else project_root() / path
                if path.is_file():
                    data_identity[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        resolved = {
            "config": deepcopy({k: v for k, v in cfg.items() if not str(k).startswith("_")}),
            "components": pipe.describe(),
            "git_dirty": git_is_dirty(),
            "config_path": cfg.get("_config_path"),
            "data_identity": data_identity,
        }
        return cls(
            run_id=new_run_id(),
            config_hash=config_hash(cfg),
            git_sha=git_sha(),
            resolved_config=resolved,
        )


def analyze(
    pipe: Pipeline,
    response: str,
    ctx: RunContext,
    *,
    example: Example | None = None,
    corpus: Corpus | None = None,
    mode: str = "retrieved",
    retrieve_k: int | None = None,
    check_contracts: bool = True,
) -> Trace:
    """Run one response end to end and return its Trace.

    ``retrieve_k`` is the candidate pool handed to the reranker; it defaults to ``k``.
    With a real reranker you want it larger than k (there is nothing to rerank
    otherwise), which is why it is a separate knob rather than being derived.
    """
    if mode not in ("retrieved", "oracle"):
        raise ValueError(f"mode must be 'retrieved' or 'oracle', got {mode!r}")
    if mode == "oracle" and (example is None or corpus is None):
        raise ValueError("oracle mode needs both an example (for gold keys) and a corpus")

    timer = _Timer()
    k = pipe.k
    pool = int(retrieve_k or k)
    gold_ids = example.gold_evidence_ids if example else frozenset()

    with _timed(timer, "extract_ms"):
        claims = pipe.extractor.extract(response)
    if check_contracts:
        check_claims(claims, response)

    evidence_by_claim: dict[str, tuple[Evidence, ...]] = {}
    verdicts: list[ClaimVerdict] = []

    for claim in claims:
        # ---- retrieve (or substitute gold) ---------------------------------
        with _timed(timer, "retrieve_ms"):
            if mode == "oracle":
                # Gold is annotated per RESPONSE, not per claim. For FEVER, one claim
                # per response, so this is exact. For a multi-claim response every
                # claim gets the same gold, which is an over-approximation -- see
                # docs/pipeline.md, limitations.
                candidates = gold_evidence_for(example, corpus)  # type: ignore[arg-type]
            else:
                candidates = pipe.retriever.retrieve(claim, pool)
        if check_contracts:
            actual_k = len(candidates) if mode == "oracle" else pool
            check_evidence_list(candidates, k=actual_k, stage="retrieve")

        # ---- rerank ---------------------------------------------------------
        with _timed(timer, "rerank_ms"):
            reranked = pipe.reranker.rerank(claim, candidates, k)
        if check_contracts:
            check_rerank_is_subset(candidates, reranked)
            check_evidence_list(reranked, k=k, stage="rerank")

        # Reference flags belong to the trace, never verifier/reranker inputs.
        evidence_by_claim[claim.id] = tuple(mark_gold(reranked, gold_ids) if example else reranked)

        # ---- verify (one call per pair, never batched -- ADR-002) -----------
        pair_verdicts: list[EvidenceVerdict] = []
        with _timed(timer, "verify_ms"):
            for ev in reranked:
                clean_ev = replace(ev, is_gold=False, retriever_meta={
                    k: v for k, v in ev.retriever_meta.items() if k not in {"source", "rank_is_annotation_order"}
                })
                pv = pipe.verifier.score(claim, clean_ev)
                if check_contracts:
                    check_pair_verdict(pv, claim, ev)
                pair_verdicts.append(pv)

        # ---- aggregate -------------------------------------------------------
        with _timed(timer, "aggregate_ms"):
            verdict = pipe.aggregator.aggregate(deepcopy(claim), tuple(pair_verdicts))
            if verdict.per_evidence != tuple(pair_verdicts):
                raise ValueError("Aggregator changed the original pairwise verdicts")
            verdicts.append(verdict)

    return Trace(
        run_id=ctx.run_id,
        config_hash=ctx.config_hash,
        git_sha=ctx.git_sha,
        timestamp=utcnow_iso(),
        response_text=response,
        claims=tuple(claims),
        evidence_by_claim=evidence_by_claim,
        verdicts=tuple(verdicts),
        timings=timer.as_dict(),
        mode=mode,
        resolved_config={
            **ctx.resolved_config,
            "k": k,
            "retrieve_k": pool,
            "example_id": example.id if example else None,
            "gold_label": example.gold_label.value if (example and example.gold_label) else None,
            "has_gold_evidence": example.has_gold_evidence if example else False,
        },
    )


def analyze_example(
    pipe: Pipeline,
    example: Example,
    corpus: Corpus,
    ctx: RunContext,
    *,
    retrieve_k: int | None = None,
    check_contracts: bool = True,
) -> tuple[Trace, Trace | None]:
    """Run one example in BOTH conditions. Returns (retrieved, oracle-or-None).

    The oracle run is skipped when an example has no gold evidence — FEVER's NEI class
    by construction. Returning ``None`` rather than an empty-evidence trace keeps the
    two cases distinguishable: "no oracle exists for this example" is not the same as
    "the oracle found nothing".
    """
    retrieved = analyze(
        pipe, example.text, ctx, example=example, corpus=corpus,
        mode="retrieved", retrieve_k=retrieve_k, check_contracts=check_contracts,
    )
    if not example.has_gold_evidence:
        return retrieved, None
    oracle = analyze(
        pipe, example.text, ctx, example=example, corpus=corpus,
        mode="oracle", retrieve_k=retrieve_k, check_contracts=check_contracts,
    )
    return retrieved, oracle


__all__ = ["RunContext", "analyze", "analyze_example", "gold_evidence_for"]
