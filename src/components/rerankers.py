"""Rerankers: reorder (and truncate) a candidate list. Never invent evidence.

Two implementations:

* ``noop``          -- returns the first k unchanged, but renumbers ranks.
* ``cross_encoder`` -- ms-marco-MiniLM-L-6-v2 scores each (claim, evidence) pair.

Why ``noop`` exists as a real registered component rather than as an ``if`` statement:
keeping it on the same code path means the trace, the timing breakdown, and the panel
layout are identical whether reranking is on or off. "Reranker off" is then a
*measurement* -- a component with a name and a latency in the trace -- rather than a
different program (ADR-008).

THE CONTRACT THAT MATTERS
-------------------------
A reranker may reorder and drop. It may **not** create, substitute, or edit. Since
``Evidence.id`` is corpus identity and nothing else, ``check_rerank_is_subset``
catches not only invented evidence but the subtler case of a reranker that rebuilds
Evidence objects from a *different* corpus copy -- the ids would not match.

Both implementations preserve ``doc_id``/``sent_id``/``text`` exactly by going through
``Evidence.reranked()``, which only ever changes rank, score and ``retriever_name``.
"""

from __future__ import annotations

import time
from typing import Optional

from src.core.interfaces import Reranker
from src.core.registry import register
from src.core.types import Claim, Evidence


@register("reranker", "noop")
class NoOpReranker(Reranker):
    """Identity, except that ranks are renumbered 1..k after truncation.

    The renumbering is not cosmetic. ``check_evidence_list`` requires ranks to be
    1-based and contiguous in list order, and downstream both RRF-style weighting and
    ``WeightedByRetrievalAggregator`` read rank as position. Truncating to k without
    renumbering would be a no-op that quietly violates the contract.
    """

    def rerank(self, claim: Claim, ev: list[Evidence], k: int) -> list[Evidence]:
        return [
            e.reranked(rank=i, score=e.score, retriever_name=e.retriever_name)
            for i, e in enumerate(ev[:k], start=1)
        ]


@register("reranker", "cross_encoder")
class CrossEncoderReranker(Reranker):
    """Cross-encoder relevance reranking.

    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` is a **relevance** model trained on MS
    MARCO passage ranking. It answers "is this passage relevant to this query?", not
    "does this passage support or refute this claim". That distinction is easy to lose:
    a sentence that flatly contradicts the claim is highly *relevant* and should rank
    high -- which is what you want feeding a verifier, but it means a high reranker
    score is emphatically not evidence of support. Anything reading these scores as a
    truth signal is misreading them.

    The score is an unbounded logit, not a probability, so ``score`` here is not
    comparable to a BM25 score or a cosine. Same rule as everywhere else in this repo:
    scores are per-component and the UI shows them in their own column.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        import torch
        from sentence_transformers import CrossEncoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        # Explicit, for the same reason as in verifiers.py: several of these tokenizers
        # ship model_max_length as a ~1e19 sentinel, which makes truncation=True a
        # silent no-op.
        self.max_length = int(max_length)
        self.model = CrossEncoder(model_name, device=device, max_length=self.max_length)

    def rerank(self, claim: Claim, ev: list[Evidence], k: int) -> list[Evidence]:
        if not ev:
            return []
        t0 = time.perf_counter()
        scores = self.model.predict(
            [(claim.text, e.text) for e in ev],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Sort by score desc, breaking ties by ORIGINAL rank so the ordering is
        # deterministic and, on a tie, defers to the retriever rather than to
        # whatever order numpy happened to produce.
        order = sorted(range(len(ev)), key=lambda i: (-float(scores[i]), ev[i].rank))

        out: list[Evidence] = []
        for new_rank, i in enumerate(order[:k], start=1):
            original = ev[i]
            moved = original.reranked(
                rank=new_rank,
                score=float(scores[i]),
                retriever_name=f"{original.retriever_name}+{self.name}",
            )
            # Keep the pre-rerank position visible. The whole point of a reranker panel
            # is seeing what MOVED, which is impossible once the old rank is discarded.
            out.append(
                Evidence(
                    id=moved.id,
                    doc_id=moved.doc_id,
                    sent_id=moved.sent_id,
                    text=moved.text,
                    score=moved.score,
                    retriever_name=moved.retriever_name,
                    rank=moved.rank,
                    is_gold=moved.is_gold,
                    retriever_meta={
                        **original.retriever_meta,
                        "rerank_score": float(scores[i]),
                        "rank_before_rerank": original.rank,
                        "rank_delta": original.rank - new_rank,   # +ve = promoted
                        "rerank_ms_total": round(elapsed_ms, 2),
                    },
                )
            )
        return out


__all__ = ["NoOpReranker", "CrossEncoderReranker"]
