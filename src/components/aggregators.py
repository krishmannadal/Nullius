"""Aggregators: k per-evidence verdicts -> one claim-level decision.

This is the slot most likely to become the research question, so it is the most
instrumented. Five implementations, four real and one null:

    max_entailment          argmax over the per-class MAXIMUM across evidence
    noisy_or                1 - prod(1 - p_i), independence assumed (and violated)
    weighted_by_retrieval   retrieval-rank-weighted mean of the class probabilities
    threshold_abstain       floors + an explicit Abstain outcome
    majority                NULL BASELINE: one vote per evidence, magnitudes discarded

Every one of them writes ``rule``, ``explanation`` and ``decisive_evidence_ids`` into
``aggregation_trace`` -- enforced by ``ClaimVerdict.__post_init__``, so a new
aggregator that forgets cannot be constructed.

THREE THINGS TO UNDERSTAND BEFORE READING THE CODE
---------------------------------------------------

**1. Zero evidence is a first-class case, not an edge case.** ``aggregate(claim, [])``
happens whenever retrieval returns nothing, and it is exactly the boundary this
project cares about: "the corpus does not settle this" versus "the retriever missed
it". Every aggregator here returns ``Insufficient`` with an explanation that says
*which* of the two it cannot distinguish. None of them pretend to know.

**2. Cosine similarity is structurally incapable of detecting contradiction.**
"Marie Curie was born in Warsaw" and "Marie Curie was born in Paris" are near-identical
strings with high cosine similarity. So when these aggregators run on
similarity-only verdicts they map similarity to a *support* signal and have **no**
contradiction signal at all -- meaning a similarity-only pipeline can never output
``Contradicted``. That is not a bug to be fixed here; it is a property of the signal,
and ``aggregation_trace["signal"]`` says so on every verdict so the UI can show it.

**3. Mapping NLI-``neutral`` onto ``Insufficient`` is a modelling decision.**
``Label.parse`` deliberately refuses to do it (ADR-007) precisely so that it has to
happen *here*, where it lands in ``aggregation_trace["rule"]`` and is visible. Three
of the five aggregators below make that mapping. Each one says so in its rule string.
"The model was unsure" and "the corpus does not settle it" are different propositions
and this conflation is a named threat, not a convenience.

Nothing here is tuned. Every floor is a placeholder and says so.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.interfaces import Aggregator
from src.core.registry import register
from src.core.types import Claim, ClaimVerdict, EvidenceVerdict, Label

#: Signal kinds recorded in aggregation_trace["signal"].
SIGNAL_NLI = "nli_3way"
SIGNAL_SIMILARITY = "similarity_only(no_contradiction_signal)"


def signal_kind(verdicts: Sequence[EvidenceVerdict]) -> str:
    return SIGNAL_NLI if verdicts and verdicts[0].has_nli else SIGNAL_SIMILARITY


def support_contra_neutral(
    v: EvidenceVerdict,
) -> tuple[float, float | None, float | None]:
    """One verdict -> (support, contradiction, neutral) signals.

    For NLI verdicts these are the three probabilities unchanged. For a
    similarity-only verdict, cosine is rescaled from [-1, 1] to [0, 1] as a *support*
    signal and the other two are ``None`` -- see point 2 in the module docstring. The
    rescaling is a monotone remap for comparability, **not** a probability, and no
    aggregator should treat it as calibrated.
    """
    if v.has_nli:
        return float(v.p_entail), float(v.p_contra), float(v.p_neutral)  # type: ignore[arg-type]
    if v.similarity is not None:
        return (float(v.similarity) + 1.0) / 2.0, None, None
    raise ValueError(f"verdict for {v.evidence_id!r} carries neither NLI probs nor similarity")


def _empty_verdict(claim: Claim, aggregator_name: str, extra: dict | None = None) -> ClaimVerdict:
    """The zero-evidence case, identical across aggregators so it is comparable.

    Confidence is 0.0, not 1.0: we are maximally unsure, not certainly-insufficient.
    """
    trace = {
        "rule": "no evidence reached the aggregator",
        "explanation": (
            "Retrieval returned nothing for this claim, so there was nothing to "
            "aggregate. This CANNOT distinguish 'the corpus does not settle this "
            "claim' from 'the corpus settles it and the retriever missed it' -- "
            "compare against the oracle run to tell them apart."
        ),
        "decisive_evidence_ids": [],
        "signal": "none",
        "n_evidence": 0,
    }
    trace.update(extra or {})
    return ClaimVerdict(
        claim_id=claim.id,
        label=Label.INSUFFICIENT,
        confidence=0.0,
        abstained=False,
        per_evidence=(),
        aggregator_name=aggregator_name,
        aggregation_trace=trace,
    )


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


# --------------------------------------------------------------------------- #
# 1. max entailment
# --------------------------------------------------------------------------- #

@register("aggregator", "max_entailment")
class MaxEntailmentAggregator(Aggregator):
    """Pool each class by its MAXIMUM over evidence, then take the argmax.

    .. math::
        P_c = \\max_i p_c(e_i) \\quad\\text{for } c \\in \\{\\text{entail}, \\text{contra}, \\text{neutral}\\}

    The simplest defensible pooling rule, and the natural baseline: one sufficiently
    convincing piece of evidence decides the claim. It is also **maximally sensitive
    to a single false positive** -- one spurious high-entailment sentence outvotes any
    number of neutral ones, because nothing else is even consulted.

    Maps pooled-neutral to ``Insufficient``. See module docstring point 3.
    """

    def aggregate(self, claim: Claim, v: list[EvidenceVerdict]) -> ClaimVerdict:
        if not v:
            return _empty_verdict(claim, self.name)

        best_s = max(v, key=lambda x: support_contra_neutral(x)[0])
        s = support_contra_neutral(best_s)[0]
        has_nli = best_s.has_nli

        if has_nli:
            best_c = max(v, key=lambda x: support_contra_neutral(x)[1] or 0.0)
            best_n = max(v, key=lambda x: support_contra_neutral(x)[2] or 0.0)
            c = support_contra_neutral(best_c)[1] or 0.0
            n = support_contra_neutral(best_n)[2] or 0.0
            scores = {Label.SUPPORTED: s, Label.CONTRADICTED: c, Label.INSUFFICIENT: n}
            drivers = {Label.SUPPORTED: best_s, Label.CONTRADICTED: best_c, Label.INSUFFICIENT: best_n}
        else:
            c = n = 0.0
            scores = {Label.SUPPORTED: s}
            drivers = {Label.SUPPORTED: best_s}

        label = max(scores, key=lambda k: scores[k])
        driver = drivers[label]
        return ClaimVerdict(
            claim_id=claim.id,
            label=label,
            confidence=float(scores[label]),
            abstained=False,
            per_evidence=tuple(v),
            aggregator_name=self.name,
            aggregation_trace={
                "rule": (
                    "label = argmax_c max_i p_c(e_i); pooled NLI-neutral is READ AS "
                    "Insufficient, which conflates 'model unsure' with 'corpus does "
                    "not settle it'"
                ),
                "explanation": (
                    f"Across {len(v)} evidence pieces the strongest signal was "
                    f"{label.value.lower()} at {scores[label]:.3f}, from {driver.evidence_id} "
                    f"(retrieval rank {driver.evidence_rank}). "
                    f"Pooled maxima: support={_fmt(s)}, contradict={_fmt(c) if has_nli else 'n/a'}, "
                    f"neutral={_fmt(n) if has_nli else 'n/a'}. "
                    "A single high-scoring piece decides this rule outright."
                ),
                "decisive_evidence_ids": [driver.evidence_id],
                "signal": signal_kind(v),
                "n_evidence": len(v),
                "pooled": {
                    "support": s,
                    "contradict": c if has_nli else None,
                    "neutral": n if has_nli else None,
                },
            },
        )


# --------------------------------------------------------------------------- #
# 2. noisy-OR
# --------------------------------------------------------------------------- #

@register("aggregator", "noisy_or")
class NoisyOrAggregator(Aggregator):
    """Combine evidence as independent noisy channels.

    .. math::
        P_c = 1 - \\prod_i \\bigl(1 - p_c(e_i)\\bigr)

    Reads as "at least one piece of evidence supports the claim", accumulating weak
    signals rather than taking only the strongest.

    **The independence assumption is false here and knowingly so.** Retrieved evidence
    is highly correlated -- top-k frequently contains several near-duplicate sentences
    from the same page (visible in the retrieval demo on FEVER). Five copies of the
    same fact at p=0.3 give P = 1 - 0.7^5 = 0.83, as though five independent witnesses
    had spoken. Noisy-OR therefore saturates toward 1 as k grows, which makes it
    **k-dependent in a way max-entailment is not** -- a direct, cheap experiment.
    """

    def __init__(self, decision_floor: float = 0.5) -> None:
        super().__init__()
        # PLACEHOLDER, untuned. Below this on both classes -> Insufficient.
        self.decision_floor = float(decision_floor)

    def aggregate(self, claim: Claim, v: list[EvidenceVerdict]) -> ClaimVerdict:
        if not v:
            return _empty_verdict(claim, self.name, {"decision_floor": self.decision_floor})

        has_nli = v[0].has_nli
        p_s = 1.0
        p_c = 1.0
        for x in v:
            s, c, _ = support_contra_neutral(x)
            p_s *= (1.0 - s)
            if c is not None:
                p_c *= (1.0 - c)
        support = 1.0 - p_s
        contra = (1.0 - p_c) if has_nli else 0.0

        if max(support, contra) < self.decision_floor:
            label = Label.INSUFFICIENT
            confidence = 1.0 - max(support, contra)
            decisive: list[str] = []
        elif support >= contra:
            label = Label.SUPPORTED
            confidence = support
            decisive = [x.evidence_id for x in v if support_contra_neutral(x)[0] > 0.1]
        else:
            label = Label.CONTRADICTED
            confidence = contra
            decisive = [x.evidence_id for x in v if (support_contra_neutral(x)[1] or 0.0) > 0.1]

        return ClaimVerdict(
            claim_id=claim.id,
            label=label,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            abstained=False,
            per_evidence=tuple(v),
            aggregator_name=self.name,
            aggregation_trace={
                "rule": (
                    f"P_c = 1 - prod_i(1 - p_c(e_i)); label = argmax over "
                    f"{{support, contradict}} if either >= {self.decision_floor}, else "
                    "Insufficient. ASSUMES INDEPENDENT EVIDENCE, which retrieved "
                    "evidence is not."
                ),
                "explanation": (
                    f"Accumulating {len(v)} pieces as independent channels gives "
                    f"P(support)={support:.3f}, P(contradict)={_fmt(contra) if has_nli else 'n/a'}. "
                    f"{len(decisive)} piece(s) contributed non-trivially. "
                    "Because near-duplicate retrieved sentences are counted as separate "
                    "witnesses, this value rises with k even when no new information "
                    "arrives."
                ),
                "decisive_evidence_ids": decisive,
                "signal": signal_kind(v),
                "n_evidence": len(v),
                "decision_floor": self.decision_floor,
                "pooled": {"support": support, "contradict": contra if has_nli else None},
            },
        )


# --------------------------------------------------------------------------- #
# 3. weighted by retrieval
# --------------------------------------------------------------------------- #

@register("aggregator", "weighted_by_retrieval")
class WeightedByRetrievalAggregator(Aggregator):
    """Weighted mean of class probabilities, weighted by retrieval position.

    .. math::
        w_i = \\frac{1}{\\text{rank}_i^{\\alpha}}, \\qquad
        P_c = \\frac{\\sum_i w_i\\, p_c(e_i)}{\\sum_i w_i}

    **Weights come from RANK, not from retrieval score, and that is deliberate.**
    Step A established that retriever scores are not comparable across retrievers or
    even across queries: BM25 is unbounded and query-dependent, cosine is bounded,
    and RRF scores live on a third scale entirely. Weighting by score would make this
    aggregator's behaviour silently depend on which retriever produced the evidence.
    Rank is the one signal that means the same thing everywhere.

    ``evidence_score`` is still recorded in the trace so the choice can be revisited
    with data rather than by argument.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        # PLACEHOLDER. alpha=0 -> uniform mean; alpha=1 -> 1/rank; larger -> sharper.
        self.alpha = float(alpha)

    def aggregate(self, claim: Claim, v: list[EvidenceVerdict]) -> ClaimVerdict:
        if not v:
            return _empty_verdict(claim, self.name, {"alpha": self.alpha})

        has_nli = v[0].has_nli
        weights = [1.0 / (float(x.evidence_rank or (i + 1)) ** self.alpha) for i, x in enumerate(v)]
        total_w = sum(weights) or 1.0

        s = sum(w * support_contra_neutral(x)[0] for w, x in zip(weights, v)) / total_w
        if has_nli:
            c = sum(w * (support_contra_neutral(x)[1] or 0.0) for w, x in zip(weights, v)) / total_w
            n = sum(w * (support_contra_neutral(x)[2] or 0.0) for w, x in zip(weights, v)) / total_w
            scores = {Label.SUPPORTED: s, Label.CONTRADICTED: c, Label.INSUFFICIENT: n}
        else:
            c = n = 0.0
            scores = {Label.SUPPORTED: s}

        label = max(scores, key=lambda k: scores[k])
        # "Decisive" = the pieces carrying the top half of the weight mass.
        ordered = sorted(zip(weights, v), key=lambda p: -p[0])
        cum, decisive = 0.0, []
        for w, x in ordered:
            decisive.append(x.evidence_id)
            cum += w
            if cum >= 0.5 * total_w:
                break

        return ClaimVerdict(
            claim_id=claim.id,
            label=label,
            confidence=float(min(max(scores[label], 0.0), 1.0)),
            abstained=False,
            per_evidence=tuple(v),
            aggregator_name=self.name,
            aggregation_trace={
                "rule": (
                    f"w_i = 1/rank_i^{self.alpha}; P_c = sum_i w_i p_c(e_i) / sum_i w_i; "
                    "label = argmax_c P_c. Weights use RANK not score, because "
                    "retriever scores are not comparable across retrievers or queries. "
                    "Weighted NLI-neutral is READ AS Insufficient."
                ),
                "explanation": (
                    f"Rank-weighted over {len(v)} pieces (alpha={self.alpha}): "
                    f"support={s:.3f}, contradict={_fmt(c) if has_nli else 'n/a'}, "
                    f"neutral={_fmt(n) if has_nli else 'n/a'}. "
                    f"The top {len(decisive)} piece(s) by rank carry half the weight, so "
                    "they dominate the outcome; lower-ranked evidence can only nudge it."
                ),
                "decisive_evidence_ids": decisive,
                "signal": signal_kind(v),
                "n_evidence": len(v),
                "alpha": self.alpha,
                "weights": [round(w / total_w, 4) for w in weights],
                "ranks": [x.evidence_rank for x in v],
                "retrieval_scores": [x.evidence_score for x in v],
                "pooled": {
                    "support": s,
                    "contradict": c if has_nli else None,
                    "neutral": n if has_nli else None,
                },
            },
        )


# --------------------------------------------------------------------------- #
# 4. threshold + abstain
# --------------------------------------------------------------------------- #

@register("aggregator", "threshold_abstain")
class ThresholdWithAbstainAggregator(Aggregator):
    """The only aggregator here that can emit all four labels, including Abstain.

    Decision rule, in order::

        no evidence                      -> Insufficient (confidence 0)
        s >= support_floor and s >= c    -> Supported
        c >= contradict_floor and c > s  -> Contradicted
        max(s, c) <  abstain_below       -> ABSTAIN          (abstained=True)
        otherwise                        -> Insufficient

    where ``s = max_i p_entail(e_i)`` and ``c = max_i p_contra(e_i)``.

    **Abstain and Insufficient are different claims about the world**, which is why
    both exist. *Insufficient* asserts something: the evidence was read and does not
    settle the claim. *Abstain* asserts nothing: the system declines to answer because
    no signal cleared the confidence floor. Only the second belongs on a risk-coverage
    curve. Collapsing them -- which every threshold-free aggregator above effectively
    does -- is the thing this one exists to avoid.

    All three floors are PLACEHOLDERS. They have not been tuned, must not be tuned on
    dev or test, and no number produced with them means anything yet.
    """

    def __init__(
        self,
        support_floor: float = 0.5,
        contradict_floor: float = 0.5,
        abstain_below: float = 0.4,
    ) -> None:
        super().__init__()
        for name, value in (
            ("support_floor", support_floor),
            ("contradict_floor", contradict_floor),
            ("abstain_below", abstain_below),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self.support_floor = float(support_floor)
        self.contradict_floor = float(contradict_floor)
        self.abstain_below = float(abstain_below)

    def aggregate(self, claim: Claim, v: list[EvidenceVerdict]) -> ClaimVerdict:
        floors = {
            "support_floor": self.support_floor,
            "contradict_floor": self.contradict_floor,
            "abstain_below": self.abstain_below,
        }
        if not v:
            return _empty_verdict(claim, self.name, floors)

        has_nli = v[0].has_nli
        best_s = max(v, key=lambda x: support_contra_neutral(x)[0])
        s = support_contra_neutral(best_s)[0]
        if has_nli:
            best_c = max(v, key=lambda x: support_contra_neutral(x)[1] or 0.0)
            c = support_contra_neutral(best_c)[1] or 0.0
        else:
            best_c, c = best_s, 0.0

        abstained = False
        if s >= self.support_floor and s >= c:
            label, confidence, driver = Label.SUPPORTED, s, best_s
            why = f"support {s:.3f} cleared the {self.support_floor} floor"
        elif has_nli and c >= self.contradict_floor and c > s:
            label, confidence, driver = Label.CONTRADICTED, c, best_c
            why = f"contradiction {c:.3f} cleared the {self.contradict_floor} floor"
        elif max(s, c) < self.abstain_below:
            label, confidence, driver = Label.ABSTAIN, max(s, c), (best_s if s >= c else best_c)
            abstained = True
            why = (
                f"the strongest signal was only {max(s, c):.3f}, below the "
                f"{self.abstain_below} abstention floor, so the system declines to answer"
            )
        else:
            label, confidence, driver = Label.INSUFFICIENT, max(s, c), (best_s if s >= c else best_c)
            why = (
                f"the strongest signal ({max(s, c):.3f}) is above the abstention floor "
                f"but cleared no decision floor, so the evidence was read and does not "
                "settle the claim"
            )

        return ClaimVerdict(
            claim_id=claim.id,
            label=label,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            abstained=abstained,
            per_evidence=tuple(v),
            aggregator_name=self.name,
            aggregation_trace={
                "rule": (
                    f"s=max_i p_entail, c=max_i p_contra. "
                    f"s>={self.support_floor} and s>=c -> Supported; "
                    f"c>={self.contradict_floor} and c>s -> Contradicted; "
                    f"max(s,c)<{self.abstain_below} -> Abstain; else Insufficient. "
                    "ALL FLOORS ARE UNTUNED PLACEHOLDERS."
                ),
                "explanation": (
                    f"Over {len(v)} evidence pieces, best support={s:.3f} "
                    f"(from {best_s.evidence_id}, rank {best_s.evidence_rank}) and best "
                    f"contradiction={_fmt(c) if has_nli else 'n/a'}"
                    + (f" (from {best_c.evidence_id}, rank {best_c.evidence_rank})" if has_nli else "")
                    + f". Verdict {label.value} because {why}."
                    + (
                        " Note: Abstain means the system declined to answer, which is not "
                        "the same as asserting the evidence is insufficient."
                        if abstained else ""
                    )
                ),
                "decisive_evidence_ids": [driver.evidence_id],
                "signal": signal_kind(v),
                "n_evidence": len(v),
                "best_support": s,
                "best_contradict": c if has_nli else None,
                **floors,
            },
        )


# --------------------------------------------------------------------------- #
# 5. NULL BASELINE
# --------------------------------------------------------------------------- #

@register("aggregator", "majority")
class MajorityAggregator(Aggregator):
    """NULL BASELINE. One vote per evidence piece; probability magnitudes discarded.

    Each verdict votes for its own argmax class, votes are counted, and the plurality
    wins. Ties go to ``Insufficient``.

    **Why this is worth running.** It is the aggregation equivalent of
    ``ClaimOnlyVerifier``: it throws away exactly the information the other four
    aggregators exist to use. A confident 0.99 counts the same as a marginal 0.34. If a
    real aggregator cannot beat this, the differences between the real ones are not
    about how they weight confidence, and the aggregation slot is not where the
    research question lives after all. That is a finding, and a cheap one.
    """

    def aggregate(self, claim: Claim, v: list[EvidenceVerdict]) -> ClaimVerdict:
        if not v:
            return _empty_verdict(claim, self.name)

        votes: dict[Label, list[str]] = {Label.SUPPORTED: [], Label.CONTRADICTED: [], Label.INSUFFICIENT: []}
        for x in v:
            s, c, n = support_contra_neutral(x)
            if c is None:  # similarity-only: no contradiction signal exists
                choice = Label.SUPPORTED if s >= 0.5 else Label.INSUFFICIENT
            else:
                triple = {Label.SUPPORTED: s, Label.CONTRADICTED: c, Label.INSUFFICIENT: n}
                choice = max(triple, key=lambda k: triple[k])
            votes[choice].append(x.evidence_id)

        counts = {label: len(ids) for label, ids in votes.items()}
        top = max(counts.values())
        winners = [label for label, count in counts.items() if count == top]
        label = Label.INSUFFICIENT if len(winners) > 1 else winners[0]

        return ClaimVerdict(
            claim_id=claim.id,
            label=label,
            confidence=float(counts[label] / len(v)),
            abstained=False,
            per_evidence=tuple(v),
            aggregator_name=self.name,
            aggregation_trace={
                "rule": (
                    "one vote per evidence piece for its own argmax class; plurality "
                    "wins; ties -> Insufficient. PROBABILITY MAGNITUDES ARE DISCARDED "
                    "-- this is a null baseline. NLI-neutral votes as Insufficient."
                ),
                "explanation": (
                    f"{len(v)} evidence pieces voted "
                    f"{counts[Label.SUPPORTED]} supported / "
                    f"{counts[Label.CONTRADICTED]} contradicted / "
                    f"{counts[Label.INSUFFICIENT]} insufficient"
                    + (
                        f", a tie between {[w.value for w in winners]} resolved to Insufficient."
                        if len(winners) > 1
                        else f", so {label.value} wins with {counts[label]}/{len(v)} votes."
                    )
                    + " A 0.99 vote and a 0.34 vote counted equally here."
                ),
                "decisive_evidence_ids": votes[label],
                "signal": signal_kind(v),
                "n_evidence": len(v),
                "votes": {k.value: c for k, c in counts.items()},
                "tie": len(winners) > 1,
            },
        )


__all__ = [
    "MajorityAggregator",
    "MaxEntailmentAggregator",
    "NoisyOrAggregator",
    "ThresholdWithAbstainAggregator",
    "WeightedByRetrievalAggregator",
    "signal_kind",
    "support_contra_neutral",
]
