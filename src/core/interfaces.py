"""Abstract base classes for every swappable pipeline stage.

Five stages, five ABCs, one rule each.  The signatures are the contract; the
docstrings state the invariants that the ``check_*`` helpers at the bottom of this
file enforce at runtime (they are cheap -- O(k) over a list of <= 100 items -- so
the pipeline calls them on every stage boundary rather than only in tests).

The one design decision worth defending in a viva:

    ``Verifier.score`` takes ONE Evidence, not a list.

That is not a stylistic choice.  If the verifier saw ``list[Evidence]`` it could
pool internally, and pooling would then be an implementation detail of whichever
verifier you happened to pick -- untraceable, unswappable, and confounded with the
NLI model itself.  Making the signature single-pair means aggregation *cannot*
happen anywhere except in an Aggregator, which is a separate registry entry with
its own name in the trace and its own mandatory ``aggregation_trace``.  The cost is
k forward passes per claim instead of one batched call; that cost is paid back the
first time you want to compare four aggregators on identical per-pair scores
without re-running the NLI model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from src.core.types import Claim, ClaimVerdict, Evidence, EvidenceVerdict


class Component(ABC):
    """Common base: gives every component a stable name for the trace.

    ``registry_name`` is set on the class by the ``@register`` decorator, so a
    component's name in a trace is exactly the string you type in the YAML.  There
    is no second source of truth to drift.
    """

    #: Set by src.core.registry.register(); "<unregistered>" means the class was
    #: instantiated directly (fine in tests, a bug in the pipeline).
    registry_name: ClassVar[str] = "<unregistered>"

    #: Set by registry.build() to the resolved constructor kwargs, so the trace can
    #: record what a component was actually configured with, not just its name.
    resolved_params: dict[str, Any]

    def __init__(self) -> None:
        # Subclasses that forget to call super().__init__() still work: describe()
        # falls back via getattr.  This is deliberate leniency at a boundary that
        # would otherwise produce confusing AttributeErrors deep in a run.
        self.resolved_params = {}

    @property
    def name(self) -> str:
        return type(self).registry_name

    def describe(self) -> dict[str, Any]:
        """What goes into ``Trace.resolved_config`` for this component."""
        return {
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "params": dict(getattr(self, "resolved_params", {}) or {}),
            "model_identity": getattr(self, "model_identity", None),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<{type(self).__name__} name={self.name!r}>"


class ClaimExtractor(Component):
    """Response text -> atomic claims.

    Contract:
      * ids unique within the returned list;
      * every claim carries ``response_id`` and ``extractor_name``;
      * ``source_span``, when not None, must satisfy
        ``span.text_from(response)`` being a substring of the response -- i.e. the
        span indexes the ORIGINAL string, not a normalised copy.  Normalising
        before splitting and then reporting spans into the normalised text is the
        single easiest way to make the frontend's highlighting silently wrong.
    """

    @abstractmethod
    def extract(self, response: str) -> list[Claim]:
        ...


class Retriever(Component):
    """Claim -> ranked evidence from the corpus.

    Contract:
      * at most ``k`` items;
      * ``rank`` is 1-based, contiguous, ascending in the returned order;
      * evidence ids unique (no duplicate corpus sentence at two ranks);
      * ``score`` is the retriever's own score, NOT normalised across retrievers --
        the frontend shows BM25 and dense scores separately on purpose, so
        squashing them into a common scale here would destroy information.
    """

    @abstractmethod
    def retrieve(self, claim: Claim, k: int) -> list[Evidence]:
        ...


class Reranker(Component):
    """Reorder (and optionally truncate) a candidate list.

    Contract:
      * the returned set is a SUBSET of the input by evidence id -- a reranker may
        not invent, substitute, or mutate corpus text;
      * ranks renumbered 1..len(result);
      * ``score`` replaced by the reranker's score, ``retriever_name`` updated to
        record which component produced this ordering.
    """

    @abstractmethod
    def rerank(self, claim: Claim, ev: list[Evidence], k: int) -> list[Evidence]:
        ...


class Verifier(Component):
    """Score exactly ONE (claim, evidence) pair.  See module docstring.

    Contract:
      * returns a verdict whose ``claim_id``/``evidence_id`` match the inputs;
      * sets ``latency_ms`` from a real measurement, not a constant;
      * MUST NOT consult any other evidence, any other claim, or any label.

    Null baseline note: ``ClaimOnlyVerifier`` deliberately ignores ``ev`` in its
    scoring while still returning a verdict keyed to it.  That is legal and is the
    point -- it keeps the null baseline on exactly the same code path so the
    comparison is apples-to-apples.
    """

    @abstractmethod
    def score(self, claim: Claim, ev: Evidence) -> EvidenceVerdict:
        ...


class Aggregator(Component):
    """Per-evidence verdicts -> one claim-level decision.

    Contract:
      * pure function of ``(claim, verdicts)``: no retrieval, no model calls, no
        corpus access.  If an aggregator needs a signal, that signal must already
        be inside an EvidenceVerdict, which forces it to be visible in the trace;
      * must populate ``aggregation_trace`` with at least
        ``rule``, ``explanation``, ``decisive_evidence_ids``
        (enforced by ClaimVerdict.__post_init__);
      * must handle the empty list -- zero retrieved evidence is a normal case and
        is exactly the "retrieval failure vs genuine insufficiency" boundary.
    """

    @abstractmethod
    def aggregate(self, claim: Claim, v: list[EvidenceVerdict]) -> ClaimVerdict:
        ...


# --------------------------------------------------------------------------- #
# Runtime contract checks -- called at stage boundaries by the pipeline
# --------------------------------------------------------------------------- #

class ContractError(AssertionError):
    """A component violated its interface contract."""


def check_claims(claims: Sequence[Claim], response: str) -> None:
    ids = [c.id for c in claims]
    if len(set(ids)) != len(ids):
        raise ContractError("ClaimExtractor returned duplicate claim ids")
    for c in claims:
        if c.source_span is not None:
            if not 0 <= c.source_span.start < c.source_span.end <= len(response):
                raise ContractError("Claim source span outside response or empty")
            span_text = c.source_span.text_from(response)
            if span_text != c.text:
                if c.extractor_meta.get("span_is_exact") is False:
                    pass
                else:
                    raise ContractError(
                        f"claim {c.id!r} has a source_span whose text does not match the claim text, "
                        f"and is not explicitly marked as an approximate span."
                    )


def check_evidence_list(evidence: Sequence[Evidence], k: int, stage: str) -> None:
    if len(evidence) > k:
        raise ContractError(f"{stage} returned {len(evidence)} items for k={k}")
    ids = [e.id for e in evidence]
    if len(set(ids)) != len(ids):
        raise ContractError(f"{stage} returned duplicate evidence ids: {ids}")
    expected_ranks = list(range(1, len(evidence) + 1))
    if [e.rank for e in evidence] != expected_ranks:
        raise ContractError(
            f"{stage} ranks must be 1-based and contiguous in list order; "
            f"got {[e.rank for e in evidence]}"
        )


def check_rerank_is_subset(before: Sequence[Evidence], after: Sequence[Evidence]) -> None:
    unknown = {e.id for e in after} - {e.id for e in before}
    if unknown:
        raise ContractError(f"Reranker introduced evidence not in its input: {sorted(unknown)}")
    original = {e.id: e for e in before}
    for evidence in after:
        prior = original[evidence.id]
        if (evidence.doc_id, evidence.sent_id, evidence.text) != (prior.doc_id, prior.sent_id, prior.text):
            raise ContractError("Reranker changed evidence content/address under an existing ID")


def check_pair_verdict(verdict: EvidenceVerdict, claim: Claim, ev: Evidence) -> None:
    if verdict.claim_id != claim.id or verdict.evidence_id != ev.id:
        raise ContractError(
            f"Verifier returned a verdict for ({verdict.claim_id}, {verdict.evidence_id}) "
            f"when asked about ({claim.id}, {ev.id})"
        )


__all__ = [
    "Aggregator",
    "ClaimExtractor",
    "Component",
    "ContractError",
    "Reranker",
    "Retriever",
    "Verifier",
    "check_claims",
    "check_evidence_list",
    "check_pair_verdict",
    "check_rerank_is_subset",
]
