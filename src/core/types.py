"""Data contracts for the claim-level hallucination-detection pipeline.

Everything that crosses a stage boundary is one of the frozen dataclasses in this
module.  Nothing here imports torch, transformers, or faiss: a saved trace must be
loadable by the frontend (or by a plain ``python -c``) on a machine with none of
the model stack installed.

Design rules encoded here, in decreasing order of how much pain they save:

1. ``Evidence.id`` is a pure function of ``(doc_id, sent_id)``.  It does NOT depend
   on the retriever, the score, or the rank.  Fusion (RRF) and oracle substitution
   both need to say "these two hits are the same sentence", and that is only sound
   if identity is corpus identity.
2. ``EvidenceVerdict`` is per (claim, evidence) pair.  There is no field in which a
   verifier could hide a pooled-over-k score.  Pooling is the aggregator's job, and
   the aggregator is a separate object with its own trace.
3. ``Trace`` cross-validates its own references on construction.  A verdict that
   points at evidence the claim never saw is an index-alignment bug, and those are
   silent unless something checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# Bump when a field is added/removed/retyped.  Saved traces are an error-analysis
# corpus that will outlive several refactors; the frontend must be able to refuse
# to render a trace it does not understand rather than mis-render it.
#   1.1.0  Evidence.retriever_meta added (additive; readers of 1.0.x traces get {}).
#   1.2.0  EvidenceVerdict.evidence_rank / evidence_score added (additive).
SCHEMA_VERSION = "1.2.0"

# Tolerance for "these three numbers are a distribution".  fp16 softmax output on
# GPU is not exact; 1e-3 is loose enough for fp16 and tight enough to catch a real
# bug (e.g. logits stored instead of probabilities).
PROB_SUM_TOL = 1e-3

_ID_FIELD_SEP = b"\x1f"  # ASCII unit separator; will not occur in normal text


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

class Label(str, Enum):
    """Verdict labels for a claim.

    ``str`` mixin so that ``json.dumps`` emits ``"Supported"`` rather than an
    object, and so pandas/Streamlit treat it as a string without conversion.

    ABSTAIN is deliberately a *label*, not only a flag: a selective-prediction
    system that abstains has still produced an output, and that output must be
    representable in the same field as the others for risk-coverage accounting.
    """

    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    INSUFFICIENT = "Insufficient"
    ABSTAIN = "Abstain"

    @classmethod
    def parse(cls, raw: str) -> Label:
        """Map a dataset label string onto our taxonomy.  Strict by design."""
        key = " ".join(str(raw).strip().upper().split())
        try:
            return _LABEL_ALIASES[key]
        except KeyError:
            raise ValueError(
                f"unknown label {raw!r}; known aliases: {sorted(_LABEL_ALIASES)}"
            ) from None


# Dataset label vocabularies only.  NLI class names (entailment / contradiction /
# neutral) are deliberately ABSENT: mapping NLI-neutral onto "Insufficient" is a
# modelling decision that belongs in an aggregator, where it shows up in
# aggregation_trace, not in a parser where it would be invisible.
_LABEL_ALIASES: dict[str, Label] = {
    "SUPPORTED": Label.SUPPORTED,
    "SUPPORTS": Label.SUPPORTED,            # FEVER
    "CONTRADICTED": Label.CONTRADICTED,
    "REFUTES": Label.CONTRADICTED,          # FEVER
    "INSUFFICIENT": Label.INSUFFICIENT,
    "NOT ENOUGH INFO": Label.INSUFFICIENT,  # FEVER
    "NEI": Label.INSUFFICIENT,
    "ABSTAIN": Label.ABSTAIN,
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def stable_id(prefix: str, *parts: object, length: int = 12) -> str:
    """Deterministic content-addressed id.

    Parts are separated by ASCII 0x1f before hashing so that ("ab", "c") and
    ("a", "bc") cannot collide -- the classic string-concatenation id bug.
    blake2b rather than md5/sha1 because it is fast and nobody has to explain in a
    viva why a broken hash was acceptable here.
    """
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(_ID_FIELD_SEP)
    return f"{prefix}_{h.hexdigest()[:length]}"


def evidence_id(doc_id: str, sent_id: int) -> str:
    """Canonical evidence identity: corpus position, nothing else.

    Human-readable on purpose -- when the frontend shows an id you should be able
    to find the sentence in the corpus without a lookup table.
    """
    return f"ev::{doc_id}::{int(sent_id)}"


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Sortable, collision-resistant run id: ``20260828T104500Z-3f9a21``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _check_finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _check_unit(name: str, value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    v = _check_finite(name, value)
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {v!r}")
    return v


def _check_json_safe(name: str, obj: Any) -> None:
    """Fail at construction, not at trace-write time three hours in."""
    try:
        json.dumps(obj)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON-serialisable: {exc}") from None


# --------------------------------------------------------------------------- #
# Claim
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Half-open character span ``[start, end)`` into the *original* response text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")

    def text_from(self, response: str) -> str:
        return response[self.start:self.end]

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SourceSpan:
        return cls(start=int(d["start"]), end=int(d["end"]))


@dataclass(frozen=True, slots=True)
class Claim:
    """One atomic factual assertion extracted from a response."""

    id: str
    response_id: str
    text: str
    source_span: SourceSpan | None
    extractor_name: str
    extractor_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("claim text must be non-empty")
        _check_json_safe("extractor_meta", self.extractor_meta)

    @classmethod
    def new(
        cls,
        response_id: str,
        text: str,
        extractor_name: str,
        source_span: SourceSpan | None = None,
        extractor_meta: dict[str, Any] | None = None,
    ) -> Claim:
        """Construct with a content-derived id.

        The id hashes ``(response_id, span, text)``.  Consequence, and it is the
        behaviour we want: editing a claim's wording in the frontend yields a
        *different* claim id, so its downstream verdicts can never be confused
        with the original's in a cache or in a saved trace.
        """
        span_key = "" if source_span is None else f"{source_span.start}:{source_span.end}"
        return cls(
            id=stable_id("clm", response_id, span_key, text),
            response_id=response_id,
            text=text,
            source_span=source_span,
            extractor_name=extractor_name,
            extractor_meta=dict(extractor_meta or {}),
        )

    def edited(self, new_text: str) -> Claim:
        """Frontend claim-editing: new id, breadcrumb back to the claim it came from."""
        meta = dict(self.extractor_meta)
        meta["edited_from"] = self.id
        meta["edited_from_text"] = self.text
        return Claim.new(
            response_id=self.response_id,
            text=new_text,
            extractor_name=f"{self.extractor_name}+manual_edit",
            source_span=self.source_span,
            extractor_meta=meta,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "response_id": self.response_id,
            "text": self.text,
            "source_span": None if self.source_span is None else self.source_span.to_dict(),
            "extractor_name": self.extractor_name,
            "extractor_meta": dict(self.extractor_meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Claim:
        span = d.get("source_span")
        return cls(
            id=str(d["id"]),
            response_id=str(d["response_id"]),
            text=str(d["text"]),
            source_span=None if span is None else SourceSpan.from_dict(span),
            extractor_name=str(d["extractor_name"]),
            extractor_meta=dict(d.get("extractor_meta") or {}),
        )


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class Evidence:
    """One corpus sentence, as seen by one retriever at one rank."""

    id: str
    doc_id: str
    sent_id: int
    text: str
    score: float
    retriever_name: str
    rank: int
    is_gold: bool | None = None  # None = unknown (unannotated), not "no"
    #: Per-retriever detail the UI shows in its own column: for a fused result, the
    #: per-arm scores and per-arm ranks that produced `score`. Kept out of `score`
    #: because `score` must stay a single comparable number per retriever, and kept
    #: out of a parallel structure because it must survive JSON round-trip with the
    #: evidence it describes.
    retriever_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = evidence_id(self.doc_id, self.sent_id)
        if self.id != expected:
            raise ValueError(
                "Evidence.id must be derived from (doc_id, sent_id): "
                f"got {self.id!r}, expected {expected!r}. Use Evidence.new()."
            )
        if self.sent_id < 0:
            raise ValueError(f"sent_id must be >= 0, got {self.sent_id}")
        if self.rank < 1:
            raise ValueError(f"rank is 1-based, got {self.rank}")
        _check_finite("score", self.score)
        _check_json_safe("retriever_meta", self.retriever_meta)

    @classmethod
    def new(
        cls,
        doc_id: str,
        sent_id: int,
        text: str,
        score: float,
        retriever_name: str,
        rank: int,
        is_gold: bool | None = None,
        retriever_meta: dict[str, Any] | None = None,
    ) -> Evidence:
        return cls(
            id=evidence_id(doc_id, sent_id),
            doc_id=doc_id,
            sent_id=int(sent_id),
            text=text,
            score=float(score),
            retriever_name=retriever_name,
            rank=int(rank),
            is_gold=is_gold,
            retriever_meta=dict(retriever_meta or {}),
        )

    def reranked(self, *, rank: int, score: float, retriever_name: str) -> Evidence:
        """Same corpus sentence, new position in a new ordering.  Id is unchanged."""
        return replace(self, rank=int(rank), score=float(score), retriever_name=retriever_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "sent_id": self.sent_id,
            "text": self.text,
            "score": self.score,
            "retriever_name": self.retriever_name,
            "rank": self.rank,
            "is_gold": self.is_gold,
            "retriever_meta": dict(self.retriever_meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Evidence:
        return cls(
            id=str(d["id"]),
            doc_id=str(d["doc_id"]),
            sent_id=int(d["sent_id"]),
            text=str(d["text"]),
            score=float(d["score"]),
            retriever_name=str(d["retriever_name"]),
            rank=int(d["rank"]),
            is_gold=d.get("is_gold"),
            retriever_meta=dict(d.get("retriever_meta") or {}),
        )


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class EvidenceVerdict:
    """Score for exactly ONE (claim, evidence) pair.

    ``p_entail`` / ``p_contra`` / ``p_neutral`` are either all present (a 3-way
    distribution summing to 1 within ``PROB_SUM_TOL``) or all None (for a verifier
    that produces no NLI probabilities, e.g. SimilarityVerifier).  ``similarity``
    is a raw cosine in [-1, 1] and is threshold-free on purpose -- thresholding is
    an aggregator decision.
    """

    claim_id: str
    evidence_id: str
    p_entail: float | None
    p_contra: float | None
    p_neutral: float | None
    similarity: float | None
    verifier_name: str
    latency_ms: float
    #: Denormalised from the Evidence this verdict scores. An Aggregator receives only
    #: verdicts (see interfaces.py), so a retrieval-weighted aggregator would otherwise
    #: have no way to see rank or retrieval score without being handed the corpus --
    #: which would break its purity. Copying them here keeps the signal inside the
    #: trace, where it is visible, rather than smuggled in through a side channel.
    evidence_rank: int | None = None
    evidence_score: float | None = None

    def __post_init__(self) -> None:
        probs = (self.p_entail, self.p_contra, self.p_neutral)
        n_present = sum(p is not None for p in probs)
        if n_present not in (0, 3):
            raise ValueError(
                "p_entail/p_contra/p_neutral must be all-None or all-present, "
                f"got {probs!r}"
            )
        if n_present == 3:
            for name, p in zip(("p_entail", "p_contra", "p_neutral"), probs):
                _check_unit(name, float(p))
            total = sum(float(p) for p in probs)
            if abs(total - 1.0) > PROB_SUM_TOL:
                raise ValueError(
                    f"NLI probabilities must sum to 1 (+/-{PROB_SUM_TOL}), got {total!r}. "
                    "Was the softmax skipped, or were logits stored instead?"
                )
        if self.similarity is not None:
            _check_unit("similarity", float(self.similarity), lo=-1.0, hi=1.0)
        if n_present == 0 and self.similarity is None:
            raise ValueError("an EvidenceVerdict with no probabilities and no similarity is empty")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {self.latency_ms}")
        if self.evidence_rank is not None and self.evidence_rank < 1:
            raise ValueError(f"evidence_rank is 1-based, got {self.evidence_rank}")
        if self.evidence_score is not None:
            _check_finite("evidence_score", self.evidence_score)

    @property
    def has_nli(self) -> bool:
        return self.p_entail is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "p_entail": self.p_entail,
            "p_contra": self.p_contra,
            "p_neutral": self.p_neutral,
            "similarity": self.similarity,
            "verifier_name": self.verifier_name,
            "latency_ms": self.latency_ms,
            "evidence_rank": self.evidence_rank,
            "evidence_score": self.evidence_score,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EvidenceVerdict:
        def _opt(key: str) -> float | None:
            v = d.get(key)
            return None if v is None else float(v)

        return cls(
            claim_id=str(d["claim_id"]),
            evidence_id=str(d["evidence_id"]),
            p_entail=_opt("p_entail"),
            p_contra=_opt("p_contra"),
            p_neutral=_opt("p_neutral"),
            similarity=_opt("similarity"),
            verifier_name=str(d["verifier_name"]),
            latency_ms=float(d["latency_ms"]),
            evidence_rank=None if d.get("evidence_rank") is None else int(d["evidence_rank"]),
            evidence_score=None if d.get("evidence_score") is None else float(d["evidence_score"]),
        )


# Keys every aggregator MUST write into aggregation_trace.  Hard contract, because
# the Aggregation panel renders "why", not just "what", and a panel that silently
# shows nothing for one aggregator is worse than a crash.
REQUIRED_AGGREGATION_TRACE_KEYS = ("rule", "explanation", "decisive_evidence_ids")


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """Final per-claim decision, plus every per-evidence verdict that produced it."""

    claim_id: str
    label: Label
    confidence: float
    abstained: bool
    per_evidence: tuple[EvidenceVerdict, ...]
    aggregator_name: str
    aggregation_trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.label, Label):
            raise TypeError(f"label must be a Label enum, got {type(self.label).__name__}")
        _check_unit("confidence", self.confidence)
        if not isinstance(self.per_evidence, tuple):
            raise TypeError("per_evidence must be a tuple (a frozen dataclass holds no lists)")
        for ev in self.per_evidence:
            if ev.claim_id != self.claim_id:
                raise ValueError(
                    f"per_evidence contains a verdict for claim {ev.claim_id!r} "
                    f"inside a ClaimVerdict for {self.claim_id!r}"
                )
        missing = [k for k in REQUIRED_AGGREGATION_TRACE_KEYS if k not in self.aggregation_trace]
        if missing:
            raise ValueError(
                f"aggregation_trace missing required keys {missing} "
                f"(aggregator={self.aggregator_name!r}); the UI renders these."
            )
        _check_json_safe("aggregation_trace", self.aggregation_trace)
        if self.label is Label.ABSTAIN and not self.abstained:
            raise ValueError("label=Abstain requires abstained=True")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "label": self.label.value,
            "confidence": self.confidence,
            "abstained": self.abstained,
            "per_evidence": [v.to_dict() for v in self.per_evidence],
            "aggregator_name": self.aggregator_name,
            "aggregation_trace": dict(self.aggregation_trace),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ClaimVerdict:
        return cls(
            claim_id=str(d["claim_id"]),
            label=Label(d["label"]),
            confidence=float(d["confidence"]),
            abstained=bool(d["abstained"]),
            per_evidence=tuple(EvidenceVerdict.from_dict(v) for v in d.get("per_evidence", [])),
            aggregator_name=str(d["aggregator_name"]),
            aggregation_trace=dict(d.get("aggregation_trace") or {}),
        )


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class Trace:
    """Everything one pipeline run did to one response.  The unit of the JSONL log.

    ``mode`` distinguishes a normal run from the oracle run (gold evidence
    substituted for retrieved evidence).  The oracle comparison is the whole point
    of retrieval-attribution analysis, so the two runs must stay distinguishable
    after the fact, from the file alone.
    """

    run_id: str
    config_hash: str
    git_sha: str | None
    timestamp: str
    response_text: str
    claims: tuple[Claim, ...]
    evidence_by_claim: dict[str, tuple[Evidence, ...]]
    verdicts: tuple[ClaimVerdict, ...]
    timings: dict[str, float] = field(default_factory=dict)
    mode: str = "retrieved"          # "retrieved" | "oracle"
    resolved_config: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        claim_ids = [c.id for c in self.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("duplicate claim ids in trace")
        known = set(claim_ids)

        for cid in self.evidence_by_claim:
            if cid not in known:
                raise ValueError(f"evidence_by_claim references unknown claim {cid!r}")

        for verdict in self.verdicts:
            if verdict.claim_id not in known:
                raise ValueError(f"verdict references unknown claim {verdict.claim_id!r}")
            available = {e.id for e in self.evidence_by_claim.get(verdict.claim_id, ())}
            scored = {ev.evidence_id for ev in verdict.per_evidence}
            dangling = scored - available
            if dangling:
                raise ValueError(
                    f"claim {verdict.claim_id!r}: verdict scores evidence never given to it: "
                    f"{sorted(dangling)}. This is an index-alignment bug."
                )
        if self.mode not in ("retrieved", "oracle"):
            raise ValueError(f"mode must be 'retrieved' or 'oracle', got {self.mode!r}")
        _check_json_safe("resolved_config", self.resolved_config)

    def claim_by_id(self, claim_id: str) -> Claim:
        for c in self.claims:
            if c.id == claim_id:
                return c
        raise KeyError(claim_id)

    def verdict_by_claim(self, claim_id: str) -> ClaimVerdict | None:
        for v in self.verdicts:
            if v.claim_id == claim_id:
                return v
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "git_sha": self.git_sha,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "response_text": self.response_text,
            "claims": [c.to_dict() for c in self.claims],
            "evidence_by_claim": {
                cid: [e.to_dict() for e in evs] for cid, evs in self.evidence_by_claim.items()
            },
            "verdicts": [v.to_dict() for v in self.verdicts],
            "timings": dict(self.timings),
            "resolved_config": dict(self.resolved_config),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Trace:
        got = str(d.get("schema_version", "0.0.0"))
        if got.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"trace schema_version {got} is incompatible with {SCHEMA_VERSION}")
        return cls(
            run_id=str(d["run_id"]),
            config_hash=str(d["config_hash"]),
            git_sha=d.get("git_sha"),
            timestamp=str(d["timestamp"]),
            response_text=str(d["response_text"]),
            claims=tuple(Claim.from_dict(c) for c in d.get("claims", [])),
            evidence_by_claim={
                cid: tuple(Evidence.from_dict(e) for e in evs)
                for cid, evs in (d.get("evidence_by_claim") or {}).items()
            },
            verdicts=tuple(ClaimVerdict.from_dict(v) for v in d.get("verdicts", [])),
            timings=dict(d.get("timings") or {}),
            mode=str(d.get("mode", "retrieved")),
            resolved_config=dict(d.get("resolved_config") or {}),
            schema_version=got,
        )

    def to_json_line(self) -> str:
        """One trace, one line.  ensure_ascii=False keeps non-ASCII evidence readable."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json_line(cls, line: str) -> Trace:
        return cls.from_dict(json.loads(line))


__all__ = [
    "PROB_SUM_TOL",
    "REQUIRED_AGGREGATION_TRACE_KEYS",
    "SCHEMA_VERSION",
    "Claim",
    "ClaimVerdict",
    "Evidence",
    "EvidenceVerdict",
    "Label",
    "SourceSpan",
    "Trace",
    "evidence_id",
    "new_run_id",
    "stable_id",
    "utcnow_iso",
]
