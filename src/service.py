"""Shared application service layer for Nullius.

Both the command-line interface (src.cli) and the FastAPI backend (src.api) route
through this layer. It coordinates:
  - Configuration loading, overrides, and caching
  - Pipeline lifecycle and component reuse (avoiding repeated model reloads)
  - Execution of retrieved analysis, oracle analysis, quick checks, and full verification
  - Trace generation, provenance, and thread-safe annotation persistence
  - Diagnostic health metrics and concurrency control

Central Invariant Preserved:
  The pipeline execution remains strictly:
      Claim -> Retriever -> Evidence[] -> Pairwise Verifier -> EvidenceVerdict[] -> Aggregator -> ClaimVerdict
  The verifier strictly scores individual (Claim, Evidence) pairs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import (
    apply_overrides,
    config_hash,
    load_config,
    project_root,
    set_global_seed,
)
from src.core.interfaces import (
    check_claims,
    check_evidence_list,
    check_pair_verdict,
    check_rerank_is_subset,
)
from src.core.registry import Pipeline, build, build_pipeline
from src.core.trace_io import (
    ALLOWED_FAILURE_CATEGORIES,
)
from src.core.trace_io import (
    save_failure_case as save_failure_case_file,
)
from src.core.types import (
    SCHEMA_VERSION,
    Claim,
    ClaimVerdict,
    Evidence,
    EvidenceVerdict,
    Trace,
    utcnow_iso,
)
from src.data.corpus import Corpus
from src.data.examples import Example, load_examples
from src.pipeline import RunContext, _timed, _Timer, analyze


@dataclass
class ExecutionState:
    run_id: str
    config_hash: str
    git_sha: str | None
    claims_inputs: dict[str, tuple[Claim, tuple[EvidenceVerdict, ...]]]
    expires_at: float
    trace: Trace
    comparisons: dict[str, Any] = field(default_factory=dict)


def _resolve_repo_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


@dataclass(frozen=True, slots=True)
class QuickClaimResult:
    """Tier 1 evidence-availability result for one claim."""

    claim_id: str
    claim_text: str
    source_span: dict[str, int] | None
    status: str  # "likely-checkable", "no-evidence-found", "pending"
    n_evidence_found: int
    top_evidence_score: float | None
    top_evidence_text: str | None
    top_evidence_source: str | None


@dataclass(frozen=True, slots=True)
class QuickVerificationResult:
    """Tier 1 summary result for a full response."""

    response_text: str
    claims: tuple[QuickClaimResult, ...]
    counts: dict[str, int]  # {"likely_checkable": int, "no_evidence_found": int, "total": int}
    timings: dict[str, float]
    config_hash: str


class NulliusService:
    """Orchestrates configuration, component lifecycle, and execution workflows."""

    def __init__(
        self,
        default_config_path: str | Path = "configs/mini.yaml",
        preload: bool = False,
        annotation_dir: str | Path = "data/annotations",
        failure_dir: str | Path = "results/failure_cases",
    ) -> None:
        self.default_config_path = str(default_config_path)
        self.annotation_dir = Path(annotation_dir)
        self.failure_dir = Path(failure_dir)
        self._lock = asyncio.Lock()
        self._thread_lock = threading.Lock()
        self._queue_depth = 0
        self._cached_config: dict[str, Any] | None = None
        self._cached_pipeline: Pipeline | None = None
        self._cached_pipeline_key = None
        self._cached_corpus: dict[tuple[str, str], Corpus] = {}
        self._cached_examples: dict[tuple[str, str], dict[str, Example]] = {}
        self._execution_store: dict[str, ExecutionState] = {}
        self._store_lock = threading.Lock()
        self._MAX_STORE_SIZE = 50
        self._STORE_TTL = 3600.0

        if preload:
            self._ensure_loaded()

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    def _resolve_config(
        self,
        config_path: str | Path | None = None,
        overrides: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        path = config_path or self.default_config_path
        cfg = load_config(_resolve_repo_path(path))
        if overrides:
            cfg = apply_overrides(cfg, overrides)
        set_global_seed(int(cfg.get("seed", 1337)))
        return cfg

    def _ensure_loaded(
        self,
        config_path: str | Path | None = None,
        overrides: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], Pipeline]:
        with self._thread_lock:
            cfg = self._resolve_config(config_path, overrides)
            data_key = tuple((name, hashlib.sha256(_resolve_repo_path(path).read_bytes()).hexdigest())
                             for name, path in sorted(cfg.get("paths", {}).items())
                             if name in {"corpus", "examples"})
            cache_key = (config_hash(cfg), data_key)
            if cache_key == self._cached_pipeline_key and self._cached_pipeline is not None:
                return self._cached_config, self._cached_pipeline
            pipe = build_pipeline(cfg)

            # Cache if using default configuration
            if config_path is None and not overrides:
                self._cached_config = cfg
                self._cached_pipeline = pipe
                self._cached_pipeline_key = cache_key

            return cfg, pipe

    def get_corpus(self, cfg: Mapping[str, Any]) -> Corpus:
        with self._thread_lock:
            corpus_path = _resolve_repo_path(cfg["paths"]["corpus"]).resolve()
            key = (str(corpus_path), hashlib.sha256(corpus_path.read_bytes()).hexdigest())
            if key not in self._cached_corpus:
                self._cached_corpus[key] = Corpus.from_jsonl(corpus_path)
            return self._cached_corpus[key]

    def get_examples(self, cfg: Mapping[str, Any]) -> dict[str, Example]:
        with self._thread_lock:
            examples_path = _resolve_repo_path(cfg["paths"]["examples"]).resolve()
            key = (str(examples_path), hashlib.sha256(examples_path.read_bytes()).hexdigest())
            if key not in self._cached_examples:
                examples_list = load_examples(examples_path)
                self._cached_examples[key] = {ex.id: ex for ex in examples_list}
            return self._cached_examples[key]

    def get_health(self) -> dict[str, Any]:
        """Diagnostic state for monitoring and extension health polling."""
        cfg, pipe = self._ensure_loaded()
        cuda_avail = False
        device_name = "cpu"
        vram_allocated_mb: float | None = None

        try:
            import torch

            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                device_name = torch.cuda.get_device_name(0)
                vram_allocated_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
        except ImportError:
            pass

        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "config_hash": config_hash(cfg),
            "harness_kind": (cfg.get("harness") or {}).get("kind", "debug"),
            "device": device_name,
            "cuda_available": cuda_avail,
            "vram_allocated_mb": vram_allocated_mb,
            "queue_depth": self._queue_depth,
            "components": pipe.describe(),
        }

    async def analyze(
        self,
        response_text: str,
        *,
        config_path: str | Path | None = None,
        overrides: Sequence[str] | None = None,
        retrieve_k: int | None = None,
        check_contracts: bool = True,
    ) -> Trace:
        """Run standard retrieved verification pipeline on a response text."""
        if not response_text or not response_text.strip():
            raise ValueError("response_text must be non-empty")

        self._queue_depth += 1
        try:
            async with self._lock:
                cfg, pipe = await asyncio.to_thread(self._ensure_loaded, config_path, overrides)
                ctx = RunContext.create(cfg, pipe)
                corpus = self.get_corpus(cfg)
                trace = await asyncio.to_thread(
                    analyze, pipe,
                    response_text,
                    ctx,
                    corpus=corpus,
                    retrieve_k=retrieve_k,
                    check_contracts=check_contracts,
                    mode="retrieved",
                )
                return self._store_run(trace)
        finally:
            self._queue_depth -= 1

    async def analyze_oracle(
        self,
        *,
        example_id: str | None = None,
        response_text: str | None = None,
        config_path: str | Path | None = None,
        overrides: Sequence[str] | None = None,
        retrieve_k: int | None = None,
        check_contracts: bool = True,
    ) -> Trace:
        """Run oracle-mode verification with gold evidence substituted for retrieved evidence."""
        self._queue_depth += 1
        try:
            async with self._lock:
                cfg, pipe = await asyncio.to_thread(self._ensure_loaded, config_path, overrides)
                corpus = self.get_corpus(cfg)
                examples = self.get_examples(cfg)

                target_example: Example | None = None
                if example_id:
                    if example_id not in examples:
                        raise KeyError(f"example_id {example_id!r} not found in loaded examples")
                    target_example = examples[example_id]
                elif response_text:
                    for ex in examples.values():
                        if ex.text.strip() == response_text.strip():
                            target_example = ex
                            break
                    if target_example is None:
                        raise ValueError(
                            "No matching example found for provided response text in oracle corpus"
                        )
                else:
                    raise ValueError("Either example_id or response_text must be provided for oracle analysis")

                if not target_example.has_gold_evidence:
                    raise ValueError(f"Example {target_example.id!r} has no gold evidence annotated")

                ctx = RunContext.create(cfg, pipe)
                trace = await asyncio.to_thread(
                    analyze, pipe,
                    target_example.text,
                    ctx,
                    example=target_example,
                    corpus=corpus,
                    mode="oracle",
                    retrieve_k=retrieve_k,
                    check_contracts=check_contracts,
                )
                return self._store_run(trace)
        finally:
            self._queue_depth -= 1

    async def verify_quick(
        self,
        response_text: str,
        *,
        config_path: str | Path | None = None,
        overrides: Sequence[str] | None = None,
        evidence_floor_score: float = 0.0,
        retrieve_k: int = 5,
        check_contracts: bool = True,
    ) -> QuickVerificationResult:
        """Tier 1 fast check: extracts claims and computes evidence-availability signal.

        Does not run heavy NLI models. Never asserts whether a claim is false.
        """
        if not response_text or not response_text.strip():
            raise ValueError("response_text must be non-empty")

        self._queue_depth += 1
        try:
            async with self._lock:
                cfg, pipe = await asyncio.to_thread(self._ensure_loaded, config_path, overrides)
                timer = _Timer()

                with _timed(timer, "extract_ms"):
                    claims = await asyncio.to_thread(pipe.extractor.extract, response_text)
                if check_contracts:
                    check_claims(claims, response_text)

                quick_claims: list[QuickClaimResult] = []
                checkable_count = 0
                no_evidence_count = 0

                for claim in claims:
                    with _timed(timer, "retrieve_ms"):
                        candidates = await asyncio.to_thread(pipe.retriever.retrieve, claim, retrieve_k)
                    if check_contracts:
                        check_evidence_list(candidates, k=retrieve_k, stage="retrieve")

                    valid_hits = [e for e in candidates if e.score >= evidence_floor_score]
                    has_evidence = len(valid_hits) > 0

                    if has_evidence:
                        status = "likely-checkable"
                        checkable_count += 1
                        top_hit = valid_hits[0]
                        top_score = top_hit.score
                        top_text = top_hit.text
                        top_source = f"{top_hit.doc_id}:{top_hit.sent_id}"
                    else:
                        status = "no-evidence-found"
                        no_evidence_count += 1
                        top_score = None
                        top_text = None
                        top_source = None

                    quick_claims.append(
                        QuickClaimResult(
                            claim_id=claim.id,
                            claim_text=claim.text,
                            source_span=claim.source_span.to_dict() if claim.source_span else None,
                            status=status,
                            n_evidence_found=len(valid_hits),
                            top_evidence_score=top_score,
                            top_evidence_text=top_text,
                            top_evidence_source=top_source,
                        )
                    )

                counts = {
                    "likely_checkable": checkable_count,
                    "no_evidence_found": no_evidence_count,
                    "total_claims": len(claims),
                }

                return QuickVerificationResult(
                    response_text=response_text,
                    claims=tuple(quick_claims),
                    counts=counts,
                    timings=timer.as_dict(),
                    config_hash=config_hash(cfg),
                )
        finally:
            self._queue_depth -= 1

    async def verify_full_stream(
        self,
        response_text: str,
        *,
        config_path: str | Path | None = None,
        overrides: Sequence[str] | None = None,
        retrieve_k: int | None = None,
        check_contracts: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Tier 2 verification yielding per-claim results as each completes for SSE streaming."""
        if not response_text or not response_text.strip():
            raise ValueError("response_text must be non-empty")

        self._queue_depth += 1
        try:
            async with self._lock:
                cfg, pipe = await asyncio.to_thread(self._ensure_loaded, config_path, overrides)
                ctx = RunContext.create(cfg, pipe)
                timer = _Timer()
                k = pipe.k
                pool = int(retrieve_k or k)

                with _timed(timer, "extract_ms"):
                    claims = await asyncio.to_thread(pipe.extractor.extract, response_text)
                if check_contracts:
                    check_claims(claims, response_text)

                yield {
                    "event": "start",
                    "data": {
                        "run_id": ctx.run_id,
                        "config_hash": ctx.config_hash,
                        "n_claims": len(claims),
                        "claims": [c.to_dict() for c in claims],
                    },
                }

                evidence_by_claim: dict[str, tuple[Evidence, ...]] = {}
                verdicts: list[ClaimVerdict] = []

                for idx, claim in enumerate(claims, start=1):
                    # retrieve
                    with _timed(timer, "retrieve_ms"):
                        candidates = await asyncio.to_thread(pipe.retriever.retrieve, claim, pool)
                    if check_contracts:
                        check_evidence_list(candidates, k=pool, stage="retrieve")

                    # rerank
                    with _timed(timer, "rerank_ms"):
                        reranked = await asyncio.to_thread(pipe.reranker.rerank, claim, candidates, k)
                    if check_contracts:
                        check_rerank_is_subset(candidates, reranked)
                        check_evidence_list(reranked, k=k, stage="rerank")

                    evidence_by_claim[claim.id] = tuple(reranked)

                    # verify pairwise (strict Claim x Evidence invariant)
                    pair_verdicts: list[EvidenceVerdict] = []
                    with _timed(timer, "verify_ms"):
                        for ev in reranked:
                            pv = await asyncio.to_thread(pipe.verifier.score, claim, ev)
                            if check_contracts:
                                check_pair_verdict(pv, claim, ev)
                            pair_verdicts.append(pv)

                    # aggregate
                    with _timed(timer, "aggregate_ms"):
                        claim_verdict = pipe.aggregator.aggregate(deepcopy(claim), tuple(pair_verdicts))
                        if claim_verdict.per_evidence != tuple(pair_verdicts):
                            raise ValueError("Aggregator changed the original pairwise verdicts")
                        verdicts.append(claim_verdict)

                    yield {
                        "event": "claim_verdict",
                        "data": {
                            "index": idx,
                            "claim": claim.to_dict(),
                            "evidence": [e.to_dict() for e in reranked],
                            "verdict": claim_verdict.to_dict(),
                        },
                    }

                trace = Trace(
                    run_id=ctx.run_id,
                    config_hash=ctx.config_hash,
                    git_sha=ctx.git_sha,
                    timestamp=utcnow_iso(),
                    response_text=response_text,
                    claims=tuple(claims),
                    evidence_by_claim=evidence_by_claim,
                    verdicts=tuple(verdicts),
                    timings=timer.as_dict(),
                    mode="retrieved",
                    resolved_config={
                        **ctx.resolved_config,
                        "k": k,
                        "retrieve_k": pool,
                    },
                )

                trace = self._store_run(trace)
                yield {
                    "event": "complete",
                    "data": trace.to_dict(),
                }
        finally:
            self._queue_depth -= 1

    def _store_run(self, trace: Trace) -> Trace:
        """Capture the immutable inputs of a pipeline run in the bounded execution store."""
        stored_trace = Trace.from_dict(deepcopy(trace.to_dict()))
        # Clean up expired
        now = time.time()
        with self._store_lock:
            expired = [rid for rid, state in self._execution_store.items() if state.expires_at < now]
            for rid in expired:
                del self._execution_store[rid]

            # Build inputs map
            claims_inputs = {}
            for claim in stored_trace.claims:
                cv = stored_trace.verdict_by_claim(claim.id)
                evidence_verdicts = tuple(cv.per_evidence) if cv else ()
                claims_inputs[claim.id] = (claim, evidence_verdicts)

            # Insert
            self._execution_store[trace.run_id] = ExecutionState(
                run_id=trace.run_id,
                config_hash=trace.config_hash,
                git_sha=trace.git_sha,
                claims_inputs=claims_inputs,
                expires_at=now + self._STORE_TTL,
                trace=stored_trace,
            )

            # Evict oldest if full
            if len(self._execution_store) > self._MAX_STORE_SIZE:
                oldest = min(self._execution_store.values(), key=lambda s: s.expires_at)
                del self._execution_store[oldest.run_id]

        return trace

    def save_annotation(
        self,
        record: Mapping[str, Any],
        out_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Save a human annotation record to disk.

        Conforms to docs/CLAUDE_CODE_EXTENSION_PROMPT.md section 5:
          timestamp, site, model_name, response_text, claim_text, claim_span,
          extractor_name, retrieved_evidence, system_verdict, system_confidence,
          human_label, human_note, config_hash
        """
        required_keys = ("response_text", "claim_text", "human_label")
        missing = [k for k in required_keys if k not in record or record[k] is None]
        if missing:
            raise ValueError(f"Annotation record missing required fields: {missing}")

        target_dir = _resolve_repo_path(out_dir or self.annotation_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "annotations.jsonl"

        annotated_record: dict[str, Any] = {
            "timestamp": record.get("timestamp") or utcnow_iso(),
            "site": record.get("site", "direct_api"),
            "model_name": record.get("model_name"),
            "response_text": str(record["response_text"]),
            "claim_text": str(record["claim_text"]),
            "claim_span": record.get("claim_span"),
            "extractor_name": record.get("extractor_name"),
            "retrieved_evidence": record.get("retrieved_evidence", []),
            "system_verdict": record.get("system_verdict"),
            "system_confidence": record.get("system_confidence"),
            "human_label": str(record["human_label"]),
            "human_note": record.get("human_note", ""),
            "config_hash": record.get("config_hash"),
            "decomposition_disagreed": bool(record.get("decomposition_disagreed", False)),
        }

        line = json.dumps(annotated_record, ensure_ascii=False) + "\n"
        with self._thread_lock, target_file.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()

        return annotated_record

    def reaggregate(
        self,
        run_id: str,
        target_aggregators: list[str],
        aggregator_configs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Run pure aggregation steps over identical stored verdicts from a previous run.
        
        Throws KeyError if run_id is expired or unknown.
        Throws RegistryError/ValueError for bad aggregator configs.
        """
        with self._store_lock:
            state = self._execution_store.get(run_id)
            if state is None or state.expires_at < time.time():
                if state is not None:
                    del self._execution_store[run_id]
                raise KeyError(f"Run ID {run_id!r} not found or expired")
            # Snapshot the state we need
            claims_inputs = deepcopy(state.claims_inputs)
            config_hash = state.config_hash
            git_sha = state.git_sha

        comparisons: dict[str, dict[str, Any]] = {}

        # Instantiate aggregators once
        aggregators = {}
        for target in target_aggregators:
            cfg = aggregator_configs.get(target, {})
            spec = {"name": target, "params": cfg}
            aggregators[target] = build("aggregator", spec)

        for claim_id, (claim, verdicts) in claims_inputs.items():
            comparisons[claim_id] = {}
            for target, agg_instance in aggregators.items():
                # Pass identical, immutable data
                claim_verdict: ClaimVerdict = agg_instance.aggregate(deepcopy(claim), tuple(verdicts))
                if claim_verdict.per_evidence != verdicts:
                    raise ValueError("Aggregator changed original pairwise inputs")
                comparisons[claim_id][target] = claim_verdict.to_dict()
                comparisons[claim_id][target]["aggregation_trace"]["configuration"] = agg_instance.describe()

        configurations = {name: agg.describe() for name, agg in aggregators.items()}
        with self._store_lock:
            if run_id in self._execution_store:
                self._execution_store[run_id].comparisons = deepcopy(comparisons)

        return {
            "run_id": run_id,
            "config_hash": config_hash,
            "git_sha": git_sha,
            "comparisons": comparisons,
            "aggregator_configurations": configurations,
        }

    def save_failure_case(
        self,
        run_id: str,
        claim_id: str,
        failure_category: str,
        researcher_note: str,
        human_label: str | None = None,
        oracle_run_id: str | None = None,
        alternative_aggregations: dict[str, Any] | None = None,
        out_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Save a structured failure case linked to server-side execution state.

        Guarantees critical provenance rules:
          - Server-side ExecutionState is the single source of truth.
          - Client Trace/verdict payloads are never accepted as fallbacks.
          - Throws KeyError if run_id, claim_id, or oracle_run_id is expired/not found.
          - Throws ValueError if failure_category is invalid or run relationships mismatch.
        """
        if failure_category not in ALLOWED_FAILURE_CATEGORIES:
            raise ValueError(
                f"Invalid failure category {failure_category!r}. Allowed: {sorted(ALLOWED_FAILURE_CATEGORIES)}"
            )

        with self._store_lock:
            state = self._execution_store.get(run_id)
            if state is None or state.expires_at < time.time():
                if state is not None:
                    del self._execution_store[run_id]
                raise KeyError(f"Run ID {run_id!r} not found or expired")

            if state.trace.mode != "retrieved":
                raise ValueError(f"Source run must be in 'retrieved' mode, got {state.trace.mode!r}")

            if claim_id not in state.claims_inputs:
                raise KeyError(f"Claim ID {claim_id!r} not found in run {run_id!r}")

            oracle_trace = None
            if oracle_run_id is not None:
                oracle_state = self._execution_store.get(oracle_run_id)
                if oracle_state is None or oracle_state.expires_at < time.time():
                    if oracle_state is not None:
                        del self._execution_store[oracle_run_id]
                    raise KeyError(f"Oracle run ID {oracle_run_id!r} not found or expired")

                if oracle_state.trace.mode != "oracle":
                    raise ValueError(f"Run {oracle_run_id!r} has mode {oracle_state.trace.mode!r}, expected 'oracle'")

                if oracle_state.trace.response_text.strip() != state.trace.response_text.strip():
                    raise ValueError("Source run and oracle run have mismatched response text")
                if oracle_state.trace.resolved_config.get("data_identity") != state.trace.resolved_config.get("data_identity"):
                    raise ValueError("Source and oracle have mismatched corpus identity")
                if oracle_state.config_hash != state.config_hash:
                    raise ValueError("Source and oracle have mismatched configuration")

                oracle_trace = oracle_state.trace

            source_trace = deepcopy(state.trace)
            if alternative_aggregations is not None:
                expected = state.comparisons.get(claim_id)
                if expected is None or alternative_aggregations != expected:
                    raise ValueError("Alternative aggregations must match server-held comparisons")
                alternative_aggregations = deepcopy(expected)

        corpus_fp = source_trace.resolved_config.get("data_identity", {}).get("corpus", {}).get("sha256")

        researcher_annotation = {
            "failure_category": failure_category,
            "researcher_note": researcher_note,
            "human_label": human_label,
        }

        path = save_failure_case_file(
            trace=source_trace,
            note=researcher_note,
            claim_id=claim_id,
            oracle_trace=oracle_trace,
            researcher_annotation=researcher_annotation,
            corpus_fingerprint=corpus_fp,
            alternative_aggregations=alternative_aggregations,
            out_dir=out_dir or self.failure_dir,
        )

        return {
            "failure_case_id": f"fc_{path.stem}",
            "file_path": str(path),
            "status": "saved",
        }



# Singleton instance for application lifecycle
_default_service: NulliusService | None = None


def get_service() -> NulliusService:
    global _default_service
    if _default_service is None:
        _default_service = NulliusService()
    return _default_service


__all__ = [
    "NulliusService",
    "QuickClaimResult",
    "QuickVerificationResult",
    "get_service",
]
