"""Pydantic schemas for the Nullius FastAPI boundary.

ADR-006: Frozen dataclasses govern core and internal execution contracts; Pydantic
models are used strictly at the API boundary for request validation and OpenAPI schemas.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.trace_io import ALLOWED_FAILURE_CATEGORIES


class SourceSpanSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int = Field(..., ge=0, description="0-based start character offset in original response")
    end: int = Field(..., ge=0, description="Half-open end character offset in original response")


class ClaimSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Stable content-derived claim id")
    response_id: str = Field(..., description="Id of the response this claim came from")
    text: str = Field(..., description="Atomic claim assertion text")
    source_span: SourceSpanSchema | None = Field(None, description="Span in the original response")
    extractor_name: str = Field(..., description="Name of the extractor that produced this claim")
    extractor_meta: dict[str, Any] = Field(default_factory=dict, description="Metadata from extraction")


class EvidenceSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Corpus identity ev::<doc_id>::<sent_id>")
    doc_id: str = Field(..., description="Document identifier")
    sent_id: int = Field(..., ge=0, description="0-based sentence index within document")
    text: str = Field(..., description="Sentence text")
    score: float = Field(..., description="Retriever or reranker score")
    retriever_name: str = Field(..., description="Component that scored this evidence")
    rank: int = Field(..., ge=1, description="1-based rank in candidate list")
    is_gold: bool | None = Field(None, description="Whether this matches gold annotation (or None if unannotated)")
    retriever_meta: dict[str, Any] = Field(default_factory=dict, description="Per-arm scores and fusion metadata")


class EvidenceVerdictSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(..., description="Target claim id")
    evidence_id: str = Field(..., description="Target evidence id")
    p_entail: float | None = Field(None, description="NLI entailment probability")
    p_contra: float | None = Field(None, description="NLI contradiction probability")
    p_neutral: float | None = Field(None, description="NLI neutral probability")
    similarity: float | None = Field(None, description="Cosine similarity in [-1, 1]")
    verifier_name: str = Field(..., description="Verifier component name")
    latency_ms: float = Field(..., ge=0, description="Wall-clock inference latency in ms")
    evidence_rank: int | None = Field(None, ge=1, description="Denormalised retrieval rank")
    evidence_score: float | None = Field(None, description="Denormalised retrieval score")


class ClaimVerdictSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(..., description="Target claim id")
    label: str = Field(..., description="Label enum: Supported, Contradicted, Insufficient, or Abstain")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregator confidence score")
    abstained: bool = Field(..., description="True if aggregator opted to abstain")
    per_evidence: list[EvidenceVerdictSchema] = Field(..., description="Per-pair verdicts that produced this decision")
    aggregator_name: str = Field(..., description="Aggregator component name")
    aggregation_trace: dict[str, Any] = Field(..., description="Explanation, rule, and decisive evidence ids")


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = Field("ok", description="Service health status")
    schema_version: str = Field(..., description="Nullius core schema version")
    config_hash: str = Field(..., description="Hash of the active resolved config")
    harness_kind: str = Field("debug", description="debug | eval")
    device: str = Field(..., description="Computation device (e.g. cuda:0 or cpu)")
    cuda_available: bool = Field(..., description="Whether CUDA is available to PyTorch")
    vram_allocated_mb: float | None = Field(None, description="Current CUDA VRAM allocation in MB")
    queue_depth: int = Field(0, description="Number of requests currently queued or processing")
    components: dict[str, Any] = Field(default_factory=dict, description="Active registered components")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Response text to extract claims from and verify")
    config_path: str | None = Field(None, description="Optional path to config file relative to project root")
    overrides: list[str] | None = Field(None, description="Config overrides in dot-notation e.g. ['pipeline.k=3']")
    retrieve_k: int | None = Field(None, ge=1, description="Candidate retrieval pool size before reranking")
    check_contracts: bool = Field(True, description="Enforce stage-boundary runtime contract checks")


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Sortable run identifier")
    config_hash: str = Field(..., description="Hash of the resolved configuration")
    git_sha: str | None = Field(None, description="Git commit SHA or null")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    response_text: str = Field(..., description="Original input text")
    mode: str = Field("retrieved", description="'retrieved' or 'oracle'")
    claims: list[ClaimSchema] = Field(..., description="Extracted atomic claims")
    evidence_by_claim: dict[str, list[EvidenceSchema]] = Field(..., description="Ranked evidence per claim")
    verdicts: list[ClaimVerdictSchema] = Field(..., description="Aggregated decision per claim")
    timings: dict[str, float] = Field(..., description="Stage latency totals in ms")
    resolved_config: dict[str, Any] = Field(..., description="Resolved config and component details")
    schema_version: str = Field(..., description="Schema version of trace")


class AnalyzeOracleRequest(BaseModel):
    example_id: str | None = Field(None, description="Example ID from loaded dataset (e.g. 'fever-dev-5688')")
    response_text: str | None = Field(None, description="Response text matching an example in the dataset")
    config_path: str | None = Field(None, description="Optional path to config file")
    overrides: list[str] | None = Field(None, description="Config overrides in dot-notation")
    retrieve_k: int | None = Field(None, ge=1, description="Candidate retrieval pool size")
    check_contracts: bool = Field(True, description="Enforce stage-boundary runtime contract checks")


class QuickClaimResultSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(..., description="Claim identifier")
    claim_text: str = Field(..., description="Claim text")
    source_span: dict[str, int] | None = Field(None, description="Start/end offsets in response")
    status: str = Field(..., description="Tier 1 state: 'likely-checkable', 'no-evidence-found', 'pending'")
    n_evidence_found: int = Field(..., description="Count of candidate evidence items found above floor")
    top_evidence_score: float | None = Field(None, description="Score of highest-ranked evidence item")
    top_evidence_text: str | None = Field(None, description="Preview of highest-ranked evidence sentence")
    top_evidence_source: str | None = Field(None, description="doc_id:sent_id of top hit")


class VerifyQuickRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Response text to decompose and check evidence availability")
    config_path: str | None = Field(None, description="Optional path to config file")
    overrides: list[str] | None = Field(None, description="Config overrides in dot-notation")
    evidence_floor_score: float = Field(0.0, description="Minimum retriever score to consider checkable")
    retrieve_k: int = Field(5, ge=1, description="Number of candidate evidence sentences to retrieve")
    check_contracts: bool = Field(True, description="Enforce stage-boundary runtime contract checks")


class VerifyQuickResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    response_text: str = Field(..., description="Original input text")
    claims: list[QuickClaimResultSchema] = Field(..., description="Per-claim Tier 1 checkability results")
    counts: dict[str, int] = Field(..., description="Summary counts by state for badge rendering")
    timings: dict[str, float] = Field(..., description="Per-stage latency breakdown")
    config_hash: str = Field(..., description="Hash of resolved configuration")


class VerifyFullRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Response text to verify with full pipeline")
    config_path: str | None = Field(None, description="Optional path to config file")
    overrides: list[str] | None = Field(None, description="Config overrides in dot-notation")
    retrieve_k: int | None = Field(None, ge=1, description="Candidate retrieval pool size")
    check_contracts: bool = Field(True, description="Enforce stage-boundary runtime contract checks")
    stream: bool = Field(False, description="Stream results progressively via Server-Sent Events (SSE)")


class AnnotateRequest(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    site: str = Field("direct_api", description="Originating site or context (e.g. 'claude.ai', 'chatgpt.com')")
    model_name: str | None = Field(None, description="Name of LLM generating the original response")
    response_text: str = Field(..., min_length=1, description="Original assistant response text")
    claim_text: str = Field(..., min_length=1, description="Specific claim text being annotated")
    claim_span: dict[str, int] | None = Field(None, description="Character span [start, end) in response")
    extractor_name: str | None = Field(None, description="Extractor used to produce the claim")
    retrieved_evidence: list[dict[str, Any]] = Field(default_factory=list, description="Evidence items shown to annotator")
    system_verdict: str | None = Field(None, description="Pipeline's verdict label")
    system_confidence: float | None = Field(None, description="Pipeline's confidence score")
    human_label: str = Field(..., description="Human annotator label: agree/disagree/unclear or taxonomy label")
    human_note: str = Field("", description="Annotator notes or commentary")
    config_hash: str | None = Field(None, description="Hash of pipeline configuration that generated verdict")
    decomposition_disagreed: bool = Field(False, description="True if annotator disagreed with claim extraction itself")


class AnnotateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = Field("saved", description="Status of annotation write")
    record: dict[str, Any] = Field(..., description="Full saved annotation record")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: str = Field(..., description="Short error name or classification")
    detail: str = Field(..., description="Explanatory detail for the failure")
    error_type: str = Field(..., description="Exception class or category")


class ReaggregateRequest(BaseModel):
    run_id: str = Field(..., description="The source execution run ID")
    target_aggregators: list[str] = Field(..., description="List of aggregators to compare")
    aggregator_configs: dict[str, dict[str, Any]] = Field(
        default_factory=dict, 
        description="Optional config per aggregator, keyed by aggregator name"
    )


class ReaggregateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    run_id: str
    config_hash: str
    git_sha: str | None
    
    # Nested mapping: claim_id -> aggregator_name -> ClaimVerdictSchema
    comparisons: dict[str, dict[str, ClaimVerdictSchema]] = Field(
        ..., description="Comparison results grouped by claim ID and aggregator name"
    )


class SaveFailureCaseRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., min_length=1, description="Source execution run ID")
    claim_id: str = Field(..., min_length=1, description="Claim identifier to flag")
    failure_category: str = Field(..., description="Failure category classification")
    researcher_note: str = Field(..., min_length=1, description="Explanatory researcher note")
    human_label: str | None = Field(None, description="Optional human ground-truth label")
    oracle_run_id: str | None = Field(None, description="Optional oracle run ID")
    alternative_aggregations: dict[str, Any] | None = Field(
        None, description="Optional snapshot of reaggregation comparisons"
    )

    @field_validator("failure_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in ALLOWED_FAILURE_CATEGORIES:
            raise ValueError(
                f"Invalid failure category {v!r}. Must be one of {sorted(ALLOWED_FAILURE_CATEGORIES)}"
            )
        return v


class FailureCaseResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    failure_case_id: str = Field(..., description="Unique failure case identifier")
    file_path: str = Field(..., description="Path to persisted failure case JSON file")
    status: str = Field("saved", description="Status of persistence")


