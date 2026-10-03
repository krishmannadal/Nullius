"""FastAPI router implementing all required endpoints for Nullius.

Endpoints:
  GET  /health          - Diagnostic runtime status, VRAM, and component info
  POST /analyze         - Standard retrieved pipeline verification
  POST /analyze/oracle  - Oracle verification with gold evidence substitution
  POST /verify/quick    - Tier 1 fast evidence-availability check
  POST /verify/full     - Tier 2 full verification (supports JSON or SSE streaming)
  POST /annotate        - Write human annotation record to local corpus
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.schemas import (
    AnalyzeOracleRequest,
    AnalyzeRequest,
    AnalyzeResponse,
    AnnotateRequest,
    AnnotateResponse,
    ErrorResponse,
    HealthResponse,
    VerifyFullRequest,
    VerifyQuickRequest,
    VerifyQuickResponse,
    ReaggregateRequest,
    ReaggregateResponse,
    SaveFailureCaseRequest,
    FailureCaseResponse,
)
from src.core.interfaces import ContractError
from src.core.registry import RegistryError
from src.service import NulliusService, get_service

router = APIRouter()


def _format_sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Events frame."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health and Diagnostics",
    description="Returns diagnostic state, device info, VRAM allocation, queue depth, and loaded components.",
)
async def health_endpoint(
    service: NulliusService = Depends(get_service),
) -> HealthResponse:
    diag = service.get_health()
    return HealthResponse(**diag)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid configuration or contract failure"},
        422: {"model": ErrorResponse, "description": "Validation error in request"},
        500: {"model": ErrorResponse, "description": "Internal pipeline execution error"},
    },
    summary="Analyze Response Text (Standard Retrieved Mode)",
    description=(
        "Executes the standard pipeline: decomposes text into claims, retrieves evidence, "
        "performs optional reranking, verifies pairwise (Claim x Evidence), and aggregates verdicts."
    ),
)
async def analyze_endpoint(
    req: AnalyzeRequest,
    service: NulliusService = Depends(get_service),
) -> AnalyzeResponse:
    try:
        trace = await service.analyze(
            req.text,
            config_path=req.config_path,
            overrides=req.overrides,
            retrieve_k=req.retrieve_k,
            check_contracts=req.check_contracts,
        )
        return AnalyzeResponse(**trace.to_dict())
    except (ContractError, RegistryError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pipeline contract error: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {exc}",
        ) from exc


@router.post(
    "/analyze/oracle",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid oracle request or contract failure"},
        404: {"model": ErrorResponse, "description": "Specified example_id not found"},
        422: {"model": ErrorResponse, "description": "Validation error or missing gold evidence"},
    },
    summary="Analyze in Oracle Mode (Gold Evidence Substitution)",
    description=(
        "Runs verification with retrieved evidence replaced by the example's annotated gold evidence. "
        "Strictly isolated from standard inference to prevent leakage."
    ),
)
async def analyze_oracle_endpoint(
    req: AnalyzeOracleRequest,
    service: NulliusService = Depends(get_service),
) -> AnalyzeResponse:
    try:
        trace = await service.analyze_oracle(
            example_id=req.example_id,
            response_text=req.response_text,
            config_path=req.config_path,
            overrides=req.overrides,
            retrieve_k=req.retrieve_k,
            check_contracts=req.check_contracts,
        )
        return AnalyzeResponse(**trace.to_dict())
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (ContractError, RegistryError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pipeline contract error: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Oracle execution failed: {exc}",
        ) from exc


@router.post(
    "/verify/quick",
    response_model=VerifyQuickResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid configuration or contract failure"},
        422: {"model": ErrorResponse, "description": "Validation error in request text"},
    },
    summary="Tier 1 Quick Availability Check",
    description=(
        "Decomposes text into claims and checks evidence availability via lightweight retrieval. "
        "Does NOT run heavy NLI models and NEVER asserts that a claim is false. "
        "Provides badge counts and per-claim checkability states."
    ),
)
async def verify_quick_endpoint(
    req: VerifyQuickRequest,
    service: NulliusService = Depends(get_service),
) -> VerifyQuickResponse:
    try:
        res = await service.verify_quick(
            req.text,
            config_path=req.config_path,
            overrides=req.overrides,
            evidence_floor_score=req.evidence_floor_score,
            retrieve_k=req.retrieve_k,
            check_contracts=req.check_contracts,
        )
        return VerifyQuickResponse(
            response_text=res.response_text,
            claims=[
                {
                    "claim_id": c.claim_id,
                    "claim_text": c.claim_text,
                    "source_span": c.source_span,
                    "status": c.status,
                    "n_evidence_found": c.n_evidence_found,
                    "top_evidence_score": c.top_evidence_score,
                    "top_evidence_text": c.top_evidence_text,
                    "top_evidence_source": c.top_evidence_source,
                }
                for c in res.claims
            ],
            counts=res.counts,
            timings=res.timings,
            config_hash=res.config_hash,
        )
    except (ContractError, RegistryError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Contract error: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick check failed: {exc}",
        ) from exc


@router.post(
    "/verify/full",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid configuration or contract failure"},
        422: {"model": ErrorResponse, "description": "Validation error in request text"},
    },
    summary="Tier 2 Full Pipeline Verification",
    description=(
        "Executes the complete verification pipeline. If stream=True or client accepts text/event-stream, "
        "streams claim verdicts progressively via Server-Sent Events (SSE)."
    ),
)
async def verify_full_endpoint(
    req: VerifyFullRequest,
    raw_request: Request,
    service: NulliusService = Depends(get_service),
) -> Any:
    # Check if SSE streaming is requested via flag or Accept header
    client_wants_sse = req.stream or "text/event-stream" in raw_request.headers.get("accept", "")

    if client_wants_sse:
        async def event_generator() -> AsyncIterator[str]:
            try:
                async for item in service.verify_full_stream(
                    req.text,
                    config_path=req.config_path,
                    overrides=req.overrides,
                    retrieve_k=req.retrieve_k,
                    check_contracts=req.check_contracts,
                ):
                    yield _format_sse_event(item["event"], item["data"])
            except Exception as exc:
                yield _format_sse_event("error", {"detail": str(exc), "type": type(exc).__name__})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Standard non-streaming execution
    try:
        trace = await service.analyze(
            req.text,
            config_path=req.config_path,
            overrides=req.overrides,
            retrieve_k=req.retrieve_k,
            check_contracts=req.check_contracts,
        )
        return AnalyzeResponse(**trace.to_dict())
    except (ContractError, RegistryError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pipeline contract error: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full verification failed: {exc}",
        ) from exc


@router.post(
    "/annotate",
    response_model=AnnotateResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Missing required annotation fields"},
    },
    summary="Save Human Annotation Record",
    description="Appends a structured human annotation record to the local annotation corpus.",
)
async def annotate_endpoint(
    req: AnnotateRequest,
    service: NulliusService = Depends(get_service),
) -> AnnotateResponse:
    try:
        saved = service.save_annotation(req.model_dump())
        return AnnotateResponse(status="saved", record=saved)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist annotation: {exc}",
        ) from exc


@router.post(
    "/reaggregate",
    response_model=ReaggregateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid aggregator configuration or contract failure"},
        422: {"model": ErrorResponse, "description": "Validation error in request"},
    },
    summary="Compare Rules on a Stored Inspection",
    description="Runs aggregation rules on identical server-stored pairwise scores. Runs expire after one hour or a server restart; client-supplied scores are never accepted.",
)
async def reaggregate_endpoint(
    req: ReaggregateRequest,
    service: NulliusService = Depends(get_service),
) -> ReaggregateResponse:
    try:
        resp_dict = service.reaggregate(
            run_id=req.run_id,
            target_aggregators=req.target_aggregators,
            aggregator_configs=req.aggregator_configs,
        )
        return ReaggregateResponse(**resp_dict)
    except (ContractError, RegistryError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Reaggregation contract error: {exc}",
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reaggregation failed: {exc}",
        ) from exc


@router.post(
    "/failure-cases",
    response_model=FailureCaseResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid relationship between runs or bad configuration"},
        404: {"model": ErrorResponse, "description": "Run ID, claim ID, or oracle run ID not found or expired"},
        422: {"model": ErrorResponse, "description": "Validation error or invalid failure category"},
    },
    summary="Save Failure Case",
    description="Persists a structured failure case linked to server-side execution state.",
)
async def save_failure_case_endpoint(
    req: SaveFailureCaseRequest,
    service: NulliusService = Depends(get_service),
) -> FailureCaseResponse:
    try:
        res = service.save_failure_case(
            run_id=req.run_id,
            claim_id=req.claim_id,
            failure_category=req.failure_category,
            researcher_note=req.researcher_note,
            human_label=req.human_label,
            oracle_run_id=req.oracle_run_id,
            alternative_aggregations=req.alternative_aggregations,
        )
        return FailureCaseResponse(**res)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        err_msg = str(exc)
        if "mismatched" in err_msg.lower() or "mode" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=err_msg,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=err_msg,
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist failure case: {exc}",
        ) from exc
