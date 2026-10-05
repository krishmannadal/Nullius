"""FastAPI application factory for Nullius.

Provides:
  - Lifespan model warming and service initialization
  - CORS middleware configured for localhost, Streamlit, and browser extension
  - Structured error handlers
  - Route registration
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.core.interfaces import ContractError
from src.core.registry import RegistryError
from src.core.types import SCHEMA_VERSION
from src.service import NulliusService, get_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app(
    default_config_path: str = "configs/mini.yaml",
    title: str = "Nullius Inspection Harness Backend",
    service: NulliusService | None = None,
    extension_ids: tuple[str, ...] = (),
) -> FastAPI:
    app = FastAPI(
        title=title,
        description=(
            "Nullius is a claim-level hallucination-detection research instrument / inspection harness. "
            "Not a validated production detector. Strictly maintains pairwise verification with decoupled aggregation."
        ),
        version=SCHEMA_VERSION,
        lifespan=lifespan,
    )
    app.state.service = service or NulliusService(default_config_path=default_config_path)
    app.dependency_overrides[get_service] = lambda: app.state.service
    extension_ids = extension_ids or tuple(filter(None, os.environ.get("NULLIUS_EXTENSION_IDS", "").split(",")))

    # Local development and browser extension origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8501",  # Streamlit default
            "http://127.0.0.1:8501",
            *[f"chrome-extension://{identity}" for identity in extension_ids],
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ContractError)
    async def contract_error_handler(request: Request, exc: ContractError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "ContractError",
                "detail": str(exc),
                "error_type": "PipelineContractViolation",
            },
        )

    @app.exception_handler(RegistryError)
    async def registry_error_handler(request: Request, exc: RegistryError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "RegistryError",
                "detail": str(exc),
                "error_type": "ComponentLookupFailure",
            },
        )

    app.include_router(router)
    return app


app = create_app()

__all__ = ["app", "create_app"]
