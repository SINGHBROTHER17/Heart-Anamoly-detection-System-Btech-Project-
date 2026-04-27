"""
FastAPI app — ECG Anomaly Detection Cloud API.

Endpoints
---------
  POST /analyze        : Accepts CSV upload or JSON payload, returns analysis
  GET  /report/{id}    : Retrieve stored report
  GET  /reports        : List recent reports
  POST /feedback       : Accept clinician feedback for future fine-tuning
  GET  /health         : Health check for load balancers / monitoring
  GET  /docs           : Auto-generated OpenAPI docs (built-in FastAPI)
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from ecg_preprocessor import InvalidInputError, SignalQualityError

from . import __version__
from .logging_config import configure_logging, get_logger, new_request_id
from .model_loader import load_model
from .rate_limit import limiter
from .schemas import (
    AnalysisResponse,
    AnalyzeJsonRequest,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from .service import get_service
from .storage import SQLiteStorage, get_storage


# Module-level startup timestamp for uptime reporting.
_startup_ts = time.time()
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level="INFO")
    log.info("api_starting", version=__version__)
    try:
        bundle = load_model()
        log.info("model_loaded", checkpoint=bundle.checkpoint_path,
                 device=str(bundle.device))
    except FileNotFoundError as exc:
        # Don't crash — serve /health reporting degraded state so ops can
        # see the problem in monitoring instead of a crash loop.
        log.error("model_load_failed", error=str(exc))
    yield
    log.info("api_stopping")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ECG Anomaly Detection API",
    description=(
        "REST API for screening 12-lead ECG recordings for 10 cardiac "
        "conditions with calibrated confidence scores. "
        "**Screening tool only — not a diagnostic device.**"
    ),
    version=__version__,
    lifespan=lifespan,
)

# Rate limiting state attached to the app so slowapi middleware can find it.
app.state.limiter = limiter


# CORS — allow the mobile web app origin. Override via CORS_ALLOW_ORIGINS env.
import os
cors_origins = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:4173"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: correlation ID + access logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    rid = new_request_id()
    start = time.perf_counter()
    log.info(
        "request_in",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "-",
    )
    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.exception("request_failed",
                      path=request.url.path, elapsed_ms=round(elapsed_ms, 2),
                      error=str(exc))
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    log.info(
        "request_out",
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=round(elapsed_ms, 2),
    )
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limit_exceeded",
                 "detail": "Too many requests. Please slow down."},
    )


@app.exception_handler(SignalQualityError)
async def signal_quality_handler(request: Request, exc: SignalQualityError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "signal_quality_too_low",
            "detail": str(exc),
            "per_lead_sqi": {k: round(v, 3) for k, v in exc.per_lead.items()},
        },
    )


@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request: Request, exc: InvalidInputError):
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_input", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Liveness probe. Reports model_loaded=False if the checkpoint was missing."""
    from .model_loader import _bundle
    return HealthResponse(
        status="ok" if _bundle is not None else "degraded",
        version=__version__,
        model_loaded=(_bundle is not None),
        uptime_seconds=round(time.time() - _startup_ts, 1),
        timestamp=datetime.now(timezone.utc),
    )


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    tags=["analysis"],
)
@limiter.limit("10/minute")
async def analyze(
    request: Request,
    file: UploadFile = File(..., description="12-lead ECG CSV"),
    sample_rate: int = Form(default=500, ge=100, le=2000),
    patient_id: Optional[str] = Form(default=None),
    storage: SQLiteStorage = Depends(get_storage),
):
    """Analyze an ECG recording from a **CSV file upload**.

    Content-Type: `multipart/form-data`
    Use `/analyze/json` instead when submitting JSON per-lead payloads.
    """
    service = get_service()

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:  # 50 MB cap
        raise HTTPException(status_code=413, detail="File too large (>50MB)")
    try:
        response, raw = service.analyze_csv_bytes(
            data, sample_rate=sample_rate, patient_id=patient_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    storage.save_report(response.report_id, raw,
                        patient_id=raw.get("patient_id"))
    log.info(
        "analysis_complete",
        report_id=response.report_id,
        sqi=round(response.signal_quality, 3),
        top_condition=max(response.results, key=lambda r: r.confidence).condition,
    )
    return response


@app.post(
    "/analyze/json",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    tags=["analysis"],
)
@limiter.limit("10/minute")
async def analyze_json(
    request: Request,
    payload: AnalyzeJsonRequest,
    storage: SQLiteStorage = Depends(get_storage),
):
    """Analyze an ECG recording from a **JSON per-lead payload**.

    Content-Type: `application/json`
    Use `/analyze` instead when submitting a CSV file.
    """
    service = get_service()
    leads_as_dicts = [lp.model_dump() for lp in payload.leads]
    response, raw = service.analyze_json_leads(
        leads_as_dicts, patient_id=payload.patient_id
    )
    storage.save_report(response.report_id, raw,
                        patient_id=raw.get("patient_id"))
    log.info(
        "analysis_complete",
        report_id=response.report_id,
        sqi=round(response.signal_quality, 3),
        top_condition=max(response.results, key=lambda r: r.confidence).condition,
    )
    return response


@app.get(
    "/report/{report_id}",
    response_model=AnalysisResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["reports"],
)
async def get_report(report_id: str, storage: SQLiteStorage = Depends(get_storage)):
    report = storage.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return report


@app.get("/reports", tags=["reports"])
async def list_reports(
    limit: int = 50,
    storage: SQLiteStorage = Depends(get_storage),
):
    """Returns the N most recent reports (for the history screen in the mobile app)."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    reports = storage.list_recent_reports(limit=limit)
    return {"count": len(reports), "reports": reports}


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
async def submit_feedback(
    feedback: FeedbackRequest,
    storage: SQLiteStorage = Depends(get_storage),
):
    # Ensure the referenced report exists.
    if storage.get_report(feedback.report_id) is None:
        raise HTTPException(status_code=404,
                            detail=f"Report {feedback.report_id} not found")

    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)
    payload = feedback.model_dump()
    payload["feedback_id"] = feedback_id
    payload["timestamp"] = timestamp.isoformat()
    storage.save_feedback(feedback_id, payload)

    log.info("feedback_received", feedback_id=feedback_id,
             report_id=feedback.report_id)
    return FeedbackResponse(
        status="recorded",
        feedback_id=feedback_id,
        report_id=feedback.report_id,
        timestamp=timestamp,
    )


@app.get("/", tags=["system"])
async def root():
    return {
        "name": "ECG Anomaly Detection API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
