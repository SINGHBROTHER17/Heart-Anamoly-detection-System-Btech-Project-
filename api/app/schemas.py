"""
Pydantic schemas for request validation and response serialization.

All request inputs go through these models so FastAPI auto-generates
OpenAPI docs and rejects malformed payloads before our code sees them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Risk tier mapping (matches the spec exactly)
# ---------------------------------------------------------------------------

RiskTier = Literal["none", "possible", "likely", "high"]


def confidence_to_tier(confidence: float) -> tuple[RiskTier, str]:
    """Map a confidence score [0, 1] to a risk tier and user-facing label.

    Thresholds match the Phase 4 spec:
      < 0.30         -> none
      0.30 <= .. <0.60 -> possible
      0.60 <= .. <0.85 -> likely
      >= 0.85        -> high
    """
    if confidence < 0.30:
        return "none", "No significant finding detected"
    if confidence < 0.60:
        return "possible", "Possible finding — monitoring recommended"
    if confidence < 0.85:
        return "likely", "Likely finding — please consult a physician"
    return "high", "Strong indicator — seek medical attention promptly"


# ---------------------------------------------------------------------------
# Input: JSON signal payload
# ---------------------------------------------------------------------------

class LeadPayload(BaseModel):
    """One per-lead JSON payload, matching the acquisition device format."""

    lead_name: str = Field(..., description="Lead name (I, II, III, aVR, aVL, aVF, V1..V6)")
    samples: list[float] = Field(..., min_length=500, description="Raw samples")
    sample_rate: int = Field(500, ge=100, le=2000, description="Hz")

    @field_validator("lead_name")
    @classmethod
    def _validate_lead_name(cls, v: str) -> str:
        valid = {"I", "II", "III", "AVR", "AVL", "AVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"}
        if v.strip().upper().replace(" ", "") not in valid:
            raise ValueError(f"Unknown lead '{v}'. Must be one of: {sorted(valid)}")
        return v.strip()


class AnalyzeJsonRequest(BaseModel):
    """Request body when submitting multiple leads as JSON (sequential recording)."""

    leads: list[LeadPayload] = Field(..., min_length=1, max_length=12)
    patient_id: Optional[str] = Field(None, description="Optional external patient ID")


# ---------------------------------------------------------------------------
# Response: per-condition result
# ---------------------------------------------------------------------------

class ConditionResult(BaseModel):
    condition: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_tier: RiskTier
    tier_label: str
    normal_range: bool = Field(..., description="True if this is within physiological normal")


class LeadQualityReport(BaseModel):
    """Per-lead SQI breakdown returned alongside the main result."""

    lead: str
    sqi: float = Field(..., ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list,
                             description="e.g. ['flatline'], ['clipping'], ['noisy']")


class AnalysisResponse(BaseModel):
    report_id: str
    timestamp: datetime
    signal_quality: float = Field(..., ge=0.0, le=1.0)
    per_lead_quality: list[LeadQualityReport]
    results: list[ConditionResult]
    overall_interpretation: str
    disclaimer: str = (
        "This is a screening tool only. Not a substitute for medical diagnosis. "
        "If you have symptoms or concerns, consult a qualified physician."
    )


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    per_lead_sqi: Optional[dict[str, float]] = None


# ---------------------------------------------------------------------------
# Feedback endpoint
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    report_id: str
    clinician_id: Optional[str] = None
    correct_conditions: list[str] = Field(default_factory=list,
        description="List of condition names the clinician agrees with")
    incorrect_conditions: list[str] = Field(default_factory=list,
        description="List of predicted conditions that were false positives")
    missed_conditions: list[str] = Field(default_factory=list,
        description="Conditions the model should have flagged but didn't")
    notes: Optional[str] = Field(None, max_length=2000)


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    report_id: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    model_loaded: bool
    uptime_seconds: float
    timestamp: datetime
