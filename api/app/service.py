"""
Analysis service — the glue between the preprocessing pipeline and the ML model.

Separated from the FastAPI route handlers so it can be unit-tested in
isolation, reused for batch jobs, and swapped for a mock during API tests.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ecg_preprocessor import ECGPreprocessor, SignalQualityError
from ecg_model.model import CONDITION_NAMES

from .model_loader import get_model_bundle
from .schemas import (
    AnalysisResponse,
    ConditionResult,
    LeadQualityReport,
    confidence_to_tier,
)


# Conditions that are considered "normal" when triggered.
# Normal Sinus Rhythm firing is reassuring; all others being positive is not.
NORMAL_CONDITIONS = {"Normal Sinus Rhythm"}


def build_overall_interpretation(
    results: list[ConditionResult],
    sqi: float,
) -> str:
    """Produce a plain-English summary for the AnalysisResponse."""
    high_or_likely = [r for r in results if r.risk_tier in ("high", "likely")
                      and r.condition not in NORMAL_CONDITIONS]
    possible = [r for r in results if r.risk_tier == "possible"
                and r.condition not in NORMAL_CONDITIONS]

    if sqi < 0.75:
        return (
            "Signal quality is borderline; interpret findings cautiously and "
            "consider re-recording before relying on this report."
        )
    if high_or_likely:
        names = ", ".join(r.condition for r in high_or_likely)
        return f"Possible cardiac abnormality detected ({names}). Please consult a physician."
    if possible:
        return (
            "Minor irregularities were observed that may warrant follow-up. "
            "Monitoring is recommended."
        )
    # Did Normal Sinus Rhythm score high?
    nsr = next((r for r in results if r.condition == "Normal Sinus Rhythm"), None)
    if nsr and nsr.risk_tier in ("likely", "high"):
        return "No significant abnormality detected; rhythm appears normal."
    return "No significant abnormality detected among the conditions screened."


class AnalysisService:
    """Holds the preprocessing pipeline (reusable) and runs end-to-end analysis."""

    def __init__(self):
        # Stateless — safe to instantiate once and reuse across requests.
        self.preprocessor = ECGPreprocessor()

    def analyze_array(
        self,
        signal: np.ndarray,
        sample_rate: int = 500,
        patient_id: str | None = None,
    ) -> tuple[AnalysisResponse, dict[str, Any]]:
        """Run preprocessing + model inference on a raw (L, N) numpy array.

        Returns
        -------
        response : AnalysisResponse   — the Pydantic model for JSON return
        raw_dict : dict               — same data as plain dict, for storage
        """
        pp_result = self.preprocessor.run(signal, sample_rate=sample_rate)

        bundle = get_model_bundle()
        probs = bundle.predict(pp_result.signal)

        condition_results: list[ConditionResult] = []
        for i, name in enumerate(CONDITION_NAMES):
            conf = float(probs[i])
            tier, label = confidence_to_tier(conf)
            condition_results.append(ConditionResult(
                condition=name,
                confidence=conf,
                risk_tier=tier,
                tier_label=label,
                normal_range=(name in NORMAL_CONDITIONS),
            ))

        per_lead_quality = [
            LeadQualityReport(lead=lq.lead, sqi=float(lq.sqi), flags=list(lq.flags))
            for lq in pp_result.quality.per_lead
        ]

        report_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        overall = build_overall_interpretation(condition_results, pp_result.quality.overall)

        response = AnalysisResponse(
            report_id=report_id,
            timestamp=timestamp,
            signal_quality=float(pp_result.quality.overall),
            per_lead_quality=per_lead_quality,
            results=condition_results,
            overall_interpretation=overall,
        )

        # The stored payload includes patient_id for history lookups.
        raw = response.model_dump(mode="json")
        if patient_id:
            raw["patient_id"] = patient_id
        return response, raw

    def analyze_csv_bytes(
        self,
        csv_bytes: bytes,
        sample_rate: int = 500,
        patient_id: str | None = None,
    ) -> tuple[AnalysisResponse, dict[str, Any]]:
        """Parse a CSV upload to a numpy array, then delegate to analyze_array."""
        import io
        import pandas as pd
        from ecg_preprocessor.constants import LEAD_ORDER, LEAD_INDEX

        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as exc:
            raise ValueError(f"Failed to parse CSV: {exc}") from exc

        if df.empty:
            raise ValueError("CSV contains no rows")

        col_map = {c.strip().upper(): c for c in df.columns}
        canonical_upper = {name.upper(): name for name in LEAD_ORDER}
        matched = set(col_map) & set(canonical_upper)
        if not matched:
            raise ValueError(
                f"CSV has no recognizable lead columns. Found: {list(df.columns)}. "
                f"Expected any of: {list(LEAD_ORDER)}"
            )

        signal = np.zeros((len(LEAD_ORDER), len(df)), dtype=np.float32)
        for upper in matched:
            canonical = canonical_upper[upper]
            idx = LEAD_INDEX[canonical]
            col = col_map[upper]
            signal[idx] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        return self.analyze_array(signal, sample_rate=sample_rate, patient_id=patient_id)

    def analyze_json_leads(
        self,
        leads: list[dict[str, Any]],
        patient_id: str | None = None,
    ) -> tuple[AnalysisResponse, dict[str, Any]]:
        """Combine per-lead JSON payloads and analyze."""
        from ecg_preprocessor.io import combine_lead_payloads
        signal, fs = combine_lead_payloads(leads, strict=False)
        return self.analyze_array(signal, sample_rate=fs, patient_id=patient_id)


_service: AnalysisService | None = None


def get_service() -> AnalysisService:
    global _service
    if _service is None:
        _service = AnalysisService()
    return _service
