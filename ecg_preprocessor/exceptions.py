"""Custom exception hierarchy for clear error surfaces."""

from __future__ import annotations


class PreprocessingError(Exception):
    """Base class for all preprocessing failures."""


class InvalidInputError(PreprocessingError):
    """Raised when raw input cannot be parsed or has invalid shape."""


class SignalQualityError(PreprocessingError):
    """Raised when a signal's SQI falls below the accept threshold.

    Attributes
    ----------
    sqi : float
        Overall signal quality score that triggered rejection.
    per_lead : dict[str, float]
        Per-lead SQI breakdown, useful for surfacing a "re-record lead X" UX.
    """

    def __init__(self, sqi: float, per_lead: dict[str, float], message: str | None = None):
        self.sqi = sqi
        self.per_lead = per_lead
        msg = message or (
            f"Signal quality too low to analyze (SQI={sqi:.2f}). "
            f"Please re-record leads with poor contact and try again."
        )
        super().__init__(msg)
