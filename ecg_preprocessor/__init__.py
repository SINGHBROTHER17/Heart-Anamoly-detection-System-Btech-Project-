"""
ecg_preprocessor
================

Production-grade preprocessing pipeline that turns raw, sequentially recorded
ECG lead data into a clean, aligned, normalized (12, 5000) tensor ready for
model inference.

Public API
----------
    ECGPreprocessor          -- end-to-end pipeline (class)
    preprocess               -- one-shot convenience function
    load_csv / load_json     -- input adapters
    LEAD_ORDER               -- canonical 12-lead order
    SignalQualityError       -- raised when SQI < threshold
    PreprocessingError       -- generic preprocessing failure
"""

from .pipeline import ECGPreprocessor, preprocess
from .io import load_csv, load_json, combine_lead_payloads
from .constants import LEAD_ORDER, TARGET_FS, WINDOW_SAMPLES
from .exceptions import (
    SignalQualityError,
    PreprocessingError,
    InvalidInputError,
)

__all__ = [
    "ECGPreprocessor",
    "preprocess",
    "load_csv",
    "load_json",
    "combine_lead_payloads",
    "LEAD_ORDER",
    "TARGET_FS",
    "WINDOW_SAMPLES",
    "SignalQualityError",
    "PreprocessingError",
    "InvalidInputError",
]

__version__ = "0.1.0"
