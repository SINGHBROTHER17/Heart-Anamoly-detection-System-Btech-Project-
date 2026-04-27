"""
End-to-end preprocessing pipeline.

Call `ECGPreprocessor().run(signal_or_path)` and get back a ready-to-feed
(12, 5000) tensor plus a full SignalQuality report.

The pipeline order is deliberate — getting it wrong degrades quality:

    1.  Load / assemble raw (12, N) array.
    2.  Bandpass (0.5-40 Hz).            kill drift and EMG
    3.  Notch (50/60 Hz).                 kill mains hum
    4.  Baseline wander removal.          kill slow drift the HPF missed
    5.  SQI — after cleanup, before alignment. We want to fail fast if
        the signal is garbage, without spending effort on alignment.
    6.  R-peak-based inter-lead alignment. Compensates for sequential
        acquisition banks.
    7.  Segment to fixed 10 s window.
    8.  Per-lead z-score normalization. Always last — everything prior
        uses absolute amplitudes (SQI, peak detection).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from .constants import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    BANDPASS_ORDER,
    LEAD_ORDER,
    POWERLINE_HZ,
    SQI_THRESHOLD,
    TARGET_FS,
    WINDOW_SAMPLES,
)
from .exceptions import InvalidInputError, SignalQualityError
from .filters import (
    bandpass_filter,
    notch_filter,
    remove_baseline_wander,
    zscore_normalize,
)
from .alignment import align_leads
from .io import load_csv, combine_lead_payloads, load_json
from .quality import SignalQuality, compute_sqi
from .segmentation import segment_fixed_window

PathLike = Union[str, Path]


@dataclass
class PreprocessingResult:
    """Bundle of everything the pipeline produces."""
    signal: np.ndarray  # (12, 5000), z-scored, aligned, normalized
    quality: SignalQuality  # per-lead + overall SQI
    r_peaks_ref: np.ndarray  # R-peaks in the reference lead (Lead II), in samples
    shifts: list[int]  # per-lead alignment shift, in samples
    fs: int


class ECGPreprocessor:
    """Reusable preprocessing pipeline. Thread-safe (no shared mutable state)."""

    def __init__(
        self,
        target_fs: int = TARGET_FS,
        window_samples: int = WINDOW_SAMPLES,
        bandpass_low: float = BANDPASS_LOW,
        bandpass_high: float = BANDPASS_HIGH,
        bandpass_order: int = BANDPASS_ORDER,
        powerline_hz: float = POWERLINE_HZ,
        sqi_threshold: float = SQI_THRESHOLD,
        reference_lead: str = "II",
        max_shift_ms: float = 200.0,
        window_mode: str = "center",
    ):
        self.target_fs = target_fs
        self.window_samples = window_samples
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.bandpass_order = bandpass_order
        self.powerline_hz = powerline_hz
        self.sqi_threshold = sqi_threshold
        self.reference_lead = reference_lead
        self.max_shift_ms = max_shift_ms
        self.window_mode = window_mode

    # ---- Public API ------------------------------------------------------

    def run(
        self,
        source: Union[np.ndarray, PathLike, Iterable[Union[PathLike, dict]]],
        sample_rate: int | None = None,
    ) -> PreprocessingResult:
        """Run the full pipeline.

        `source` may be:
            * a (12, N) or (L, N) np.ndarray of raw samples
            * a path to a CSV file
            * an iterable of per-lead JSON payloads (paths / strings / dicts)
        """
        raw, fs = self._load(source, sample_rate)

        # 1. Bandpass
        cleaned = bandpass_filter(
            raw,
            fs=fs,
            low=self.bandpass_low,
            high=self.bandpass_high,
            order=self.bandpass_order,
        )
        # 2. Notch
        cleaned = notch_filter(cleaned, fs=fs, freq=self.powerline_hz)
        # 3. Baseline wander
        cleaned = remove_baseline_wander(cleaned, fs=fs)

        # 4. Quality check — fail fast on bad electrode contact / clipping.
        quality = compute_sqi(cleaned, fs=fs, lead_names=list(LEAD_ORDER))
        if quality.overall < self.sqi_threshold:
            raise SignalQualityError(
                sqi=quality.overall,
                per_lead={lq.lead: lq.sqi for lq in quality.per_lead},
            )

        # 5. Inter-lead alignment
        aligned, shifts = align_leads(
            cleaned,
            fs=fs,
            reference_lead=self.reference_lead,
            max_shift_ms=self.max_shift_ms,
        )

        # 6. Segment
        windowed = segment_fixed_window(
            aligned, window=self.window_samples, mode=self.window_mode
        )

        # 7. Normalize
        normalized = zscore_normalize(windowed)

        # R-peaks for visualization (detected on the reference lead).
        from .peaks import detect_r_peaks
        from .constants import LEAD_INDEX
        ref_idx = LEAD_INDEX.get(self.reference_lead, 1)
        # Use the windowed (pre-normalization) reference for R-peak reporting
        # so the indices match the visible ECG trace.
        ref_peaks = detect_r_peaks(windowed[ref_idx], fs=fs)

        return PreprocessingResult(
            signal=normalized,
            quality=quality,
            r_peaks_ref=ref_peaks,
            shifts=shifts,
            fs=fs,
        )

    # ---- Loaders ---------------------------------------------------------

    def _load(
        self,
        source: Union[np.ndarray, PathLike, Iterable],
        sample_rate: int | None,
    ) -> tuple[np.ndarray, int]:
        if isinstance(source, np.ndarray):
            arr = source.astype(np.float32, copy=False)
            if arr.ndim != 2:
                raise InvalidInputError(
                    f"Array input must be 2-D (leads, samples); got shape {arr.shape}"
                )
            # Pad rows if caller gave fewer than 12 leads, to keep pipeline shape stable.
            if arr.shape[0] < len(LEAD_ORDER):
                padded = np.zeros((len(LEAD_ORDER), arr.shape[1]), dtype=np.float32)
                padded[: arr.shape[0]] = arr
                arr = padded
            elif arr.shape[0] > len(LEAD_ORDER):
                arr = arr[: len(LEAD_ORDER)]
            fs = int(sample_rate or self.target_fs)
            if fs != self.target_fs:
                from .io import _resample
                arr = _resample(arr, fs, self.target_fs)
                fs = self.target_fs
            return arr, fs

        if isinstance(source, (str, Path)):
            p = Path(str(source))
            if p.suffix.lower() == ".csv":
                return load_csv(p, sample_rate=sample_rate or self.target_fs)
            if p.suffix.lower() == ".json":
                # Single-lead JSON — wrap it.
                samples, fs, lead_name = load_json(p)
                arr = np.zeros((len(LEAD_ORDER), samples.size), dtype=np.float32)
                from .constants import LEAD_INDEX
                arr[LEAD_INDEX[lead_name]] = samples
                return arr, fs
            raise InvalidInputError(f"Unsupported file extension: {p.suffix}")

        # Fallback: iterable of per-lead JSON payloads.
        try:
            payloads = list(source)  # type: ignore[arg-type]
        except TypeError as exc:
            raise InvalidInputError(
                f"Unsupported source type: {type(source).__name__}"
            ) from exc
        return combine_lead_payloads(payloads)


# ---------------------------------------------------------------------------
# One-shot convenience
# ---------------------------------------------------------------------------

def preprocess(
    source: Union[np.ndarray, PathLike, Iterable],
    sample_rate: int | None = None,
    **kwargs,
) -> PreprocessingResult:
    """Shorthand for `ECGPreprocessor(**kwargs).run(source, sample_rate)`."""
    return ECGPreprocessor(**kwargs).run(source, sample_rate=sample_rate)
