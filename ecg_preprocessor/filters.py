"""
Digital filtering: bandpass, notch, baseline wander removal, normalization.

Design notes
------------
*   All IIR filters are applied with `sosfiltfilt` (zero-phase, no group delay),
    which is essential for ECG: any phase distortion smears QRS morphology and
    ruins downstream ST-segment measurements.
*   Bandpass is designed in SOS form directly (avoids the numerical instability
    you get with high-order `ba`).
*   Baseline wander uses the clinical double-median-filter method
    (200 ms then 600 ms kernel) — it tracks slow drift without introducing the
    wavelet reconstruction artifacts you sometimes see on T-waves.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, medfilt, sosfiltfilt, tf2sos

from .constants import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    BANDPASS_ORDER,
    POWERLINE_HZ,
    TARGET_FS,
)


# ---------------------------------------------------------------------------
# Bandpass
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    fs: int = TARGET_FS,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Zero-phase 4th-order Butterworth bandpass, applied along the last axis.

    Accepts shape (..., n_samples). The 4th-order design is on each side, so
    the effective roll-off is 8th-order after `filtfilt` doubling.
    """
    _validate_signal(signal)
    nyq = 0.5 * fs
    if not (0 < low < high < nyq):
        raise ValueError(
            f"Invalid bandpass corners for fs={fs}: need 0 < low({low}) < high({high}) < nyq({nyq})"
        )
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    # sosfiltfilt applies the filter forward then backward -> zero phase.
    padlen = min(3 * (sos.shape[0] * 2 + 1), signal.shape[-1] - 1)
    return sosfiltfilt(sos, signal, axis=-1, padlen=padlen).astype(np.float32)


# ---------------------------------------------------------------------------
# Notch (powerline)
# ---------------------------------------------------------------------------

def notch_filter(
    signal: np.ndarray,
    fs: int = TARGET_FS,
    freq: float = POWERLINE_HZ,
    quality: float = 30.0,
) -> np.ndarray:
    """Narrow notch at `freq` Hz (default 50 Hz for IN/EU mains)."""
    _validate_signal(signal)
    nyq = 0.5 * fs
    if not (0 < freq < nyq):
        raise ValueError(f"Notch frequency {freq} must satisfy 0 < freq < nyq({nyq})")
    b, a = iirnotch(freq / nyq, quality)
    sos = tf2sos(b, a)
    padlen = min(3 * (sos.shape[0] * 2 + 1), signal.shape[-1] - 1)
    return sosfiltfilt(sos, signal, axis=-1, padlen=padlen).astype(np.float32)


# ---------------------------------------------------------------------------
# Baseline wander
# ---------------------------------------------------------------------------

def remove_baseline_wander(
    signal: np.ndarray,
    fs: int = TARGET_FS,
) -> np.ndarray:
    """Estimate and subtract baseline drift with cascaded median filters.

    Method:
        1. 200 ms median filter  -> removes P and T waves (narrower than 200 ms).
        2. 600 ms median filter  -> removes QRS complexes (narrower than 600 ms).
        3. Subtract resulting baseline estimate from the input signal.

    This is the standard clinical approach and preserves ST-segment shape far
    better than a simple high-pass filter.
    """
    _validate_signal(signal)
    win1 = _odd(int(0.200 * fs))  # 200 ms
    win2 = _odd(int(0.600 * fs))  # 600 ms

    # medfilt works on 1-D or N-D but applies kernel per-element; we want
    # 1-D filtering along the last axis only, so iterate over leading axes.
    def _apply(x: np.ndarray) -> np.ndarray:
        b1 = medfilt(x, kernel_size=win1)
        b2 = medfilt(b1, kernel_size=win2)
        return x - b2

    if signal.ndim == 1:
        return _apply(signal).astype(np.float32)

    out = np.empty_like(signal, dtype=np.float32)
    flat = signal.reshape(-1, signal.shape[-1])
    for i in range(flat.shape[0]):
        out.reshape(-1, signal.shape[-1])[i] = _apply(flat[i])
    return out


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def zscore_normalize(
    signal: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-lead z-score normalization along the last axis.

    A lead with (near-)zero std (flat line) is left as zeros rather than
    producing NaN or exploding the output.
    """
    _validate_signal(signal)
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    safe_std = np.where(std < eps, 1.0, std)
    z = (signal - mean) / safe_std
    # Lead was flat -> keep it at zero rather than amplifying tiny noise.
    z = np.where(std < eps, 0.0, z)
    return z.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_signal(signal: np.ndarray) -> None:
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(signal).__name__}")
    if signal.ndim < 1:
        raise ValueError("Signal must have at least one dimension")
    if signal.shape[-1] < 8:
        raise ValueError(f"Signal too short along last axis: {signal.shape[-1]}")


def _odd(n: int) -> int:
    """medfilt requires odd kernel sizes."""
    return n if n % 2 == 1 else n + 1
