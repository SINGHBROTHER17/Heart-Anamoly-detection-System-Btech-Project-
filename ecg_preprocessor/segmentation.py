"""
Windowing: crop or pad every lead to exactly WINDOW_SAMPLES (5000 @ 500 Hz).

For a recording longer than 10 s we prefer the *center* window because the
first and last seconds of a typical clinical recording tend to include
electrode-placement noise.
"""

from __future__ import annotations

import numpy as np

from .constants import WINDOW_SAMPLES
from .exceptions import InvalidInputError


def segment_fixed_window(
    signal: np.ndarray,
    window: int = WINDOW_SAMPLES,
    mode: str = "center",
) -> np.ndarray:
    """Pad or crop `signal` along its last axis to exactly `window` samples.

    Parameters
    ----------
    signal : shape (..., n_samples)
    window : target length in samples
    mode : 'center' | 'start' | 'end' — where to anchor when cropping/padding

    Returns
    -------
    out : shape (..., window)
    """
    if signal.ndim < 1:
        raise InvalidInputError("Signal must have at least one dimension")
    if signal.size == 0 or signal.shape[-1] == 0:
        raise InvalidInputError("Signal is empty along the sample axis")
    n = signal.shape[-1]
    if n == window:
        return signal.astype(np.float32, copy=False)

    if n > window:
        return _crop(signal, window, mode).astype(np.float32)
    return _pad(signal, window, mode).astype(np.float32)


def _crop(signal: np.ndarray, window: int, mode: str) -> np.ndarray:
    n = signal.shape[-1]
    if mode == "center":
        start = (n - window) // 2
    elif mode == "start":
        start = 0
    elif mode == "end":
        start = n - window
    else:
        raise InvalidInputError(f"Unknown mode: {mode}")
    return signal[..., start : start + window]


def _pad(signal: np.ndarray, window: int, mode: str) -> np.ndarray:
    n = signal.shape[-1]
    pad_total = window - n
    if mode == "center":
        left = pad_total // 2
        right = pad_total - left
    elif mode == "start":
        left, right = 0, pad_total
    elif mode == "end":
        left, right = pad_total, 0
    else:
        raise InvalidInputError(f"Unknown mode: {mode}")
    pad_width = [(0, 0)] * (signal.ndim - 1) + [(left, right)]
    return np.pad(signal, pad_width, mode="constant", constant_values=0.0)
