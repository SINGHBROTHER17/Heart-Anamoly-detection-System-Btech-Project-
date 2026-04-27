"""
Pan-Tompkins QRS / R-peak detector.

Implements the original 1985 algorithm faithfully:

    1. Bandpass 5-15 Hz     (isolate QRS spectrum)
    2. Derivative           (emphasize QRS slope)
    3. Squaring             (all-positive, amplify peaks)
    4. Moving-window integral (150 ms window -> smooth QRS into a bump)
    5. Adaptive thresholding + 200 ms refractory period
    6. Back-search in the bandpass-filtered signal for the actual R-peak
       (the integrator delays peaks by ~half the window width)

Adaptive thresholds are maintained as in the paper:
    SPKI = 0.125 * peak + 0.875 * SPKI   (signal peaks)
    NPKI = 0.125 * peak + 0.875 * NPKI   (noise peaks)
    THRESHOLD_I1 = NPKI + 0.25 * (SPKI - NPKI)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .constants import TARGET_FS


def detect_r_peaks(
    signal: np.ndarray,
    fs: int = TARGET_FS,
    refractory_ms: float = 200.0,
) -> np.ndarray:
    """Return indices of detected R-peaks in a single-lead ECG.

    Parameters
    ----------
    signal : 1-D array
    fs : sampling rate (Hz)
    refractory_ms : minimum interval between R-peaks (physiological limit ~200 ms)

    Returns
    -------
    peaks : np.ndarray[int]
    """
    if signal.ndim != 1:
        raise ValueError(f"detect_r_peaks expects 1-D input, got shape {signal.shape}")
    if signal.size < fs:  # need at least one second
        return np.array([], dtype=np.int64)

    refractory = int(round(refractory_ms / 1000.0 * fs))

    # --- 1. Bandpass 5-15 Hz -------------------------------------------------
    nyq = 0.5 * fs
    sos = butter(2, [5.0 / nyq, 15.0 / nyq], btype="band", output="sos")
    padlen = min(3 * (sos.shape[0] * 2 + 1), signal.shape[-1] - 1)
    bp = sosfiltfilt(sos, signal, padlen=padlen)

    # --- 2. Derivative -------------------------------------------------------
    # 5-point derivative from the original paper:
    #   y[n] = (1/8T) * ( -x[n-2] - 2*x[n-1] + 2*x[n+1] + x[n+2] )
    k = np.array([-1, -2, 0, 2, 1], dtype=np.float64) / 8.0
    deriv = np.convolve(bp, k, mode="same")

    # --- 3. Squaring ---------------------------------------------------------
    squared = deriv * deriv

    # --- 4. Moving-window integral (150 ms) ----------------------------------
    win = max(1, int(round(0.150 * fs)))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    # --- 5. Adaptive thresholding on integrated signal -----------------------
    # Seed SPKI/NPKI from a 2-second learning phase.
    learn_end = min(2 * fs, len(integrated))
    learn_seg = integrated[:learn_end]
    spki = float(np.max(learn_seg)) * 0.25 if learn_seg.size else 0.0
    npki = float(np.mean(learn_seg)) * 0.5 if learn_seg.size else 0.0

    # Identify candidate local maxima in the integrated signal.
    candidates = _local_maxima(integrated)

    peaks_integrated: list[int] = []
    last_peak = -refractory  # allow first peak immediately
    for idx in candidates:
        if idx - last_peak < refractory:
            # Within refractory — keep the stronger peak.
            if peaks_integrated and integrated[idx] > integrated[peaks_integrated[-1]]:
                peaks_integrated[-1] = idx
                last_peak = idx
                spki = 0.125 * integrated[idx] + 0.875 * spki
            continue

        threshold = npki + 0.25 * (spki - npki)
        if integrated[idx] > threshold:
            peaks_integrated.append(idx)
            last_peak = idx
            spki = 0.125 * integrated[idx] + 0.875 * spki
        else:
            npki = 0.125 * integrated[idx] + 0.875 * npki

    # --- 6. Back-search: locate true R-peak in the bandpass signal -----------
    # The integrator window shifts peaks by ~win/2 samples; search a window
    # of ±win samples around each detection for the actual local max (abs val
    # handles negative R-peaks seen on aVR and some pathological leads).
    peaks_final: list[int] = []
    search = win
    for p in peaks_integrated:
        lo = max(0, p - search)
        hi = min(len(bp), p + search)
        if hi <= lo:
            continue
        local = bp[lo:hi]
        peaks_final.append(lo + int(np.argmax(np.abs(local))))

    # Dedupe (back-search can collapse two nearby detections).
    if not peaks_final:
        return np.array([], dtype=np.int64)
    peaks_final = sorted(set(peaks_final))
    pruned = [peaks_final[0]]
    for p in peaks_final[1:]:
        if p - pruned[-1] >= refractory:
            pruned.append(p)
    return np.asarray(pruned, dtype=np.int64)


def _local_maxima(x: np.ndarray) -> np.ndarray:
    """Indices where x[i-1] < x[i] >= x[i+1] (robust to flat plateaus)."""
    if x.size < 3:
        return np.array([], dtype=np.int64)
    # Use strict left, non-strict right to avoid double-counting plateaus.
    left = x[1:-1] > x[:-2]
    right = x[1:-1] >= x[2:]
    idx = np.where(left & right)[0] + 1
    return idx.astype(np.int64)
