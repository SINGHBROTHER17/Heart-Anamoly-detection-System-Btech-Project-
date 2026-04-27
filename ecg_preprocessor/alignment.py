"""
Inter-lead alignment for sequentially recorded ECG.

Problem:
    The acquisition hardware records one 3-lead bank at a time. When we
    stack leads into a (12, N) array, their R-peaks are NOT co-temporal —
    sample index `k` in Lead I and sample index `k` in V1 correspond to
    different heartbeats. The model assumes all 12 leads are synchronous,
    so we must realign.

Strategy (R-peak position matching):
    1. Pick a reference lead. Default Lead II (canonical choice — cleanest
       and highest-amplitude QRS in most patients).
    2. Detect R-peaks in the reference lead.
    3. For every other lead, detect R-peaks. If none detected with normal
       polarity, retry on the negated signal (catches leads with inverted
       QRS such as aVR).
    4. For each candidate shift s in [-max_shift, +max_shift]:
           score(s) = number of (lead_peak + s) values that land within a
                      small tolerance (~40 ms) of a reference R-peak.
       Pick the s that maximizes score. Ties broken by preferring smaller
       |s|, so we avoid gratuitous "full-beat" shifts when the data is
       already well aligned.
    5. Apply the winning shift to the lead (zero-pad vacated samples).

Why R-peak positions instead of waveform cross-correlation:
    A cross-correlation window matches ONE beat's morphology, but QRS
    morphology differs systematically across leads (that is the point of
    a 12-lead ECG — each lead sees the heart from a different angle).
    Matching R-peak positions uses only the temporal information that's
    guaranteed to be shared, and tolerates arbitrary morphology differences.

Leads with no detectable R-peaks (flatline, severe noise) are left
un-shifted; the SQI stage upstream will already have flagged them.
"""

from __future__ import annotations

import numpy as np

from .constants import LEAD_INDEX, TARGET_FS
from .peaks import detect_r_peaks

MAX_SHIFT_MS = 200.0
MATCH_TOLERANCE_MS = 40.0  # an R-peak counts as matched if within ±40 ms


def align_leads(
    signal: np.ndarray,
    fs: int = TARGET_FS,
    reference_lead: str = "II",
    max_shift_ms: float = MAX_SHIFT_MS,
    match_tolerance_ms: float = MATCH_TOLERANCE_MS,
) -> tuple[np.ndarray, list[int]]:
    """Align all leads to the reference using R-peak position matching.

    Parameters
    ----------
    signal : np.ndarray, shape (n_leads, n_samples)
    fs : sampling rate
    reference_lead : canonical lead name used as the time reference
    max_shift_ms : maximum absolute shift considered (ms)
    match_tolerance_ms : tolerance for calling two peaks matched (ms)

    Returns
    -------
    aligned : np.ndarray, shape (n_leads, n_samples)
        Each lead shifted (zero-padded) to align its R-peaks with the ref.
    shifts : list[int]
        Applied shift per lead, in samples. Positive = lead moved right
        (lead's R-peaks were earlier than ref's and got delayed).
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected (n_leads, n_samples), got {signal.shape}")
    n_leads = signal.shape[0]
    if reference_lead not in LEAD_INDEX:
        raise ValueError(f"Unknown reference lead: {reference_lead}")
    ref_idx = LEAD_INDEX[reference_lead]
    if ref_idx >= n_leads:
        ref_idx = 0

    max_shift = int(round(max_shift_ms / 1000.0 * fs))
    tol = max(1, int(round(match_tolerance_ms / 1000.0 * fs)))

    ref_peaks = detect_r_peaks(signal[ref_idx], fs=fs)

    aligned = signal.copy()
    shifts: list[int] = [0] * n_leads

    if ref_peaks.size == 0:
        # No usable reference — can't align anything. Return as-is.
        return aligned, shifts

    ref_sorted = np.sort(ref_peaks)

    for i in range(n_leads):
        if i == ref_idx:
            continue
        lead_peaks = detect_r_peaks(signal[i], fs=fs)
        if lead_peaks.size == 0:
            # Retry with inverted polarity (e.g. aVR usually has downward R).
            lead_peaks = detect_r_peaks(-signal[i], fs=fs)
            if lead_peaks.size == 0:
                continue

        best_shift = _best_shift_by_matching(
            lead_peaks, ref_sorted, max_shift=max_shift, tol=tol
        )
        shifts[i] = int(best_shift)
        aligned[i] = _shift_with_zero_pad(signal[i], best_shift)

    return aligned, shifts


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _best_shift_by_matching(
    lead_peaks: np.ndarray,
    ref_peaks: np.ndarray,
    max_shift: int,
    tol: int,
) -> int:
    """Pick integer shift s in [-max_shift, max_shift] that maximizes match count.

    Ties broken by preferring smaller |s|. We generate candidate shifts from
    all (ref_peak - lead_peak) pairs within the search window — this is
    orders of magnitude faster than scanning every integer shift, and
    provably includes the optimum (any optimal shift aligns at least one
    peak exactly, so it's in the difference set).
    """
    candidates: set[int] = {0}
    for lp in lead_peaks:
        # ref peaks that could contribute: within ±max_shift of lp.
        lo = np.searchsorted(ref_peaks, lp - max_shift, side="left")
        hi = np.searchsorted(ref_peaks, lp + max_shift, side="right")
        for rp in ref_peaks[lo:hi]:
            candidates.add(int(rp - lp))

    best_shift = 0
    best_matches = -1
    best_abs = 10**9
    for s in candidates:
        if abs(s) > max_shift:
            continue
        matches = _count_matches(lead_peaks + s, ref_peaks, tol)
        if matches > best_matches or (matches == best_matches and abs(s) < best_abs):
            best_matches = matches
            best_shift = s
            best_abs = abs(s)
    return best_shift


def _count_matches(shifted: np.ndarray, ref: np.ndarray, tol: int) -> int:
    """How many entries in `shifted` have a neighbor in `ref` within `tol`?"""
    if shifted.size == 0 or ref.size == 0:
        return 0
    idx = np.searchsorted(ref, shifted)
    count = 0
    for k, target in zip(idx, shifted):
        best = tol + 1
        if k < ref.size:
            best = min(best, abs(int(ref[k]) - int(target)))
        if k > 0:
            best = min(best, abs(int(ref[k - 1]) - int(target)))
        if best <= tol:
            count += 1
    return count


def _shift_with_zero_pad(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift 1-D array by `shift` samples, zero-padding vacated region."""
    out = np.zeros_like(x)
    if shift == 0:
        return x.copy()
    if shift > 0:
        out[shift:] = x[: len(x) - shift]
    else:
        out[: len(x) + shift] = x[-shift:]
    return out
