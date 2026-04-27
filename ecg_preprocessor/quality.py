"""
Signal Quality Index (SQI).

We compute a composite per-lead SQI in [0, 1] from four sub-metrics, then
average across leads for the overall score. The sub-metrics are chosen to
catch the specific failure modes we see with the ADS1292 + dry-electrode
acquisition hardware:

*   flatline        — electrode pulled off, cable unplugged
*   clipping        — gain too high, saturation at ADC rail
*   noise power     — loose contact, motion artifact, myographic noise
*   kurtosis        — a healthy ECG has kurtosis ~5; very low or very
                      high values indicate non-ECG content

The overall SQI is a weighted product-of-penalties (not a raw average),
because a catastrophic failure on any single sub-metric should dominate the
score. A flatline lead should not be "rescued" by nice kurtosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis as scipy_kurtosis

from .constants import LEAD_ORDER, TARGET_FS


@dataclass
class LeadQuality:
    """Per-lead SQI breakdown."""
    lead: str
    sqi: float
    flatline_score: float
    clipping_score: float
    noise_score: float
    kurtosis_score: float
    flags: list[str] = field(default_factory=list)


@dataclass
class SignalQuality:
    """Overall SQI plus per-lead breakdown."""
    overall: float
    per_lead: list[LeadQuality]

    def as_dict(self) -> dict:
        return {
            "overall": float(self.overall),
            "per_lead": {
                lq.lead: {
                    "sqi": float(lq.sqi),
                    "flatline": float(lq.flatline_score),
                    "clipping": float(lq.clipping_score),
                    "noise": float(lq.noise_score),
                    "kurtosis": float(lq.kurtosis_score),
                    "flags": list(lq.flags),
                }
                for lq in self.per_lead
            },
        }


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def compute_sqi(
    signal: np.ndarray,
    fs: int = TARGET_FS,
    lead_names: Sequence[str] | None = None,
) -> SignalQuality:
    """Compute per-lead and overall SQI.

    Parameters
    ----------
    signal : np.ndarray, shape (n_leads, n_samples)
    fs : sampling rate
    lead_names : names matching signal rows. Defaults to canonical 12-lead.
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected 2-D signal (leads, samples), got shape {signal.shape}")
    n_leads = signal.shape[0]
    if lead_names is None:
        if n_leads != len(LEAD_ORDER):
            raise ValueError(
                f"Got {n_leads} leads but no lead_names provided; expected {len(LEAD_ORDER)}"
            )
        lead_names = list(LEAD_ORDER)

    per_lead: list[LeadQuality] = []
    for i, name in enumerate(lead_names):
        per_lead.append(_score_lead(signal[i], fs, name))

    # Overall: geometric mean so a single bad lead pulls the score down hard.
    scores = np.array([lq.sqi for lq in per_lead], dtype=np.float64)
    # +eps so a single 0.0 doesn't completely zero the product.
    overall = float(np.exp(np.log(scores + 1e-6).mean()) - 1e-6)
    overall = max(0.0, min(1.0, overall))
    return SignalQuality(overall=overall, per_lead=per_lead)


# ---------------------------------------------------------------------------
# Per-lead scoring
# ---------------------------------------------------------------------------

def _score_lead(x: np.ndarray, fs: int, lead_name: str) -> LeadQuality:
    flags: list[str] = []

    flat = _flatline_score(x)
    if flat < 0.3:
        flags.append("flatline")

    clip = _clipping_score(x)
    if clip < 0.3:
        flags.append("clipping")

    noise = _noise_score(x, fs)
    if noise < 0.3:
        flags.append("noisy")

    kurt = _kurtosis_score(x)
    if kurt < 0.3:
        flags.append("non_ecg_kurtosis")

    # Composite: weighted geometric mean. Weights reflect severity —
    # flatline and clipping are disqualifying, noise/kurtosis are softer.
    weights = np.array([0.35, 0.30, 0.20, 0.15])
    scores = np.array([flat, clip, noise, kurt])
    sqi = float(np.prod(np.maximum(scores, 1e-3) ** weights))
    sqi = max(0.0, min(1.0, sqi))

    return LeadQuality(
        lead=lead_name,
        sqi=sqi,
        flatline_score=flat,
        clipping_score=clip,
        noise_score=noise,
        kurtosis_score=kurt,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Sub-metrics. Each returns a value in [0, 1] where 1 is perfect.
# ---------------------------------------------------------------------------

def _flatline_score(x: np.ndarray) -> float:
    """Detect flat segments. Uses windowed std vs global std ratio."""
    if x.size == 0:
        return 0.0
    global_std = float(np.std(x))
    if global_std < 1e-8:
        return 0.0  # Entire signal is flat.

    # Scan with 500 ms windows; count fraction where local std is < 5% global.
    win = max(64, x.size // 20)
    stride = max(1, win // 2)
    n_windows = max(1, (x.size - win) // stride + 1)
    flat_count = 0
    for i in range(n_windows):
        seg = x[i * stride : i * stride + win]
        if seg.size == 0:
            continue
        if np.std(seg) < 0.05 * global_std:
            flat_count += 1
    flat_frac = flat_count / n_windows
    # 0% flat -> 1.0, 30%+ flat -> 0.0
    return float(np.clip(1.0 - (flat_frac / 0.30), 0.0, 1.0))


def _clipping_score(x: np.ndarray) -> float:
    """Detect ADC clipping: repeated samples at the signal's extrema."""
    if x.size == 0:
        return 0.0
    vmax, vmin = float(np.max(x)), float(np.min(x))
    if vmax == vmin:
        return 0.0
    # Tolerance band near each extreme.
    top_band = vmax - 0.005 * (vmax - vmin)
    bot_band = vmin + 0.005 * (vmax - vmin)
    top_frac = float(np.mean(x >= top_band))
    bot_frac = float(np.mean(x <= bot_band))
    worst = max(top_frac, bot_frac)
    # 1% at rail is fine (natural peaks); 5%+ is clipping.
    return float(np.clip(1.0 - (worst - 0.01) / 0.04, 0.0, 1.0))


def _noise_score(x: np.ndarray, fs: int) -> float:
    """Ratio of power in the physiological ECG band vs total power.

    A clean ECG has most of its energy in 1-40 Hz; motion/EMG/line artifact
    shows up as excess power in <1 Hz or >40 Hz bands.
    """
    if x.size < fs // 2:
        # Not enough data for a meaningful Welch estimate.
        return 0.5
    try:
        nperseg = min(len(x), max(256, fs))
        f, psd = welch(x, fs=fs, nperseg=nperseg)
    except Exception:
        return 0.5
    total = float(np.trapezoid(psd, f))
    if total <= 0:
        return 0.0
    ecg_band = (f >= 1.0) & (f <= 40.0)
    band_power = float(np.trapezoid(psd[ecg_band], f[ecg_band]))
    ratio = band_power / total
    # ratio > 0.85 -> clean; < 0.4 -> very noisy.
    return float(np.clip((ratio - 0.40) / 0.45, 0.0, 1.0))


def _kurtosis_score(x: np.ndarray) -> float:
    """ECG Pearson kurtosis is highly variable (~4-30 across healthy leads).

    We reward anything in the wide "ECG-like" band [4, 40] with ~1.0, and
    penalize extremes: near-Gaussian (<3, typical of noise) and extremely
    peaky (>60, typical of isolated clipping spikes or runs of zeros).
    """
    if x.size < 100 or float(np.std(x)) < 1e-8:
        return 0.0
    # Fisher=False -> Pearson kurtosis, normal ~ 3.
    k = float(scipy_kurtosis(x, fisher=False, bias=False))
    if not np.isfinite(k):
        return 0.0
    # Piecewise: plateau at 1.0 inside [4, 40], soft shoulders outside.
    if 4.0 <= k <= 40.0:
        return 1.0
    if k < 4.0:
        # 3.0 (Gaussian) -> ~0.37 ; 2.0 -> ~0.13 ; below that collapses.
        return float(np.exp(-((k - 4.0) ** 2) / (2 * 1.0 ** 2)))
    # k > 40
    return float(np.exp(-((k - 40.0) ** 2) / (2 * 20.0 ** 2)))
