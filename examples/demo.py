"""
examples/demo.py
================

End-to-end demonstration of the Phase 2 preprocessing pipeline.

Synthesises a 12-lead ECG with plausible PQRST morphology, injects the
three most common real-world artifacts (baseline wander, powerline hum,
per-lead desynchronization from sequential recording), then runs the
pipeline and prints the before/after summary.

Run from the project root:

    python -m examples.demo
    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running as "python examples/demo.py" from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg_preprocessor import ECGPreprocessor, LEAD_ORDER, SignalQualityError


def _synth_beat(t: np.ndarray, beat_start: float, amp: float = 1.0) -> np.ndarray:
    """Sum of Gaussians for P, Q, R, S, T at plausible offsets from beat onset."""
    return (
        0.15 * amp * np.exp(-((t - beat_start) ** 2) / 0.04 ** 2)          # P
        - 0.10 * amp * np.exp(-((t - (beat_start + 0.15)) ** 2) / 0.01 ** 2)   # Q
        + 1.00 * amp * np.exp(-((t - (beat_start + 0.17)) ** 2) / 0.012 ** 2)  # R
        - 0.20 * amp * np.exp(-((t - (beat_start + 0.20)) ** 2) / 0.015 ** 2)  # S
        + 0.30 * amp * np.exp(-((t - (beat_start + 0.35)) ** 2) / 0.04 ** 2)   # T
    )


def build_synthetic_12lead(
    duration: float = 12.0,
    hr_bpm: float = 72.0,
    fs: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a desynchronized, noisy (12, N) ECG + the ground-truth roll applied.

    Returns
    -------
    signal : np.ndarray, shape (12, N)
    applied_shifts : np.ndarray, shape (12,)
        Per-lead sample roll we injected — pipeline should recover roughly
        the negation of these values.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)

    # Build a clean beat train once, then replicate across 12 leads with
    # different amplitudes (a crude stand-in for lead-specific morphology).
    beats = np.zeros_like(t)
    for beat_start in np.arange(0.3, duration - 0.3, 60.0 / hr_bpm):
        beats += _synth_beat(t, beat_start)

    # Per-lead amplitude scaling.
    amp_scaling = np.linspace(0.85, 1.15, len(LEAD_ORDER))
    signal = np.tile(beats[None, :], (len(LEAD_ORDER), 1)) * amp_scaling[:, None]

    # Inject realistic artifacts.
    signal += 0.02 * rng.standard_normal(signal.shape)               # broadband noise
    signal += 0.15 * np.sin(2 * np.pi * 0.3 * t)[None, :]            # baseline wander
    signal += 0.04 * np.sin(2 * np.pi * 50.0 * t)[None, :]           # 50 Hz mains

    # Desynchronize each lead except the reference (Lead II, index 1).
    applied_shifts = rng.integers(-70, 71, size=len(LEAD_ORDER)).astype(np.int64)
    applied_shifts[1] = 0
    for i, s in enumerate(applied_shifts):
        if s != 0:
            signal[i] = np.roll(signal[i], int(s))

    return signal.astype(np.float32), applied_shifts


def main() -> int:
    fs = 500
    print("=" * 72)
    print("Phase 2 demo — synthetic 12-lead ECG through the preprocessing pipeline")
    print("=" * 72)

    raw, applied = build_synthetic_12lead(fs=fs)
    print(f"\nBuilt synthetic signal: shape={raw.shape}, fs={fs} Hz, "
          f"duration={raw.shape[1] / fs:.1f}s")
    print(f"Injected artifacts: noise, 0.3 Hz drift, 50 Hz mains, per-lead roll")
    print(f"Applied rolls (samples, negative of what pipeline should recover):")
    for i, (lead, s) in enumerate(zip(LEAD_ORDER, applied)):
        print(f"   lead {i:2d} [{lead:>3s}]: {s:+4d} samples ({s * 1000 / fs:+6.1f} ms)")

    pp = ECGPreprocessor()
    try:
        result = pp.run(raw, sample_rate=fs)
    except SignalQualityError as exc:
        print(f"\n[X] Pipeline refused the signal (SQI={exc.sqi:.2f}).")
        print("   Per-lead SQI:")
        for lead, sqi in exc.per_lead.items():
            print(f"     {lead}: {sqi:.2f}")
        return 1

    print(f"\n[OK] Pipeline output")
    print(f"   shape:           {result.signal.shape}")
    print(f"   dtype:           {result.signal.dtype}")
    print(f"   per-lead mean:   {result.signal.mean(axis=-1).round(6).tolist()}")
    print(f"   per-lead std:    {result.signal.std(axis=-1).round(4).tolist()}")

    print(f"\n[OK] Overall SQI: {result.quality.overall:.3f}")
    print(f"   Per-lead SQI:")
    for lq in result.quality.per_lead:
        flag_str = f"  flags: {lq.flags}" if lq.flags else ""
        print(f"     {lq.lead:>3s}: {lq.sqi:.2f}"
              f"  (flat={lq.flatline_score:.2f} "
              f"clip={lq.clipping_score:.2f} "
              f"noise={lq.noise_score:.2f} "
              f"kurt={lq.kurtosis_score:.2f}){flag_str}")

    print(f"\n[OK] R-peaks in reference lead (II): {len(result.r_peaks_ref)} detected "
          f"(expected ~{int(result.signal.shape[1] / fs * 72 / 60)} for 72 BPM/10s)")

    print(f"\n[OK] Alignment recovery (should sum to ~0 per lead within ±20 samples):")
    residuals = []
    for i, (lead, inj, rec) in enumerate(zip(LEAD_ORDER, applied, result.shifts)):
        residual = int(inj) + int(rec)
        residuals.append(residual)
        mark = "OK" if abs(residual) <= 20 else "FAIL"
        print(f"   lead {i:2d} [{lead:>3s}]: "
              f"injected={inj:+4d}  recovered={rec:+4d}  residual={residual:+4d}  [{mark}]")
    aligned = sum(1 for r in residuals if abs(r) <= 20)
    print(f"\n   {aligned}/{len(LEAD_ORDER)} leads aligned within tolerance")

    return 0


if __name__ == "__main__":
    sys.exit(main())
