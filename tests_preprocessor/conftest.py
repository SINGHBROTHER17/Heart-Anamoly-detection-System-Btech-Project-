"""Shared pytest fixtures and a reusable synthetic ECG generator."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_preprocessor.constants import LEAD_ORDER, TARGET_FS


def synth_ecg(
    duration: float = 12.0,
    hr_bpm: float = 60.0,
    fs: int = TARGET_FS,
    amp: float = 1.0,
    noise_sigma: float = 0.02,
    baseline_drift: float = 0.05,
    powerline: float = 0.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Return a 1-D synthetic ECG trace with realistic-ish PQRST morphology.

    Uses a sum of Gaussian bumps for each of P, Q, R, S, T at clinically
    plausible offsets from beat onset. Not intended to fool a cardiologist —
    just to exercise the pipeline end-to-end with a signal whose SQI, R-peak
    positions and spectral content are deterministic and known.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)
    sig = np.zeros_like(t)
    for beat_start in np.arange(0.3, duration - 0.3, 60.0 / hr_bpm):
        # P
        sig += 0.15 * amp * np.exp(-((t - beat_start) ** 2) / (0.04 ** 2))
        # Q
        sig -= 0.10 * amp * np.exp(-((t - (beat_start + 0.15)) ** 2) / (0.01 ** 2))
        # R
        sig += 1.00 * amp * np.exp(-((t - (beat_start + 0.17)) ** 2) / (0.012 ** 2))
        # S
        sig -= 0.20 * amp * np.exp(-((t - (beat_start + 0.20)) ** 2) / (0.015 ** 2))
        # T
        sig += 0.30 * amp * np.exp(-((t - (beat_start + 0.35)) ** 2) / (0.04 ** 2))
    if noise_sigma > 0:
        sig += noise_sigma * rng.standard_normal(len(t))
    if baseline_drift > 0:
        sig += baseline_drift * np.sin(2 * np.pi * 0.3 * t)
    if powerline > 0:
        sig += powerline * np.sin(2 * np.pi * 50.0 * t)
    return sig.astype(np.float32)


def synth_12lead(
    duration: float = 12.0,
    hr_bpm: float = 60.0,
    fs: int = TARGET_FS,
    seed: int | None = 0,
    desync_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (12, N) synthetic ECG with optional per-lead rolled shifts.

    Returns
    -------
    signal : np.ndarray, shape (12, N)
    shifts : np.ndarray, shape (12,)  — the rolls actually applied
    """
    rng = np.random.default_rng(seed)
    base = synth_ecg(duration=duration, hr_bpm=hr_bpm, fs=fs, seed=seed)
    leads = np.tile(base[None, :], (len(LEAD_ORDER), 1)) \
        * np.linspace(0.85, 1.15, len(LEAD_ORDER))[:, None]

    if desync_samples is None:
        desync_samples = np.zeros(len(LEAD_ORDER), dtype=np.int64)

    shifts = np.asarray(desync_samples, dtype=np.int64)
    for i, s in enumerate(shifts):
        if s != 0:
            leads[i] = np.roll(leads[i], int(s))
    return leads.astype(np.float32), shifts


@pytest.fixture
def clean_12lead():
    """Clean, co-temporal 12-lead synthetic ECG."""
    return synth_12lead(seed=0)[0]


@pytest.fixture
def desynchronized_12lead():
    """12-lead ECG with known per-lead shifts, for alignment tests."""
    rng = np.random.default_rng(7)
    shifts = rng.integers(-80, 81, size=12).astype(np.int64)
    shifts[1] = 0   # lead II is the reference — don't roll it
    sig, applied = synth_12lead(seed=0, desync_samples=shifts)
    return sig, applied


@pytest.fixture
def sample_rate():
    return TARGET_FS
