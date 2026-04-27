"""Unit tests for filters: bandpass, notch, baseline removal, normalization."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from ecg_preprocessor.filters import (
    bandpass_filter,
    notch_filter,
    remove_baseline_wander,
    zscore_normalize,
)


FS = 500


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpass:

    def test_passes_signal_in_band_unchanged_amplitude(self):
        t = np.arange(0, 5, 1 / FS)
        x = np.sin(2 * np.pi * 10 * t).astype(np.float32)   # 10 Hz, in band
        y = bandpass_filter(x, fs=FS)
        # Filtering a steady sine should keep amplitude near 1.0.
        assert abs(y.std() - x.std()) / x.std() < 0.05

    def test_attenuates_dc_and_low_freq_drift(self):
        t = np.arange(0, 5, 1 / FS)
        x = np.sin(2 * np.pi * 0.1 * t).astype(np.float32)   # 0.1 Hz drift
        y = bandpass_filter(x, fs=FS)
        assert y.std() < 0.1 * x.std()

    def test_attenuates_high_freq_above_band(self):
        t = np.arange(0, 10, 1 / FS)
        # 150 Hz is well into the stopband (cutoff 40 Hz, 8th-order effective
        # after sosfiltfilt doubling). Edge transients from zero-phase padding
        # inflate the full-signal std; the meaningful check is the interior.
        x = np.sin(2 * np.pi * 150 * t).astype(np.float32)
        y = bandpass_filter(x, fs=FS)
        n = len(y)
        interior = y[n // 10 : -n // 10]
        assert interior.std() < 0.1 * x.std()

    def test_preserves_multi_lead_shape(self):
        sig = np.random.randn(12, 2500).astype(np.float32)
        y = bandpass_filter(sig, fs=FS)
        assert y.shape == sig.shape
        assert y.dtype == np.float32

    def test_rejects_invalid_corners(self):
        x = np.random.randn(2500).astype(np.float32)
        with pytest.raises(ValueError):
            bandpass_filter(x, fs=FS, low=40, high=10)

    def test_rejects_corner_above_nyquist(self):
        x = np.random.randn(2500).astype(np.float32)
        with pytest.raises(ValueError):
            bandpass_filter(x, fs=FS, low=0.5, high=FS)   # >= Nyquist

    def test_rejects_non_array_input(self):
        with pytest.raises(TypeError):
            bandpass_filter([1, 2, 3], fs=FS)   # type: ignore[arg-type]

    def test_zero_input_stays_zero(self):
        x = np.zeros(2500, dtype=np.float32)
        y = bandpass_filter(x, fs=FS)
        assert np.allclose(y, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# notch_filter
# ---------------------------------------------------------------------------

class TestNotch:

    def test_removes_50hz_powerline(self):
        t = np.arange(0, 5, 1 / FS)
        ecg = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        noise = 0.5 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
        x = ecg + noise
        y = notch_filter(x, fs=FS, freq=50.0)

        # The 50 Hz component should be strongly attenuated in the PSD.
        f, psd_before = welch(x, fs=FS, nperseg=1024)
        f, psd_after = welch(y, fs=FS, nperseg=1024)
        band = (f >= 48) & (f <= 52)
        assert psd_after[band].sum() < 0.1 * psd_before[band].sum()

    def test_preserves_in_band_signal(self):
        t = np.arange(0, 5, 1 / FS)
        x = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        y = notch_filter(x, fs=FS, freq=50.0)
        # 10 Hz content is far from the notch — should be essentially preserved.
        assert abs(y.std() - x.std()) / x.std() < 0.05

    def test_60hz_mode(self):
        t = np.arange(0, 5, 1 / FS)
        noise = np.sin(2 * np.pi * 60 * t).astype(np.float32)
        y = notch_filter(noise, fs=FS, freq=60.0)
        assert y.std() < 0.2 * noise.std()

    def test_rejects_freq_out_of_range(self):
        x = np.random.randn(2500).astype(np.float32)
        with pytest.raises(ValueError):
            notch_filter(x, fs=FS, freq=FS)   # >= Nyquist


# ---------------------------------------------------------------------------
# remove_baseline_wander
# ---------------------------------------------------------------------------

class TestBaselineWander:

    def test_removes_slow_drift(self):
        t = np.arange(0, 10, 1 / FS)
        drift = 3.0 * np.sin(2 * np.pi * 0.2 * t)           # 0.2 Hz, big drift
        ecg = np.sin(2 * np.pi * 15 * t) * 0.5              # pseudo-QRS energy
        x = (drift + ecg).astype(np.float32)
        y = remove_baseline_wander(x, fs=FS)

        # Low-freq power (< 0.5 Hz) should drop dramatically.
        f, psd_before = welch(x, fs=FS, nperseg=2048)
        f, psd_after = welch(y, fs=FS, nperseg=2048)
        low = f < 0.5
        assert psd_after[low].sum() < 0.1 * psd_before[low].sum()

    def test_1d_input(self):
        x = np.random.randn(5000).astype(np.float32)
        y = remove_baseline_wander(x, fs=FS)
        assert y.shape == x.shape

    def test_2d_input_each_lead_independent(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((12, 5000)).astype(np.float32)
        y = remove_baseline_wander(sig, fs=FS)
        assert y.shape == sig.shape
        # Removing baseline from random noise should leave something
        # near-zero-mean and similar variance (median filter is mild on noise).
        assert np.all(np.abs(y.mean(axis=-1)) < 0.5)


# ---------------------------------------------------------------------------
# zscore_normalize
# ---------------------------------------------------------------------------

class TestZScore:

    def test_output_has_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((12, 5000)).astype(np.float32) * 3 + 7
        y = zscore_normalize(x)
        assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)
        assert np.allclose(y.std(axis=-1), 1.0, atol=1e-4)

    def test_flat_lead_stays_zero(self):
        x = np.zeros((3, 1000), dtype=np.float32)
        x[1] = 5.0   # constant non-zero lead
        y = zscore_normalize(x)
        assert np.allclose(y[0], 0.0)
        assert np.allclose(y[1], 0.0)   # constant -> zero after safe-std handling

    def test_no_nans_on_degenerate_input(self):
        x = np.zeros((2, 500), dtype=np.float32)
        y = zscore_normalize(x)
        assert not np.isnan(y).any()

    def test_preserves_dtype(self):
        x = np.random.randn(5, 1000).astype(np.float32)
        y = zscore_normalize(x)
        assert y.dtype == np.float32


# ---------------------------------------------------------------------------
# Cross-cutting: pipeline order invariants
# ---------------------------------------------------------------------------

def test_filter_chain_on_realistic_signal():
    """Bandpass -> notch -> baseline -> z-score should produce clean output."""
    from tests_preprocessor.conftest import synth_ecg

    x = synth_ecg(duration=10.0, baseline_drift=0.2, powerline=0.1, seed=0)
    y = bandpass_filter(x, fs=FS)
    y = notch_filter(y, fs=FS)
    y = remove_baseline_wander(y, fs=FS)
    y = zscore_normalize(y[None, :])[0]

    # End result: zero mean, unit std, no NaNs.
    assert abs(float(y.mean())) < 1e-4
    assert abs(float(y.std()) - 1.0) < 1e-3
    assert not np.isnan(y).any()
