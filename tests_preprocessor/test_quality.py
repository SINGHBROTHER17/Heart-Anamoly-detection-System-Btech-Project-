"""Unit tests for the Signal Quality Index module."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_preprocessor.constants import LEAD_ORDER
from ecg_preprocessor.quality import (
    SignalQuality,
    compute_sqi,
    _clipping_score,
    _flatline_score,
    _kurtosis_score,
    _noise_score,
)


FS = 500


# ---------------------------------------------------------------------------
# Individual sub-metrics
# ---------------------------------------------------------------------------

class TestFlatline:

    def test_completely_flat_scores_zero(self):
        x = np.zeros(5000, dtype=np.float32)
        assert _flatline_score(x) == pytest.approx(0.0, abs=1e-6)

    def test_constant_nonzero_scores_zero(self):
        x = np.full(5000, 2.5, dtype=np.float32)
        assert _flatline_score(x) == pytest.approx(0.0, abs=1e-6)

    def test_random_signal_scores_high(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(5000).astype(np.float32)
        assert _flatline_score(x) > 0.9


class TestClipping:

    def test_clipped_signal_scores_low(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(5000).astype(np.float32)
        # Clip 20% of samples at the top.
        x[: 1000] = x.max()
        score = _clipping_score(x)
        assert score < 0.3

    def test_clean_signal_scores_high(self):
        # A synthetic ECG has narrow peaks and most samples near zero --
        # nothing resembling ADC clipping.
        from tests_preprocessor.conftest import synth_ecg
        x = synth_ecg(duration=10.0, seed=0)
        assert _clipping_score(x) > 0.9

    def test_zero_signal_returns_zero(self):
        x = np.zeros(5000, dtype=np.float32)
        assert _clipping_score(x) == pytest.approx(0.0, abs=1e-6)


class TestNoise:

    def test_in_band_signal_scores_high(self):
        t = np.arange(0, 10, 1 / FS)
        # 10 Hz sine is well inside the ECG band.
        x = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        assert _noise_score(x, fs=FS) > 0.8

    def test_out_of_band_signal_scores_low(self):
        t = np.arange(0, 10, 1 / FS)
        x = np.sin(2 * np.pi * 120 * t).astype(np.float32)   # above band
        assert _noise_score(x, fs=FS) < 0.3


class TestKurtosis:

    def test_gaussian_noise_scores_low(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(5000).astype(np.float32)
        # Gaussian has kurtosis ~3 (normal distribution); that sits below
        # the ECG band's plateau [4, 40] so the score must be < 1.0 and
        # notably lower than for real ECG.
        assert _kurtosis_score(x) < 0.7

    def test_peaky_ecg_like_signal_scores_high(self):
        from tests_preprocessor.conftest import synth_ecg
        x = synth_ecg(duration=10.0, seed=0)
        assert _kurtosis_score(x) > 0.8


# ---------------------------------------------------------------------------
# compute_sqi (integration)
# ---------------------------------------------------------------------------

class TestComputeSqi:

    def test_returns_dataclass_with_per_lead(self, clean_12lead):
        sqi = compute_sqi(clean_12lead, fs=FS)
        assert isinstance(sqi, SignalQuality)
        assert 0.0 <= sqi.overall <= 1.0
        assert len(sqi.per_lead) == len(LEAD_ORDER)
        for lq in sqi.per_lead:
            assert lq.lead in LEAD_ORDER
            assert 0.0 <= lq.sqi <= 1.0

    def test_clean_signal_scores_high(self, clean_12lead):
        sqi = compute_sqi(clean_12lead, fs=FS)
        assert sqi.overall > 0.8

    def test_flatline_lead_drags_score_down(self, clean_12lead):
        sig = clean_12lead.copy()
        sig[5] = 0.0   # flatline one lead
        sqi = compute_sqi(sig, fs=FS)
        assert sqi.per_lead[5].sqi < 0.2
        assert "flatline" in sqi.per_lead[5].flags
        # Geometric mean means one bad lead pulls overall noticeably.
        assert sqi.overall < 0.7

    def test_clipped_lead_flagged(self, clean_12lead):
        sig = clean_12lead.copy()
        # Saturate 30% of lead 0 at its max.
        top = float(sig[0].max())
        sig[0, :1500] = top
        sqi = compute_sqi(sig, fs=FS)
        assert "clipping" in sqi.per_lead[0].flags

    def test_rejects_non_2d_input(self):
        x = np.random.randn(5000).astype(np.float32)
        with pytest.raises(ValueError):
            compute_sqi(x, fs=FS)

    def test_accepts_custom_lead_names(self):
        sig = np.random.randn(3, 5000).astype(np.float32)
        sqi = compute_sqi(sig, fs=FS, lead_names=["X", "Y", "Z"])
        assert [lq.lead for lq in sqi.per_lead] == ["X", "Y", "Z"]

    def test_as_dict_serializable(self, clean_12lead):
        sqi = compute_sqi(clean_12lead, fs=FS)
        d = sqi.as_dict()
        # Must be JSON-friendly (no numpy scalars).
        import json
        json.dumps(d)   # will raise if not serializable
