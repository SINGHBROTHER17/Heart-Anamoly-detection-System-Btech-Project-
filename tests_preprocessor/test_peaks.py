"""Unit tests for Pan-Tompkins R-peak detection."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_preprocessor.peaks import detect_r_peaks


FS = 500


class TestDetectRPeaks:

    def test_finds_correct_number_of_beats(self):
        from tests_preprocessor.conftest import synth_ecg
        # 60 BPM over 12 s -> 12 beats (first is at t=0.3 s, last before 11.7 s)
        sig = synth_ecg(duration=12.0, hr_bpm=60.0, seed=0)
        peaks = detect_r_peaks(sig, fs=FS)
        # Tolerate ±1 beat for edge effects.
        assert 11 <= len(peaks) <= 13

    def test_peaks_near_expected_sample_positions(self):
        from tests_preprocessor.conftest import synth_ecg
        sig = synth_ecg(duration=10.0, hr_bpm=60.0, seed=0, noise_sigma=0.0,
                        baseline_drift=0.0, powerline=0.0)
        peaks = detect_r_peaks(sig, fs=FS)

        # R-peak of each beat occurs at beat_start + 0.17 s.
        expected = np.arange(0.3, 9.7, 1.0) + 0.17
        expected_samples = (expected * FS).astype(int)

        # Every detected peak should be within ±25 ms of an expected position.
        for p in peaks:
            nearest = expected_samples[np.argmin(np.abs(expected_samples - p))]
            assert abs(p - nearest) < int(0.025 * FS)

    def test_tachycardia_150_bpm(self):
        from tests_preprocessor.conftest import synth_ecg
        sig = synth_ecg(duration=10.0, hr_bpm=150.0, seed=0)
        peaks = detect_r_peaks(sig, fs=FS)
        # 150 BPM over 10 s -> ~25 beats.
        assert 22 <= len(peaks) <= 27

    def test_bradycardia_40_bpm(self):
        from tests_preprocessor.conftest import synth_ecg
        sig = synth_ecg(duration=15.0, hr_bpm=40.0, seed=0)
        peaks = detect_r_peaks(sig, fs=FS)
        # 40 BPM over 15 s -> ~10 beats.
        assert 8 <= len(peaks) <= 12

    def test_empty_output_for_flatline(self):
        x = np.zeros(5000, dtype=np.float32)
        peaks = detect_r_peaks(x, fs=FS)
        assert peaks.size == 0

    def test_empty_output_for_short_signal(self):
        x = np.random.randn(100).astype(np.float32)   # < 1s
        peaks = detect_r_peaks(x, fs=FS)
        assert peaks.size == 0

    def test_rejects_non_1d_input(self):
        x = np.random.randn(12, 5000).astype(np.float32)
        with pytest.raises(ValueError):
            detect_r_peaks(x, fs=FS)

    def test_refractory_period_enforced(self):
        from tests_preprocessor.conftest import synth_ecg
        sig = synth_ecg(duration=10.0, hr_bpm=60.0, seed=0)
        peaks = detect_r_peaks(sig, fs=FS, refractory_ms=200.0)
        if len(peaks) >= 2:
            diffs = np.diff(peaks)
            assert diffs.min() >= int(0.200 * FS)

    def test_robust_to_baseline_drift(self):
        from tests_preprocessor.conftest import synth_ecg
        sig = synth_ecg(duration=10.0, hr_bpm=60.0, baseline_drift=0.5, seed=0)
        peaks = detect_r_peaks(sig, fs=FS)
        assert 9 <= len(peaks) <= 11

    def test_robust_to_powerline_noise(self):
        from tests_preprocessor.conftest import synth_ecg
        sig = synth_ecg(duration=10.0, hr_bpm=60.0, powerline=0.3, seed=0)
        peaks = detect_r_peaks(sig, fs=FS)
        assert 9 <= len(peaks) <= 11
