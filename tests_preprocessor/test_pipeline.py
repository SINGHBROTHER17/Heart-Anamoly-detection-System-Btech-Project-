"""Integration tests for the end-to-end ECGPreprocessor pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_preprocessor import (
    ECGPreprocessor,
    LEAD_ORDER,
    SignalQualityError,
    WINDOW_SAMPLES,
    preprocess,
)


FS = 500


class TestPipelineHappyPath:

    def test_output_shape_is_12x5000(self, clean_12lead):
        r = ECGPreprocessor().run(clean_12lead, sample_rate=FS)
        assert r.signal.shape == (12, WINDOW_SAMPLES)

    def test_output_is_zscore_normalized(self, clean_12lead):
        r = ECGPreprocessor().run(clean_12lead, sample_rate=FS)
        # Per-lead mean near zero, std near 1 (except any flat lead).
        lead_means = r.signal.mean(axis=-1)
        lead_stds = r.signal.std(axis=-1)
        assert np.all(np.abs(lead_means) < 1e-4)
        assert np.all(np.abs(lead_stds - 1.0) < 1e-3)

    def test_quality_overall_range(self, clean_12lead):
        r = ECGPreprocessor().run(clean_12lead, sample_rate=FS)
        assert 0.0 <= r.quality.overall <= 1.0
        assert r.quality.overall > 0.6   # clean signal should pass SQI

    def test_r_peaks_found_in_reference(self, clean_12lead):
        r = ECGPreprocessor().run(clean_12lead, sample_rate=FS)
        # 60 BPM over 10 s -> ~10 beats after windowing.
        assert 8 <= len(r.r_peaks_ref) <= 12

    def test_shifts_returned_for_each_lead(self, clean_12lead):
        r = ECGPreprocessor().run(clean_12lead, sample_rate=FS)
        assert len(r.shifts) == 12


class TestPipelineDesync:

    def test_recovers_desynchronization(self, desynchronized_12lead):
        sig, applied = desynchronized_12lead
        r = ECGPreprocessor().run(sig, sample_rate=FS)
        # Each lead's residual (applied + recovered) should be within tolerance.
        tolerance_samples = 20   # 40 ms at 500 Hz
        for i in range(12):
            residual = int(applied[i]) + int(r.shifts[i])
            assert abs(residual) <= tolerance_samples, (
                f"Lead {i}: applied={applied[i]}, recovered={r.shifts[i]}"
            )


class TestPipelineBadSignal:

    def test_rejects_all_zero_signal(self):
        zeros = np.zeros((12, 6000), dtype=np.float32)
        with pytest.raises(SignalQualityError) as excinfo:
            ECGPreprocessor().run(zeros, sample_rate=FS)
        assert excinfo.value.sqi < 0.6
        assert len(excinfo.value.per_lead) == 12

    def test_error_has_per_lead_breakdown(self):
        zeros = np.zeros((12, 6000), dtype=np.float32)
        try:
            ECGPreprocessor().run(zeros, sample_rate=FS)
        except SignalQualityError as exc:
            for lead in LEAD_ORDER:
                assert lead in exc.per_lead
                assert exc.per_lead[lead] < 0.5

    def test_configurable_sqi_threshold(self):
        """Setting threshold to 0 should let anything through."""
        zeros = np.zeros((12, 6000), dtype=np.float32)
        # Threshold 0 means we always accept.
        r = ECGPreprocessor(sqi_threshold=0.0).run(zeros, sample_rate=FS)
        assert r.signal.shape == (12, WINDOW_SAMPLES)


class TestPipelineInputTypes:

    def test_numpy_array_input(self, clean_12lead):
        r = ECGPreprocessor().run(clean_12lead, sample_rate=FS)
        assert r.signal.shape == (12, WINDOW_SAMPLES)

    def test_csv_file_input(self, tmp_path, clean_12lead):
        csv_path = tmp_path / "ecg.csv"
        pd.DataFrame({
            name: clean_12lead[i] for i, name in enumerate(LEAD_ORDER)
        }).to_csv(csv_path, index=False)
        r = ECGPreprocessor().run(csv_path)
        assert r.signal.shape == (12, WINDOW_SAMPLES)

    def test_json_payloads_input(self, tmp_path, clean_12lead):
        payloads = []
        for i, name in enumerate(LEAD_ORDER):
            p = tmp_path / f"{name}.json"
            p.write_text(json.dumps({
                "lead_name": name,
                "samples": list(clean_12lead[i].astype(float)),
                "sample_rate": FS,
            }))
            payloads.append(p)
        r = ECGPreprocessor().run(payloads)
        assert r.signal.shape == (12, WINDOW_SAMPLES)

    def test_partial_leads_zero_padded(self, clean_12lead):
        # Only supply 6 leads.
        partial = clean_12lead[:6]
        # Partial input can fail SQI because half the leads are flat zeros.
        # Force threshold to 0 to just test the shape plumbing.
        r = ECGPreprocessor(sqi_threshold=0.0).run(partial, sample_rate=FS)
        assert r.signal.shape == (12, WINDOW_SAMPLES)

    def test_handles_differing_sample_rate(self, clean_12lead):
        # Simulate that the source was actually recorded at 1000 Hz.
        from scipy.signal import resample_poly
        upsampled = resample_poly(clean_12lead, 2, 1, axis=-1).astype(np.float32)
        r = ECGPreprocessor().run(upsampled, sample_rate=1000)
        assert r.signal.shape == (12, WINDOW_SAMPLES)


class TestConvenienceFunction:

    def test_preprocess_one_shot(self, clean_12lead):
        r = preprocess(clean_12lead, sample_rate=FS)
        assert r.signal.shape == (12, WINDOW_SAMPLES)

    def test_preprocess_accepts_kwargs(self, clean_12lead):
        r = preprocess(clean_12lead, sample_rate=FS, sqi_threshold=0.1)
        assert r.signal.shape == (12, WINDOW_SAMPLES)
