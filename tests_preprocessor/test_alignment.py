"""Unit tests for inter-lead alignment."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_preprocessor.alignment import align_leads
from ecg_preprocessor.constants import LEAD_ORDER

FS = 500


class TestAlignLeads:

    def test_output_shape_preserved(self, clean_12lead):
        aligned, shifts = align_leads(clean_12lead, fs=FS)
        assert aligned.shape == clean_12lead.shape
        assert len(shifts) == clean_12lead.shape[0]

    def test_reference_lead_is_never_shifted(self, desynchronized_12lead):
        sig, _ = desynchronized_12lead
        aligned, shifts = align_leads(sig, fs=FS, reference_lead="II")
        # Lead II is index 1 in LEAD_ORDER.
        assert shifts[1] == 0
        assert np.array_equal(aligned[1], sig[1])

    def test_recovers_known_shifts(self, desynchronized_12lead):
        """Sum of applied roll + recovered shift should be within tolerance."""
        sig, applied = desynchronized_12lead
        _, shifts = align_leads(sig, fs=FS, reference_lead="II")

        for i in range(12):
            residual = int(applied[i]) + int(shifts[i])
            # Alignment match tolerance is ±40 ms == ±20 samples at 500 Hz.
            assert abs(residual) <= 20, (
                f"Lead {i}: rolled {applied[i]}, recovered {shifts[i]}, "
                f"residual {residual} samples"
            )

    def test_shifts_bounded_by_max_shift(self, desynchronized_12lead):
        sig, _ = desynchronized_12lead
        max_shift_ms = 200.0
        _, shifts = align_leads(sig, fs=FS, max_shift_ms=max_shift_ms)
        max_samples = int(round(max_shift_ms / 1000.0 * FS))
        assert all(abs(s) <= max_samples for s in shifts)

    def test_flatline_lead_not_shifted(self, clean_12lead):
        sig = clean_12lead.copy()
        sig[5] = 0.0   # kill one lead
        _, shifts = align_leads(sig, fs=FS)
        assert shifts[5] == 0

    def test_unknown_reference_raises(self, clean_12lead):
        with pytest.raises(ValueError):
            align_leads(clean_12lead, fs=FS, reference_lead="XYZ")

    def test_non_2d_input_raises(self):
        x = np.random.randn(5000).astype(np.float32)
        with pytest.raises(ValueError):
            align_leads(x, fs=FS)

    def test_clean_signal_no_shift(self, clean_12lead):
        _, shifts = align_leads(clean_12lead, fs=FS)
        # All leads already aligned -> shifts all within tolerance of zero.
        assert all(abs(s) <= 20 for s in shifts)

    def test_inverted_lead_still_aligns(self, clean_12lead):
        """A lead with inverted polarity (like aVR) should still be alignable."""
        sig = clean_12lead.copy()
        sig[3] = -sig[3]            # invert aVR (index 3)
        sig[3] = np.roll(sig[3], 30)   # and shift by 60 ms
        _, shifts = align_leads(sig, fs=FS)
        # Expect recovered shift close to -30 (reversing the roll).
        assert abs(shifts[3] + 30) <= 20
