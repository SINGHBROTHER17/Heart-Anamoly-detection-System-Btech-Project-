"""Unit tests for window segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_preprocessor.segmentation import segment_fixed_window
from ecg_preprocessor.exceptions import InvalidInputError


class TestSegmentFixedWindow:

    def test_same_length_passthrough(self):
        x = np.random.randn(12, 5000).astype(np.float32)
        y = segment_fixed_window(x, window=5000)
        assert y.shape == x.shape
        assert y.dtype == np.float32

    def test_crop_center(self):
        x = np.arange(12 * 8000, dtype=np.float32).reshape(12, 8000)
        y = segment_fixed_window(x, window=5000, mode="center")
        assert y.shape == (12, 5000)
        # Center crop on row 0 starts at (8000-5000)//2 = 1500.
        assert y[0, 0] == x[0, 1500]
        assert y[0, -1] == x[0, 1500 + 4999]

    def test_crop_start(self):
        x = np.arange(12 * 8000, dtype=np.float32).reshape(12, 8000)
        y = segment_fixed_window(x, window=5000, mode="start")
        assert np.array_equal(y, x[:, :5000])

    def test_crop_end(self):
        x = np.arange(12 * 8000, dtype=np.float32).reshape(12, 8000)
        y = segment_fixed_window(x, window=5000, mode="end")
        assert np.array_equal(y, x[:, -5000:])

    def test_pad_center(self):
        x = np.ones((12, 3000), dtype=np.float32)
        y = segment_fixed_window(x, window=5000, mode="center")
        assert y.shape == (12, 5000)
        # Content lives in the middle; edges are zero.
        assert y[0, 0] == 0
        assert y[0, -1] == 0
        assert y[0, 2500] == 1.0

    def test_pad_start(self):
        x = np.ones((1, 3000), dtype=np.float32)
        y = segment_fixed_window(x, window=5000, mode="start")
        assert y[0, 0] == 1.0
        assert y[0, 3000] == 0.0

    def test_pad_end(self):
        x = np.ones((1, 3000), dtype=np.float32)
        y = segment_fixed_window(x, window=5000, mode="end")
        assert y[0, 0] == 0.0
        assert y[0, -1] == 1.0

    def test_rejects_empty(self):
        with pytest.raises((InvalidInputError, ValueError, IndexError)):
            segment_fixed_window(np.array([], dtype=np.float32).reshape(0, 0), window=5000)

    def test_unknown_mode_raises(self):
        x = np.ones((1, 3000), dtype=np.float32)
        with pytest.raises(InvalidInputError):
            segment_fixed_window(x, window=5000, mode="nope")

    def test_preserves_leading_axes(self):
        x = np.random.randn(3, 12, 8000).astype(np.float32)
        y = segment_fixed_window(x, window=5000, mode="center")
        assert y.shape == (3, 12, 5000)
