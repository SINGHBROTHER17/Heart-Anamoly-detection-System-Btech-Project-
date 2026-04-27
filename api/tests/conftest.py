"""
Shared fixtures for API integration tests.

The real calibrated model checkpoint is 10+ MB and only produced by Phase 3
on a GPU. For tests we install a mock ModelBundle that returns deterministic
random-ish probabilities, so the full HTTP surface can be tested in under
a second on any machine.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(autouse=True)
def _disable_rate_limit(monkeypatch):
    """Rate limiting breaks test suites (all tests share one 'IP')."""
    monkeypatch.setenv("ECG_DISABLE_RATE_LIMIT", "1")
    # Also flip the already-imported limiter off, in case main is already loaded.
    from app.rate_limit import limiter
    limiter.enabled = False
    yield


@pytest.fixture(autouse=True)
def _mock_model(monkeypatch):
    """Replace load_model / get_model_bundle with a deterministic fake."""
    from app import model_loader

    class FakeBundle:
        def __init__(self):
            self.device = torch.device("cpu")
            self.checkpoint_path = "<fake>"

        def predict(self, signal: np.ndarray) -> np.ndarray:
            # Seed on signal magnitude so tests are deterministic.
            rng = np.random.default_rng(int(np.abs(signal).sum() * 1000) % (2**32))
            # Generate confidence scores spread across the tier thresholds so
            # tests exercise all tier branches.
            return rng.uniform(0.05, 0.9, size=10).astype(np.float32)

    fake = FakeBundle()
    monkeypatch.setattr(model_loader, "_bundle", fake)
    monkeypatch.setattr(model_loader, "load_model", lambda *a, **k: fake)
    yield


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a FastAPI TestClient with an isolated SQLite DB per test."""
    db_path = tmp_path / "reports.db"
    monkeypatch.setenv("ECG_DB_PATH", str(db_path))

    # Reset storage singleton so it picks up the new DB path.
    from app import storage as storage_mod
    monkeypatch.setattr(storage_mod, "_storage", None)

    # Reset the analysis service singleton too.
    from app import service as service_mod
    monkeypatch.setattr(service_mod, "_service", None)

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def synth_ecg_csv() -> bytes:
    """Generate a clean synthetic 12-lead ECG as a CSV blob."""
    fs = 500
    duration = 10.0
    t = np.arange(0, duration, 1 / fs)
    # Simple PQRST beats at 60 BPM.
    base = np.zeros_like(t)
    for beat_start in np.arange(0.3, duration - 0.3, 1.0):
        base += 0.15 * np.exp(-((t - beat_start) ** 2) / 0.04 ** 2)
        base -= 0.10 * np.exp(-((t - (beat_start + 0.15)) ** 2) / 0.01 ** 2)
        base += 1.00 * np.exp(-((t - (beat_start + 0.17)) ** 2) / 0.012 ** 2)
        base -= 0.20 * np.exp(-((t - (beat_start + 0.20)) ** 2) / 0.015 ** 2)
        base += 0.30 * np.exp(-((t - (beat_start + 0.35)) ** 2) / 0.04 ** 2)
    rng = np.random.default_rng(0)
    base += 0.02 * rng.standard_normal(len(t))

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    data = {name: base * (0.8 + i * 0.04) for i, name in enumerate(lead_names)}
    df = pd.DataFrame(data)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def flatline_ecg_csv() -> bytes:
    """CSV with all zeros — should trigger SignalQualityError (422)."""
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    df = pd.DataFrame({name: np.zeros(6000) for name in lead_names})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
