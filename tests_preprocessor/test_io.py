"""Unit tests for I/O adapters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_preprocessor.constants import LEAD_INDEX, LEAD_ORDER, TARGET_FS
from ecg_preprocessor.exceptions import InvalidInputError
from ecg_preprocessor.io import combine_lead_payloads, load_csv, load_json


FS = 500


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TestLoadCsv:

    def _make_csv(self, tmp_path: Path, leads: list[str], n_samples: int = 5000) -> Path:
        rng = np.random.default_rng(0)
        data = {l: rng.standard_normal(n_samples) for l in leads}
        p = tmp_path / "ecg.csv"
        pd.DataFrame(data).to_csv(p, index=False)
        return p

    def test_loads_full_12lead(self, tmp_path):
        p = self._make_csv(tmp_path, list(LEAD_ORDER))
        sig, fs = load_csv(p, sample_rate=FS)
        assert sig.shape == (12, 5000)
        assert fs == TARGET_FS

    def test_lead_order_respected(self, tmp_path):
        p = self._make_csv(tmp_path, list(LEAD_ORDER))
        df = pd.read_csv(p)
        sig, _ = load_csv(p, sample_rate=FS)
        # Row LEAD_INDEX["V3"] must equal column "V3".
        assert np.allclose(sig[LEAD_INDEX["V3"]], df["V3"].values, atol=1e-5)

    def test_partial_leads_zero_filled(self, tmp_path):
        p = self._make_csv(tmp_path, ["I", "II", "V1"])
        sig, _ = load_csv(p, sample_rate=FS)
        assert sig.shape == (12, 5000)
        # Missing leads -> zero row.
        assert np.all(sig[LEAD_INDEX["aVR"]] == 0)
        assert np.all(sig[LEAD_INDEX["V6"]] == 0)

    def test_case_insensitive_columns(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "i": rng.standard_normal(1000),
            "ii": rng.standard_normal(1000),
            "v1": rng.standard_normal(1000),
        })
        p = tmp_path / "mixed.csv"
        df.to_csv(p, index=False)
        sig, _ = load_csv(p, sample_rate=FS)
        assert sig.shape == (12, 1000)
        # "i" -> "I", and data should land in row 0.
        assert np.allclose(sig[LEAD_INDEX["I"]], df["i"].values, atol=1e-5)

    def test_resamples_to_target(self, tmp_path):
        # Create 250 Hz data of 8 seconds -> 2000 samples.
        rng = np.random.default_rng(0)
        df = pd.DataFrame({l: rng.standard_normal(2000) for l in LEAD_ORDER})
        p = tmp_path / "slow.csv"
        df.to_csv(p, index=False)
        sig, fs = load_csv(p, sample_rate=250)
        assert fs == TARGET_FS
        # 8 s * 500 Hz -> 4000 samples.
        assert sig.shape == (12, 4000)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(InvalidInputError):
            load_csv(tmp_path / "does_not_exist.csv")

    def test_empty_csv_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(InvalidInputError):
            load_csv(p)

    def test_unrecognized_columns_raises(self, tmp_path):
        df = pd.DataFrame({"foo": np.arange(100), "bar": np.arange(100)})
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(InvalidInputError):
            load_csv(p)

    def test_nans_replaced_with_zero(self, tmp_path):
        df = pd.DataFrame({l: np.arange(100, dtype=float) for l in LEAD_ORDER})
        df.loc[10:20, "I"] = np.nan
        p = tmp_path / "nans.csv"
        df.to_csv(p, index=False)
        sig, _ = load_csv(p, sample_rate=FS)
        assert not np.isnan(sig).any()


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

class TestLoadJson:

    def test_dict_payload(self):
        payload = {
            "lead_name": "V1",
            "samples": list(np.random.randn(5000)),
            "sample_rate": FS,
        }
        samples, fs, name = load_json(payload)
        assert samples.shape == (5000,)
        assert fs == TARGET_FS
        assert name == "V1"

    def test_string_payload(self):
        payload = json.dumps({
            "lead_name": "II",
            "samples": [1.0, 2.0, 3.0, 4.0, 5.0] * 200,
            "sample_rate": FS,
        })
        samples, fs, name = load_json(payload)
        assert samples.size == 1000
        assert name == "II"

    def test_file_payload(self, tmp_path):
        p = tmp_path / "lead_v2.json"
        p.write_text(json.dumps({
            "lead_name": "V2",
            "samples": [0.0] * 500,
            "sample_rate": FS,
        }))
        samples, fs, name = load_json(p)
        assert name == "V2"
        assert samples.size == 500

    def test_resamples(self):
        payload = {
            "lead_name": "I",
            "samples": list(np.random.randn(1000)),   # 4 s at 250 Hz
            "sample_rate": 250,
        }
        samples, fs, _ = load_json(payload)
        assert fs == TARGET_FS
        assert samples.size == 2000   # 4 s at 500 Hz

    def test_case_insensitive_lead_name(self):
        payload = {"lead_name": "v1", "samples": [1.0] * 500, "sample_rate": FS}
        _, _, name = load_json(payload)
        assert name == "V1"

    def test_unknown_lead_raises(self):
        payload = {"lead_name": "V99", "samples": [1.0] * 500, "sample_rate": FS}
        with pytest.raises(InvalidInputError):
            load_json(payload)

    def test_missing_field_raises(self):
        with pytest.raises(InvalidInputError):
            load_json({"lead_name": "I", "sample_rate": FS})

    def test_empty_samples_raises(self):
        with pytest.raises(InvalidInputError):
            load_json({"lead_name": "I", "samples": [], "sample_rate": FS})

    def test_multidim_samples_raises(self):
        payload = {
            "lead_name": "I",
            "samples": [[1.0, 2.0], [3.0, 4.0]],
            "sample_rate": FS,
        }
        with pytest.raises(InvalidInputError):
            load_json(payload)

    def test_bad_json_string_raises(self):
        with pytest.raises(InvalidInputError):
            load_json("{ not valid json")


# ---------------------------------------------------------------------------
# combine_lead_payloads
# ---------------------------------------------------------------------------

class TestCombineLeadPayloads:

    def test_combines_three_leads(self):
        rng = np.random.default_rng(0)
        payloads = [
            {"lead_name": l, "samples": list(rng.standard_normal(5000)), "sample_rate": FS}
            for l in ["I", "II", "V1"]
        ]
        sig, fs = combine_lead_payloads(payloads)
        assert sig.shape == (12, 5000)
        assert fs == TARGET_FS
        # Leads not supplied stay zero.
        assert np.all(sig[LEAD_INDEX["V6"]] == 0)
        # Supplied leads are non-zero.
        assert sig[LEAD_INDEX["I"]].std() > 0

    def test_strict_mode_raises_on_missing(self):
        payloads = [{"lead_name": "I", "samples": [1.0] * 500, "sample_rate": FS}]
        with pytest.raises(InvalidInputError):
            combine_lead_payloads(payloads, strict=True)

    def test_last_write_wins_on_duplicate_lead(self):
        first = {"lead_name": "I", "samples": [1.0] * 500, "sample_rate": FS}
        second = {"lead_name": "I", "samples": [2.0] * 500, "sample_rate": FS}
        sig, _ = combine_lead_payloads([first, second])
        assert np.allclose(sig[LEAD_INDEX["I"]], 2.0)

    def test_empty_input_raises(self):
        with pytest.raises(InvalidInputError):
            combine_lead_payloads([])
