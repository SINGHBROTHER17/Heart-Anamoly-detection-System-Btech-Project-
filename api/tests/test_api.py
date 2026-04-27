"""Integration tests for all FastAPI endpoints."""

from __future__ import annotations

import io


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "degraded")
        assert body["model_loaded"] is True
        assert body["uptime_seconds"] >= 0
        assert "version" in body

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["name"] == "ECG Anomaly Detection API"


# ---------------------------------------------------------------------------
# /analyze — CSV upload path
# ---------------------------------------------------------------------------

class TestAnalyzeCSV:

    def test_happy_path(self, client, synth_ecg_csv):
        r = client.post(
            "/analyze",
            files={"file": ("ecg.csv", synth_ecg_csv, "text/csv")},
            data={"sample_rate": 500},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "report_id" in body
        assert 0.0 <= body["signal_quality"] <= 1.0
        assert len(body["results"]) == 10
        for cond in body["results"]:
            assert 0.0 <= cond["confidence"] <= 1.0
            assert cond["risk_tier"] in ("none", "possible", "likely", "high")
            assert "tier_label" in cond
        assert "disclaimer" in body
        assert len(body["per_lead_quality"]) == 12

    def test_response_headers_request_id(self, client, synth_ecg_csv):
        r = client.post(
            "/analyze",
            files={"file": ("ecg.csv", synth_ecg_csv, "text/csv")},
        )
        assert "X-Request-ID" in r.headers

    def test_rejects_flatline(self, client, flatline_ecg_csv):
        r = client.post(
            "/analyze",
            files={"file": ("flat.csv", flatline_ecg_csv, "text/csv")},
        )
        assert r.status_code == 422
        body = r.json()
        assert body["error"] == "signal_quality_too_low"
        assert "per_lead_sqi" in body

    def test_rejects_non_csv(self, client):
        r = client.post(
            "/analyze",
            files={"file": ("ecg.txt", b"not a csv", "text/plain")},
        )
        assert r.status_code == 400

    def test_rejects_empty_csv(self, client):
        r = client.post(
            "/analyze",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert r.status_code == 400

    def test_rejects_csv_with_no_recognized_columns(self, client):
        bad_csv = b"foo,bar\n1,2\n3,4\n"
        r = client.post(
            "/analyze",
            files={"file": ("bad.csv", bad_csv, "text/csv")},
        )
        assert r.status_code == 400

    def test_missing_file(self, client):
        r = client.post("/analyze")
        assert r.status_code == 422  # FastAPI's "field required" for multipart


# ---------------------------------------------------------------------------
# /analyze/json — JSON payload path
# ---------------------------------------------------------------------------

class TestAnalyzeJSON:

    def test_json_payload(self, client):
        import numpy as np
        fs = 500
        t = np.arange(0, 10.0, 1 / fs)
        base = np.zeros_like(t)
        for beat_start in np.arange(0.3, 9.7, 1.0):
            base += 1.0 * np.exp(-((t - (beat_start + 0.17)) ** 2) / 0.012 ** 2)
            base += 0.3 * np.exp(-((t - (beat_start + 0.35)) ** 2) / 0.04 ** 2)
        samples = (base + 0.02 * np.random.default_rng(0).standard_normal(len(t))).tolist()

        leads = [
            {"lead_name": name, "samples": samples, "sample_rate": fs}
            for name in ["I", "II", "III", "aVR", "aVL", "aVF",
                         "V1", "V2", "V3", "V4", "V5", "V6"]
        ]
        r = client.post(
            "/analyze/json",
            json={"leads": leads, "patient_id": "pt_123"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["results"]) == 10

    def test_json_invalid_lead_name(self, client):
        r = client.post(
            "/analyze/json",
            json={"leads": [{"lead_name": "ZZ", "samples": [0.0] * 1000,
                             "sample_rate": 500}]},
        )
        assert r.status_code == 422  # Pydantic validation

    def test_json_samples_too_short(self, client):
        r = client.post(
            "/analyze/json",
            json={"leads": [{"lead_name": "I", "samples": [0.0] * 100,
                             "sample_rate": 500}]},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /report/{id} and /reports
# ---------------------------------------------------------------------------

class TestReports:

    def test_get_report_roundtrip(self, client, synth_ecg_csv):
        # Create a report.
        r = client.post("/analyze",
                        files={"file": ("ecg.csv", synth_ecg_csv, "text/csv")})
        assert r.status_code == 200
        report_id = r.json()["report_id"]

        # Fetch it back.
        r2 = client.get(f"/report/{report_id}")
        assert r2.status_code == 200
        assert r2.json()["report_id"] == report_id

    def test_get_missing_report_404(self, client):
        r = client.get("/report/nonexistent-id")
        assert r.status_code == 404

    def test_list_reports(self, client, synth_ecg_csv):
        # Generate a couple of reports.
        for _ in range(3):
            client.post("/analyze",
                        files={"file": ("ecg.csv", synth_ecg_csv, "text/csv")})
        r = client.get("/reports?limit=10")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 3
        assert len(body["reports"]) == 3

    def test_list_reports_invalid_limit(self, client):
        r = client.get("/reports?limit=0")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------

class TestFeedback:

    def test_feedback_happy(self, client, synth_ecg_csv):
        # Make a report first.
        r = client.post("/analyze",
                        files={"file": ("ecg.csv", synth_ecg_csv, "text/csv")})
        report_id = r.json()["report_id"]

        # Submit feedback.
        r2 = client.post("/feedback", json={
            "report_id": report_id,
            "clinician_id": "dr_smith",
            "correct_conditions": ["Normal Sinus Rhythm"],
            "incorrect_conditions": ["Atrial Fibrillation"],
            "missed_conditions": [],
            "notes": "Patient has history of anxiety; rhythm is normal."
        })
        assert r2.status_code == 200
        body = r2.json()
        assert body["status"] == "recorded"
        assert body["report_id"] == report_id

    def test_feedback_unknown_report(self, client):
        r = client.post("/feedback", json={
            "report_id": "does-not-exist",
            "correct_conditions": [],
            "incorrect_conditions": [],
            "missed_conditions": [],
        })
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

class TestCORS:

    def test_preflight(self, client):
        r = client.options(
            "/analyze",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # CORSMiddleware returns 200 on preflight.
        assert r.status_code in (200, 204)
        assert "access-control-allow-origin" in {k.lower() for k in r.headers}
