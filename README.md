# ❤️ Heart Anomaly Detection System

End-to-end clinical decision support for 12-lead ECG screening.

Reconstructs a 12-lead ECG from sequentially recorded leads, runs it through a
trained 1D-CNN + Transformer model, and serves **calibrated confidence scores**
(not hard classifications) for 10 cardiac conditions through a REST API and a
mobile-first web app.

> **This is a screening tool, not a diagnostic device.** Every response from
> the system carries a disclaimer to that effect.

---

## What's in this project

| Phase | What it does | Where it lives |
|-------|--------------|----------------|
| 2 — Preprocessing pipeline | Turns raw sequentially-recorded ECG into a clean, aligned, normalized `(12, 5000)` tensor | `ecg_preprocessor/`, `tests_preprocessor/`, `examples/` |
| 3 — ML model training | Hybrid 1D-CNN + Transformer trained on PTB-XL with temperature-scaled calibration | `ecg_model/`, `ECG_Training_Colab.ipynb` |
| 4 — Cloud REST API | FastAPI service with `/analyze`, `/report/{id}`, `/feedback`, `/health` | `api/` |
| 5 — Mobile-first web app | React + Vite + Tailwind, interactive 12-lead viewer, demo mode, history | `frontend/` |

The 10 conditions the system screens for:
`Normal Sinus Rhythm`, `Atrial Fibrillation`, `ST Elevation`, `Left Bundle
Branch Block`, `Right Bundle Branch Block`, `Left Ventricular Hypertrophy`,
`Bradycardia`, `Tachycardia`, `First Degree AV Block`, `Premature Ventricular
Contraction`.

---

## Repository layout

```
heart_anomaly_system/
├── ecg_preprocessor/          Phase 2 - Python preprocessing package
│   ├── pipeline.py                main orchestrator (ECGPreprocessor)
│   ├── filters.py                 bandpass / notch / baseline / zscore
│   ├── quality.py                 Signal Quality Index (4-component)
│   ├── peaks.py                   Pan-Tompkins R-peak detector
│   ├── alignment.py               R-peak position-matching alignment
│   ├── segmentation.py            fixed-window crop/pad
│   ├── io.py                      CSV / JSON loaders
│   └── ...
├── tests_preprocessor/         105 pytest tests for Phase 2
├── examples/demo.py            Runnable Phase 2 demo
│
├── ecg_model/                 Phase 3 - ML training package
│   ├── model.py                    architecture + calibration
│   ├── dataset.py                  PTB-XL loader + augmentations
│   ├── train.py                    training loop
│   └── evaluate.py                 test-set evaluation + plots
├── ECG_Training_Colab.ipynb   Colab notebook - runs end-to-end on T4 GPU
│
├── api/                       Phase 4 - FastAPI service
│   ├── app/
│   │   ├── main.py                    endpoints + middleware
│   │   ├── schemas.py                 Pydantic models (+ risk-tier mapping)
│   │   ├── model_loader.py            singleton model loaded at startup
│   │   ├── service.py                 preprocessing + inference glue
│   │   ├── storage.py                 SQLite report/feedback storage
│   │   ├── logging_config.py          structlog JSON + correlation IDs
│   │   └── rate_limit.py              slowapi 10 req/min limiter
│   ├── tests/                    19 integration tests (mock model)
│   ├── scripts/benchmark.py      latency / throughput measurement
│   ├── Dockerfile                multi-stage, non-root, healthcheck
│   └── docker-compose.yml        local dev with volume mounts
│
├── frontend/                  Phase 5 - React + Vite + Tailwind
│   ├── src/
│   │   ├── pages/                     UploadPage, ReportPage, HistoryPage
│   │   ├── components/                Layout, EcgPlot, ConditionCard, ...
│   │   ├── services/api.js            axios client, error normalization
│   │   ├── utils/                     risk tiers, synthetic ECG generator
│   │   └── hooks/useDarkMode.js
│   ├── vite.config.js                dev proxy to /api
│   └── tailwind.config.js
│
├── .vscode/                   VS Code workspace (settings / launch / tasks)
├── Makefile                   Common targets (make api, make test, ...)
├── pytest.ini                 Top-level test config
├── .env.example               Environment variables template
└── README.md                  This file
```

---

## Quickstart

### 1. Install

```bash
# Everything: API Python deps + frontend Node deps
make install

# Or manually:
pip install -r api/requirements.txt
cd frontend && npm install && cd ..
```

### 2. Get a trained model

You need `calibrated_model.pt` in `api/model/` before the API can do real
inference. Two options:

**A) Train one yourself (best results, ~2-4 hours on Colab T4 GPU):**

1. Open `ECG_Training_Colab.ipynb` in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Run all cells — downloads PTB-XL (~3 GB), trains, calibrates, saves to Drive.
4. Download the resulting `calibrated_model.pt` from your Drive.
5. Place it at `api/model/calibrated_model.pt`.

**B) Run the API without a real model for local development.**
The integration tests use a mock model via `conftest.py`. The API will
report `status: degraded` at `/health` but won't crash. This is enough
to develop the frontend against.

### 3. Run it

Two shells:

```bash
# Terminal 1
make api            # -> http://localhost:8000
                     # API docs at http://localhost:8000/docs

# Terminal 2
make frontend       # -> http://localhost:5173
```

The Vite dev proxy forwards `/api/*` to `localhost:8000`, so the frontend
hits same-origin URLs.

### 4. Test it

```bash
make test                    # Runs all 124 tests (105 preprocessing + 19 API)
make test-preprocessor       # Just Phase 2
make test-api                # Just Phase 4
make benchmark               # Measure inference latency
```

---

## VS Code

Open the project root in VS Code. Recommended extensions are pinned in
`.vscode/extensions.json` — VS Code will prompt to install them.

**One-click debugging** (`Run → Start Debugging`):

- `API: run uvicorn (debug)` — breakpoints in API code work
- `API: pytest` — debug a failing API test
- `Preprocessor: pytest` — debug a failing preprocessing test
- `Preprocessor: run demo` — step through the end-to-end demo
- `Benchmark: latency` — profile inference
- `Frontend: Chrome against local dev server` — debug JSX in Chrome

**One-click tasks** (`Terminal → Run Task`):

- `API: run dev server`, `Frontend: run dev server`
- `Full stack: API + Frontend (parallel)` — starts both at once
- `API: run tests`, `Preprocessor: run tests`
- `API: docker compose up`

---

## The data flow

```
┌─────────────┐    ┌─────────────────────┐    ┌──────────┐    ┌──────────────┐
│ ECG input   │ -> │  Preprocessing      │ -> │ ML Model │ -> │ Risk tiers + │
│ (CSV/JSON)  │    │  Phase 2            │    │ Phase 3  │    │ interpretation│
└─────────────┘    ├─────────────────────┤    └──────────┘    └──────┬───────┘
                   │ bandpass + notch    │                           │
                   │ baseline wander     │                           │
                   │ SQI gate (< 0.60)   │                           │
                   │ R-peak alignment    │                           │
                   │ segment to 5000     │                           │
                   │ z-score             │                           │
                   └─────────────────────┘                           │
                                                                      │
                                                   ┌──────────────────┘
                                                   ▼
                                          ┌────────────────────┐
                                          │ FastAPI (Phase 4)  │
                                          │ - /analyze         │
                                          │ - /report/{id}     │
                                          │ - /feedback        │
                                          │ - /health          │
                                          └────────┬───────────┘
                                                   │
                                                   ▼
                                          ┌────────────────────┐
                                          │ React app          │
                                          │ (Phase 5)          │
                                          │ - Upload           │
                                          │ - 12-lead viewer   │
                                          │ - Condition cards  │
                                          │ - History          │
                                          └────────────────────┘
```

---

## Risk tier mapping (exact, per spec)

| Confidence      | Tier       | Label                                              |
|-----------------|------------|----------------------------------------------------|
| `< 0.30`        | `none`     | No significant finding detected                    |
| `0.30 – 0.59`   | `possible` | Possible finding — monitoring recommended          |
| `0.60 – 0.84`   | `likely`   | Likely finding — please consult a physician        |
| `≥ 0.85`        | `high`     | Strong indicator — seek medical attention promptly |

Defined in `api/app/schemas.py::confidence_to_tier()` and mirrored in
`frontend/src/utils/riskTiers.js`.

---

## Deployment notes

- **Cloud Run / Kubernetes**: the `Dockerfile` is CPU-only by default;
  override with `ECG_DEVICE=cuda` if you deploy to GPU infrastructure.
- **Persistent storage**: the default SQLite backend is fine for
  single-instance deployments. For Cloud Run autoscaling, swap
  `SQLiteStorage` for a Firestore/Postgres implementation — the interface
  is only 4 methods (`save_report`, `get_report`, `list_recent_reports`,
  `save_feedback`), so implementation is a drop-in.
- **Model updates**: checkpoint is mounted at runtime, not baked into the
  image. Ship new models without rebuilding the container.
- **CORS**: set `CORS_ALLOW_ORIGINS` env var to your production frontend
  origin (comma-separated for multiple).
- **Rate limiting**: 10 requests/minute per IP on `/analyze` and
  `/analyze/json`. Adjust in `api/app/main.py::analyze`.

---

## What's not included (intentionally)

- **Hardware ingestion (ESP32/ADS1292)** — out of scope per spec; we consume
  CSV or JSON.
- **PDF report generation** — the frontend offers "Download report (JSON)".
  Wire up a PDF renderer (e.g. `@react-pdf/renderer`) if that's a requirement.
- **Authentication** — the API is open. For production, add OAuth / API keys
  via FastAPI `Depends(...)` middleware; the routes are already structured
  for it.
- **EHR integration** — the `/feedback` endpoint stores clinician feedback
  locally for future fine-tuning. Connect it to your EHR if needed.

---

## Key design decisions

- **Multi-label, not multi-class.** A patient can simultaneously have
  multiple conditions, so output is 10 independent sigmoids, not a softmax.
- **Post-hoc temperature scaling** for calibration. Simplest method that
  reliably gets ECE < 0.05 on held-out data without affecting accuracy.
- **R-peak *position* matching** for inter-lead alignment (not waveform
  cross-correlation). Lead-to-lead morphology differences are *information*
  in a 12-lead ECG; only time information is shared.
- **Geometric mean SQI** across leads. A single bad electrode drags the
  overall score down instead of being averaged away.
- **SQLite for storage** in the API. Zero setup for dev and small prod;
  drop-in interface for Firestore or Postgres for scale.
- **SQI gate at 0.60** before inference. No point running a model on
  electrode-off signal; fail fast with a per-lead breakdown so the UX
  can say "re-record V2" instead of a generic error.

---

## License & disclaimer

Research and educational use only. Not FDA cleared. Not CE marked. Not a
substitute for a qualified physician's judgment.
