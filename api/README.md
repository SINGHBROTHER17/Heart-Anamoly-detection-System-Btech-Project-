# API — ECG Anomaly Detection FastAPI Service

Production REST API wrapping the preprocessing pipeline and trained model.

## Run locally

```bash
pip install -r requirements.txt

# Place your trained checkpoint here first:
#   ./model/calibrated_model.pt

# Then start the server from the project root:
cd ..
make api
# or: cd api && PYTHONPATH=..:. uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the OpenAPI Swagger UI.

## Docker

```bash
docker compose up --build
```

Mounts:
- `./model/` → `/app/model/` (read-only; place `calibrated_model.pt` here)
- `./data/`  → `/app/data/`  (writable; SQLite DB persisted here)

## Endpoints

| Method | Path                | Purpose                                |
|--------|---------------------|----------------------------------------|
| POST   | `/analyze`          | Multipart file upload (CSV)            |
| POST   | `/analyze/json`     | JSON per-lead payload                  |
| GET    | `/report/{id}`      | Retrieve stored report                 |
| GET    | `/reports`          | List recent reports (for history view) |
| POST   | `/feedback`         | Submit clinician feedback              |
| GET    | `/health`           | Health check (used by load balancers)  |
| GET    | `/docs`             | Swagger UI                             |
| GET    | `/openapi.json`     | OpenAPI schema                         |

Rate limit: 10 req/minute per IP on `/analyze*` endpoints.

## Tests

```bash
python -m pytest tests/ -v
```

19 integration tests using an auto-mocked model fixture (no checkpoint
needed for CI).

## Benchmark

```bash
python -m scripts.benchmark --n 50
python -m scripts.benchmark --n 100 --concurrency 4
```

Reports p50 / p90 / p95 / p99 latency, throughput, parameter count, model
size. Spec target: p95 < 500 ms per analysis.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ECG_CHECKPOINT_PATH` | `./model/calibrated_model.pt` | Model checkpoint to load |
| `ECG_DEVICE` | `cpu` (or `cuda` if available) | PyTorch device |
| `ECG_DB_PATH` | `./data/reports.db` | SQLite database path |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000,...` | Comma-separated |
| `ECG_DISABLE_RATE_LIMIT` | `0` | Set to `1` to disable rate limiting |
