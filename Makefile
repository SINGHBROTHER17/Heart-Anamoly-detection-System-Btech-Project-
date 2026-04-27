# Heart Anomaly Detection System — convenience targets.
#
# Usage:  make <target>
# Run `make help` for a list of available targets.

.DEFAULT_GOAL := help
.PHONY: help install install-api install-frontend test test-preprocessor test-api \
        api frontend benchmark docker-build docker-up docker-down clean

PY     ?= python3
PIP    ?= pip
NPM    ?= npm

# ---- Help ------------------------------------------------------------------
help:
	@echo "Heart Anomaly Detection System — Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install           Install both API and frontend dependencies"
	@echo "  make install-api       Install Python API dependencies"
	@echo "  make install-frontend  Install Node frontend dependencies"
	@echo ""
	@echo "Run:"
	@echo "  make api               Start API dev server on :8000"
	@echo "  make frontend          Start frontend dev server on :5173"
	@echo ""
	@echo "Test:"
	@echo "  make test              Run all test suites"
	@echo "  make test-preprocessor Run preprocessing tests"
	@echo "  make test-api          Run FastAPI integration tests"
	@echo ""
	@echo "Benchmark / docker:"
	@echo "  make benchmark         Measure inference latency"
	@echo "  make docker-build      Build API container"
	@echo "  make docker-up         docker compose up"
	@echo "  make docker-down       docker compose down"
	@echo ""
	@echo "  make clean             Remove caches and build artifacts"

# ---- Install ---------------------------------------------------------------
install: install-api install-frontend

install-api:
	$(PIP) install -r api/requirements.txt

install-frontend:
	cd frontend && $(NPM) install

# ---- Run -------------------------------------------------------------------
api:
	cd api && PYTHONPATH=..:. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && $(NPM) run dev

# ---- Test ------------------------------------------------------------------
test: test-preprocessor test-api

test-preprocessor:
	$(PY) -m pytest tests_preprocessor/ -v

test-api:
	cd api && $(PY) -m pytest tests/ -v

# ---- Benchmark / model ------------------------------------------------------
benchmark:
	cd api && PYTHONPATH=..:. $(PY) -m scripts.benchmark --n 50

# ---- Docker ----------------------------------------------------------------
docker-build:
	cd api && docker compose build

docker-up:
	cd api && docker compose up

docker-down:
	cd api && docker compose down

# ---- Clean -----------------------------------------------------------------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf frontend/dist frontend/.vite
	rm -rf build dist *.egg-info
	@echo "Cleaned."
