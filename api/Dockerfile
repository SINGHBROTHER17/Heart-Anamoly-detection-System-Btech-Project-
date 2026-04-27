# Multi-stage Dockerfile for the ECG Anomaly Detection API.
# Stage 1 installs dependencies; stage 2 is the lean runtime image.

# ---- Stage 1: builder --------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build-time system deps only in the builder stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
# Install to a prefix so we can COPY just the packages into stage 2.
RUN pip install --prefix=/install -r requirements.txt \
    && pip install --prefix=/install torch --index-url https://download.pytorch.org/whl/cpu

# ---- Stage 2: runtime --------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.11/site-packages:/app

# Minimal runtime system deps (libgomp is needed by PyTorch).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring over installed Python packages from the builder.
COPY --from=builder /install /install

# Copy application code.
COPY api/app /app/app
COPY ecg_preprocessor /app/ecg_preprocessor
COPY ecg_model /app/ecg_model

# The model checkpoint is mounted at runtime (not baked in) — keeps the
# image small and lets you ship model updates without rebuilding.
ENV ECG_CHECKPOINT_PATH=/app/model/calibrated_model.pt \
    ECG_DB_PATH=/app/data/reports.db \
    ECG_DEVICE=cpu

RUN mkdir -p /app/model /app/data

EXPOSE 8000

# Use a non-root user for defence in depth.
RUN groupadd --system app && useradd --system --gid app app \
    && chown -R app:app /app
USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
