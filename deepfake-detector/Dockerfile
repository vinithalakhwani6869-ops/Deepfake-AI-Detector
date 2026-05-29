# Production-hardened multi-stage Dockerfile for deepfake detection API
# Build: docker build -t deepfake-detector:latest .
# Run: docker run --gpus all -p 8000:8000 -v ./checkpoints:/app/checkpoints:ro deepfake-detector:latest

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Builder
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim as builder

# Install system build dependencies (minimal: only what pip needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python environment
WORKDIR /build

COPY requirements.txt .

# Build wheels for all dependencies (enables faster installation in runtime stage)
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Runtime (minimal image)
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 `# OpenMP for PyTorch` \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (prevents container escalation)
RUN groupadd -r deepfake && useradd -r -g deepfake deepfake

# Set working directory
WORKDIR /app

# Copy wheels from builder stage (faster than pip install)
COPY --from=builder /wheels /wheels
COPY --from=builder /build/requirements.txt .

# Install Python dependencies from wheels (no network access needed)
RUN pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt

# Copy application code
COPY --chown=deepfake:deepfake . .

# Set Python environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create directories for logs and temp uploads with proper permissions
RUN mkdir -p /app/logs /app/temp_uploads && \
    chown -R deepfake:deepfake /app/logs /app/temp_uploads

# Switch to non-root user
USER deepfake

# Health check (queries /health endpoint every 30s, 3 consecutive failures = unhealthy)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Expose API port
EXPOSE 8000

# Production startup: disable reload, bind all interfaces, enable access logs
# Note: Use --host 0.0.0.0 in containers; FastAPI serves on 127.0.0.1:8000 by default
CMD ["uvicorn", \
     "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
