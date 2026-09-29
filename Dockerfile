# Multi-stage secure production build for SSIF Microservice
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install security updates and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned dependencies
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r requirements.lock

# Copy project source code
COPY . /app

# Create non-root system user for least privilege execution (CWE-250)
RUN useradd -m -u 10001 ssifuser && \
    chown -R ssifuser:ssifuser /app
USER ssifuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
