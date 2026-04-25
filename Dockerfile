# Travel Care AI — FastAPI + static UI + local corpus
# Cloud Run: set secrets/env vars; platform provides PORT (default 8080 for local docker run).

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY static/ static/
COPY data/ data/
COPY .env.example .env.example

# Non-root user; writable logs dir for default file logging
RUN useradd --create-home --user-group --shell /usr/sbin/nologin --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# Prefer console-only logs in containers unless you mount a volume and set LOG_FILE
ENV LOG_DISABLE_FILE=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; p=os.environ.get('PORT','8080'); urllib.request.urlopen(f'http://127.0.0.1:{p}/api/health', timeout=4).read(64)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
