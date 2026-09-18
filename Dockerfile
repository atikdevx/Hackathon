# GridWise LLM energy optimizer: production image.
# Build:  docker build -t gridwise-llm:1.0.0 .
# Run:    docker run --rm -p 8000:8000 -e LLM_API_KEY=<your-key> gridwise-llm:1.0.0
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Unprivileged runtime user; no secrets are baked in (configure via environment at run time).
RUN useradd --create-home --uid 10001 gridwise
USER gridwise

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=2)" || exit 1

# sh -c so $PORT (set by hosts such as Render/Railway/Cloud Run) is honored; exec keeps signals working.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --workers ${WEB_CONCURRENCY:-2}"]
