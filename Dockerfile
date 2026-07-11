FROM python:3.12-slim

# OS deps: build-essential for any wheels that compile, libgomp1 for onnxruntime,
# curl for the healthcheck, ca-certificates for TLS (fastembed/Supabase/Groq).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN useradd --create-home --uid 10001 navi

ENV ARROW_DEFAULT_MEMORY_POOL=system \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8600 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code (respects .dockerignore). models/ is NOT ignored and ships
# with the image (needed by the ONNX summarizer).
COPY . .

RUN chown -R navi:navi /app
USER navi

EXPOSE 8600

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8600/_stcore/health || exit 1

CMD ["streamlit", "run", "Home.py", \
     "--server.port", "8600", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]
