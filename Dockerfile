# syntax=docker/dockerfile:1

ARG UV_VERSION=0.12.5
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM --platform=linux/amd64 python:3.12-slim-bookworm AS builder

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:/usr/local/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY channels.toml ./
COPY src ./src
COPY static ./static
COPY models ./models

# Keep the editable project environment rooted at /app, matching chorus.models.ROOT.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

FROM --platform=linux/amd64 python:3.12-slim-bookworm AS runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:/usr/local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8000 \
    TTS_MODELS_DIRS=/models \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    XDG_CACHE_HOME=/cache \
    TORCH_HOME=/cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates espeak-ng ffmpeg libgomp1 libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 chorus \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/chorus --shell /usr/sbin/nologin chorus \
    && mkdir -p /app /models /cache \
    && chown -R 10001:10001 /app /models /cache /home/chorus

WORKDIR /app
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --from=builder --chown=10001:10001 /app/src /app/src
COPY --from=builder --chown=10001:10001 /app/static /app/static
COPY --from=builder --chown=10001:10001 /app/channels.toml /app/channels.toml
COPY --from=builder --chown=10001:10001 /app/models /app/models
# Isolated Breeze/Fish runtimes are intentionally not included in this image.
COPY --chown=10001:10001 docker/entrypoint.py /app/docker/entrypoint.py

VOLUME ["/models", "/cache"]
EXPOSE 8000

# Downloads happen before the API binds; allow a cold model volume to initialize.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

USER 10001:10001
ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
CMD ["--engines", "kokoro", "--devices", "cpu", "--download-missing"]
