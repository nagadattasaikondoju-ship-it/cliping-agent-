# CPU image: web UI + CLI.  docker compose up  ->  http://localhost:8000
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-noto-color-emoji fonts-inter fontconfig \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY clipagent ./clipagent
RUN pip install --no-cache-dir ".[all]"

ENV CLIPAGENT_DATA_DIR=/data \
    XDG_CACHE_HOME=/cache \
    XDG_CONFIG_HOME=/config \
    HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1
VOLUME ["/data", "/cache", "/config"]
EXPOSE 8000

ENTRYPOINT ["clipagent"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--data-dir", "/data"]
