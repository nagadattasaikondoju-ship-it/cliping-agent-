# Hugging Face Spaces (Docker SDK): the public clipagent studio.
# Free "CPU basic" hardware: 2 vCPU, 16 GB RAM. The app listens on 7860.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-noto-color-emoji fonts-inter fontconfig \
 && rm -rf /var/lib/apt/lists/*

# Spaces run containers as UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    XDG_CACHE_HOME=/home/user/.cache \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    CLIPAGENT_PUBLIC=1
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user clipagent ./clipagent
RUN pip install --no-cache-dir --user ".[web]"

# Bake the models into the image so the first visitor doesn't wait for downloads.
RUN clipagent setup --whisper-model base && clipagent setup --whisper-model small

EXPOSE 7860
CMD ["clipagent", "serve", "--host", "0.0.0.0", "--port", "7860", "--data-dir", "/home/user/app/data"]
