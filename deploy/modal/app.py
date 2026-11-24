"""The public clipagent studio on Modal (https://modal.com).

    pip install modal && modal setup        # once, logs you in
    modal deploy deploy/modal/app.py        # prints the public URL

Or let GitHub do it: add MODAL_TOKEN_ID and MODAL_TOKEN_SECRET as repository
secrets and .github/workflows/deploy-modal.yml deploys on every push to main.

One container (2 cores, 4 GB RAM) serves everyone and scales to zero when
idle. Projects live on a Modal Volume and expire after 24 hours (public mode).
Optional: put GEMINI_API_KEY or ANTHROPIC_API_KEY in a Modal Secret and set
CLIPAGENT_MODAL_SECRETS=<secret name> when deploying.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import modal

# The repo root, only needed on your computer where the image is built. Inside the
# container this file lives at /root/app.py, which has no grandparent folder.
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "fonts-noto-color-emoji", "fonts-inter", "fontconfig")
    .add_local_file(ROOT / "pyproject.toml", "/src/pyproject.toml", copy=True)
    .add_local_file(ROOT / "README.md", "/src/README.md", copy=True)
    .add_local_dir(ROOT / "clipagent", "/src/clipagent", copy=True,
                   ignore=["**/__pycache__", "**/*.pyc"])
    .run_commands(
        "pip install '/src[web]'",
        # Bake the models into the image so the first visitor doesn't wait.
        "clipagent setup --whisper-model base",
        "clipagent setup --whisper-model small",
    )
    .env({
        "CLIPAGENT_PUBLIC": "1",
        "CLIPAGENT_PUBLIC_URL": os.environ.get("CLIPAGENT_PUBLIC_URL", "https://clipagent-ten.vercel.app"),
        "PYTHONUNBUFFERED": "1",
    })
)

data = modal.Volume.from_name("clipagent-data", create_if_missing=True)
secrets = [modal.Secret.from_name(name.strip())
           for name in os.environ.get("CLIPAGENT_MODAL_SECRETS", "").split(",") if name.strip()]

app = modal.App("clipagent")


@app.function(
    image=image,
    volumes={"/data": data},
    secrets=secrets,
    cpu=2.0,
    memory=4096,
    max_containers=1,       # one shared queue and job store
    scaledown_window=600,   # stay up 10 min after the last request (jobs run in the background)
    timeout=3600,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from clipagent.web.server import create_app

    def keep_volume_in_sync() -> None:
        while True:
            time.sleep(30)
            try:
                data.commit()
            except Exception as exc:  # never let syncing kill the server
                print(f"volume commit failed: {exc}")

    threading.Thread(target=keep_volume_in_sync, daemon=True).start()
    return create_app("/data")
