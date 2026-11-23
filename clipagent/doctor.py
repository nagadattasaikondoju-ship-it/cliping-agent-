"""`clipagent doctor` (check requirements) and `clipagent setup` (download models)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from .env import CACHE_DIR

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""

    def to_dict(self) -> dict:
        return dict(name=self.name, status=self.status, detail=self.detail, fix=self.fix)


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _ffmpeg_checks() -> list[Check]:
    out = []
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return [Check("ffmpeg", FAIL, "not found on PATH",
                      "macOS: brew install ffmpeg · Ubuntu/Debian: sudo apt install ffmpeg · "
                      "Windows: winget install Gyan.FFmpeg")]
    ver = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    out.append(Check("ffmpeg", OK, ver.replace("ffmpeg version ", "").split(" Copyright")[0]))
    filters = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    if " ass " in filters:
        out.append(Check("ffmpeg libass (captions)", OK, "ass filter available"))
    else:
        out.append(Check("ffmpeg libass (captions)", FAIL, "ffmpeg was built without libass",
                         "Install a full ffmpeg build (the apt/brew packages include libass)"))
    if not shutil.which("ffprobe"):
        out.append(Check("ffprobe", FAIL, "not found", "ffprobe ships with ffmpeg; reinstall ffmpeg"))
    return out


def _whisper_cached(size: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        repo = size if "/" in size else f"Systran/faster-whisper-{size}"
        hit = try_to_load_from_cache(repo, "model.bin")
        return isinstance(hit, str)
    except Exception:
        return False


def gpu_status() -> tuple[bool, str]:
    try:
        import ctranslate2  # type: ignore

        n = ctranslate2.get_cuda_device_count()
        if n > 0:
            return True, f"{n} CUDA device(s) usable by Whisper"
    except Exception:
        pass
    return False, "no CUDA GPU - transcription runs on CPU (about real time with 'small')"


def run_checks(whisper_model: str = "small") -> list[Check]:
    checks: list[Check] = []
    v = sys.version_info
    checks.append(Check("Python", OK if v >= (3, 10) else FAIL, f"{v.major}.{v.minor}.{v.micro}",
                        "" if v >= (3, 10) else "Install Python 3.10 or newer"))
    checks += _ffmpeg_checks()

    for mod, pkg, need in [("numpy", "numpy", True), ("cv2", "opencv-python-headless", True),
                           ("faster_whisper", "faster-whisper", True), ("anthropic", "anthropic", True),
                           ("yt_dlp", "yt-dlp", True), ("PIL", "pillow", True),
                           ("fastapi", "fastapi (web UI)", False),
                           ("googleapiclient", "google-api-python-client (YouTube posting)", False)]:
        if _has(mod):
            checks.append(Check(f"package {pkg.split()[0]}", OK, "installed"))
        else:
            checks.append(Check(f"package {pkg.split()[0]}", FAIL if need else WARN, "missing",
                                "pip install -e ." if need else 'pip install -e ".[web,publish]"'))

    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        checks.append(Check("Claude API key", OK, "ANTHROPIC_API_KEY is set"))
    else:
        checks.append(Check("Claude API key", WARN,
                            "not set - clip picks fall back to the rule-based scorer",
                            "Create a key at https://console.anthropic.com and put "
                            "ANTHROPIC_API_KEY=... in .env"))

    gpu, detail = gpu_status()
    checks.append(Check("GPU", OK if gpu else WARN, detail,
                        "" if gpu else "Optional: an NVIDIA GPU with CUDA 12 + cuDNN 9 makes "
                        "transcription 5-10x faster (or use the GPU Docker image)"))

    if _whisper_cached(whisper_model):
        checks.append(Check(f"Whisper model '{whisper_model}'", OK, "downloaded"))
    else:
        checks.append(Check(f"Whisper model '{whisper_model}'", WARN, "not downloaded yet",
                            "Run `clipagent setup` (or it downloads on first use)"))
    face = CACHE_DIR / "face_detection_yunet_2023mar.onnx"
    checks.append(Check("Face model (YuNet)", OK if face.exists() else WARN,
                        "downloaded" if face.exists() else "not downloaded yet",
                        "" if face.exists() else "Run `clipagent setup`"))

    from .emoji import emoji_backend

    backend = emoji_backend()
    checks.append(Check("Emoji renderer", OK if backend else WARN, backend or "none available",
                        "" if backend else "Install the Noto Color Emoji font or allow internet "
                        "access to cdn.jsdelivr.net (Twemoji)"))

    if os.environ.get("PEXELS_API_KEY"):
        checks.append(Check("B-roll (Pexels)", OK, "PEXELS_API_KEY is set"))
    else:
        checks.append(Check("B-roll (Pexels)", WARN, "PEXELS_API_KEY not set - only --broll-dir works",
                            "Free key at https://www.pexels.com/api/"))

    if os.environ.get("YTDLP_COOKIES") or os.environ.get("YTDLP_COOKIES_FROM_BROWSER"):
        checks.append(Check("YouTube cookies", OK, "configured"))
    else:
        checks.append(Check("YouTube cookies", WARN, "not configured (only needed for YouTube URLs "
                            "on servers)", "Pass --cookies-from-browser chrome, or set "
                            "YTDLP_COOKIES=/path/cookies.txt"))

    from .publish import publisher_status

    for name, (ready, detail) in publisher_status().items():
        checks.append(Check(f"Posting: {name}", OK if ready else WARN, detail))
    return checks


def print_report(checks: list[Check]) -> int:
    icons = {OK: "\033[32m✔\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✘\033[0m"}
    for c in checks:
        print(f" {icons[c.status]} {c.name:<32} {c.detail}")
        if c.fix and c.status != OK:
            print(f"   {'':<32} → {c.fix}")
    fails = sum(c.status == FAIL for c in checks)
    warns = sum(c.status == WARN for c in checks)
    print(f"\n{fails} problem(s), {warns} optional item(s) missing.")
    return 1 if fails else 0


def setup(whisper_model: str = "small", log=print) -> int:
    """Download the Whisper and face models ahead of time."""
    from .faces import _yunet_model

    log("Downloading face model (YuNet, ~230 KB)...")
    log("  ok" if _yunet_model() else "  FAILED - check internet access to github.com")
    log(f"Downloading Whisper model '{whisper_model}' (small ≈ 480 MB, large-v3 ≈ 3 GB)...")
    try:
        from faster_whisper import download_model

        path = download_model(whisper_model)
        log(f"  ok: {path}")
    except Exception as exc:
        log(f"  FAILED: {exc}")
        return 1
    from .emoji import render_emoji

    log("Checking emoji rendering...")
    log("  ok" if render_emoji("🔥") else "  unavailable (emoji captions will be skipped)")
    return 0
