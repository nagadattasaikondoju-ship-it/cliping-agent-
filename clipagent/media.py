"""Thin wrappers around ffmpeg / ffprobe / yt-dlp."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MediaError(RuntimeError):
    pass


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(f"'{name}' was not found on PATH. Install it first (see README).")
    return path


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"Command failed ({proc.returncode}): {' '.join(cmd[:4])} ...\n{tail}")
    return proc


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    duration: float
    fps: float
    has_audio: bool


def probe(path: str | Path) -> VideoInfo:
    require_binary("ffprobe")
    proc = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ])
    data = json.loads(proc.stdout)
    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"No video stream in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])
    num, _, den = video.get("avg_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    duration = float(data["format"].get("duration") or video.get("duration") or 0)
    width, height = int(video["width"]), int(video["height"])
    # Respect rotation metadata from phones.
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = abs(int(sd["rotation"]))
    if rotation in (90, 270):
        width, height = height, width
    return VideoInfo(Path(path), width, height, duration, fps or 30.0, has_audio)


_URL_RE = re.compile(r"^https?://", re.I)


def is_url(source: str) -> bool:
    return bool(_URL_RE.match(source))


def download(url: str, out_dir: Path, cookies_file: str | None = None,
             cookies_from_browser: str | None = None) -> Path:
    """Download a video (YouTube, Vimeo, X, Twitch VODs, ...) with yt-dlp."""
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on install
        raise MediaError("Install yt-dlp to download from URLs: pip install yt-dlp") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():
        candidates = sorted(out_dir.glob("source.*"))
        if not candidates:
            raise MediaError(f"Download finished but no file found for {url}")
        path = candidates[0]
    return path


def extract_audio(video: Path, out: Path, sample_rate: int = 16000) -> Path:
    require_binary("ffmpeg")
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn",
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out),
    ])
    return out


def loudness_envelope(video: Path, hop: float = 0.1, sample_rate: int = 8000) -> np.ndarray:
    """RMS loudness (dBFS) per `hop` seconds. Returns an empty array for silent videos."""
    require_binary("ffmpeg")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1",
         "-ar", str(sample_rate), "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    frame = max(1, int(sample_rate * hop))
    n = len(samples) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = samples[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return (20 * np.log10(rms)).astype(np.float32)
