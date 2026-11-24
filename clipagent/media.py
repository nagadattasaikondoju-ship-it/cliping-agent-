"""Thin wrappers around ffmpeg / ffprobe / yt-dlp."""

from __future__ import annotations

import base64
import json
import os
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


_BOT_CHECK = ("confirm you’re not a bot", "confirm you're not a bot", "sign in to confirm")


def _cookie_file(out_dir: Path, cookies_file: str | None) -> str | None:
    """The cookies.txt to use: the one passed in, else $YTDLP_COOKIES (a path), else
    $YTDLP_COOKIES_TXT (the file's contents, e.g. from a hosting secret). Contents are
    copied into the job folder because yt-dlp writes updated cookies back to the file."""
    path = cookies_file or os.environ.get("YTDLP_COOKIES")
    if path:
        return path
    text = os.environ.get("YTDLP_COOKIES_TXT", "").strip()
    if not text:
        return None
    if "\t" not in text:  # pasted as base64 (single line, no tabs)
        try:
            text = base64.b64decode(text, validate=False).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            pass
    dest = out_dir / "cookies.txt"
    dest.write_text(text.replace("\r\n", "\n") + "\n")
    return str(dest)


def download(url: str, out_dir: Path, cookies_file: str | None = None,
             cookies_from_browser: str | None = None, max_minutes: float = 0) -> Path:
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
    if max_minutes:
        from yt_dlp.utils import match_filter_func

        # "<=?" lets a video through when YouTube hides its duration; the length
        # is checked again after the download.
        opts["match_filter"] = match_filter_func(f"duration <=? {int(max_minutes * 60)}")
    cookies = _cookie_file(out_dir, cookies_file)
    if cookies:
        opts["cookiefile"] = cookies
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as exc:
        if any(s in str(exc).lower() for s in _BOT_CHECK):
            hint = ("The YouTube cookies on this server have expired or were rejected; "
                    "export fresh ones." if cookies else
                    "Download the video yourself and upload the file instead. To fix links "
                    "for good, give the server YouTube cookies (see README: YouTube links).")
            raise MediaError(f"YouTube blocked the download from this server (it asks to sign "
                             f"in to prove it's not a bot). {hint}") from exc
        raise
    if not path.exists():
        candidates = sorted(p for p in out_dir.glob("source.*"))
        if not candidates:
            if max_minutes:
                raise MediaError(f"Nothing was downloaded. The video may be longer than the "
                                 f"{max_minutes:g}-minute limit.")
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
