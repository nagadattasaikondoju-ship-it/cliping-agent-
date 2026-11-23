"""Cut, reframe, caption and encode one clip with ffmpeg."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .media import VideoInfo, require_binary, run
from .reframe import ReframePlan, video_filter


def _filter_path(p: Path) -> str:
    # Escape for use inside an ffmpeg filter argument.
    return str(p.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def render_clip(info: VideoInfo, start: float, end: float, plan: ReframePlan,
                ass_path: Path | None, out_path: Path, settings: Settings) -> Path:
    require_binary("ffmpeg")
    W, H = settings.width, settings.height
    graph = video_filter(plan, W, H)
    if ass_path is not None:
        graph = graph[: -len("[v]")] + f",ass='{_filter_path(ass_path)}'[v]"

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(info.path),
        "-filter_complex", graph, "-map", "[v]",
    ]
    if info.has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k",
                "-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]
    cmd += [
        "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
        "-pix_fmt", "yuv420p", "-r", f"{min(info.fps, 60):.3f}",
        "-movflags", "+faststart", str(out_path),
    ]
    run(cmd)
    return out_path
