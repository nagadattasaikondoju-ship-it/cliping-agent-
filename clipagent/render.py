"""Cut, reframe, add B-roll, captions and emoji, and encode one clip with ffmpeg.

Layer order (bottom to top): reframed speaker -> B-roll cutaways -> captions
and title (libass) -> emoji.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .emoji import EMOJI_SIZE
from .media import VideoInfo, require_binary, run
from .reframe import ReframePlan, video_filter


@dataclass
class Overlays:
    ass_path: Path | None = None
    # (start, duration, video path) relative to the clip
    broll: list[tuple[float, float, str]] = field(default_factory=list)
    # (start, end, png path) relative to the clip
    emoji: list[tuple[float, float, str]] = field(default_factory=list)
    emoji_y: int = 0


def _filter_path(p: Path) -> str:
    # Escape for use inside an ffmpeg filter argument.
    return str(p.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def build_command(info: VideoInfo, start: float, end: float, plan: ReframePlan,
                  overlays: Overlays, out_path: Path, settings: Settings) -> list[str]:
    W, H = settings.width, settings.height
    inputs = ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(info.path)]
    chains = [video_filter(plan, W, H, out_label="l0")]
    cur, n_in, n = "l0", 1, 0

    for t0, dur, path in overlays.broll:
        inputs += ["-i", path]
        n += 1
        chains.append(
            f"[{n_in}:v]trim=duration={dur:.3f},setpts=PTS-STARTPTS+{t0:.3f}/TB,"
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,"
            f"format=yuv420p[br{n}];"
            f"[{cur}][br{n}]overlay=0:0:enable='between(t,{t0:.3f},{t0 + dur:.3f})':"
            f"eof_action=pass[l{n}]")
        cur, n_in = f"l{n}", n_in + 1

    if overlays.ass_path is not None:
        n += 1
        chains.append(f"[{cur}]ass='{_filter_path(overlays.ass_path)}'[l{n}]")
        cur = f"l{n}"

    for t0, t1, png in overlays.emoji:
        inputs += ["-loop", "1", "-t", f"{end - start:.3f}", "-i", png]
        n += 1
        pop = max(0.05, min(0.15, (t1 - t0) / 4))
        # Pop in: scale from 60% to 100% over `pop` seconds.
        chains.append(
            f"[{n_in}:v]format=rgba,scale=w='iw*min(1,0.6+0.4*max(0,t-{t0:.3f})/{pop:.3f})':"
            f"h=-1:eval=frame[em{n}];"
            f"[{cur}][em{n}]overlay=x=(W-w)/2:y={overlays.emoji_y}+({EMOJI_SIZE}-h)/2:"
            f"enable='between(t,{t0:.3f},{t1:.3f})':eof_action=pass[l{n}]")
        cur, n_in = f"l{n}", n_in + 1

    graph = ";".join(chains)
    assert graph.endswith(f"[{cur}]")
    graph = graph[: -len(f"[{cur}]")] + "[v]"  # name the final output
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]"]
    if info.has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k",
                "-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]
    cmd += [
        "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
        "-pix_fmt", "yuv420p", "-r", f"{min(info.fps, 60):.3f}",
        "-t", f"{end - start:.3f}", "-movflags", "+faststart", str(out_path),
    ]
    return cmd


def render_clip(info: VideoInfo, start: float, end: float, plan: ReframePlan,
                overlays: Overlays, out_path: Path, settings: Settings) -> Path:
    require_binary("ffmpeg")
    run(build_command(info, start, end, plan, overlays, out_path, settings))
    return out_path
