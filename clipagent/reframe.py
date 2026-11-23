"""Reframe landscape video to 9:16.

Layouts:
- crop:  a virtual camera follows the active speaker (see faces.py), with a
         dead-zone, easing, and hard cuts when the speaker changes.
- split: two people talking -> each gets their own half of the frame, stacked.
- blur:  no usable faces (screen recordings, slides, b-roll) -> the whole frame
         over a blurred, zoomed copy of itself.
- fit:   the source is already vertical.
The camera path is compiled into an ffmpeg `crop` expression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .faces import FaceAnalysis, _yunet_model, analyze_faces  # noqa: F401  (re-exported)
from .media import VideoInfo


@dataclass
class ReframePlan:
    layout: str  # "crop" | "split" | "blur" | "fit"
    crop_w: int = 0
    crop_h: int = 0
    keyframes: list[tuple[float, float]] = field(default_factory=list)  # (t, crop_x)
    face_ratio: float = 0.0
    # split layout: [(x, y, w, h), (x, y, w, h)] crops for the top and bottom halves
    split_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    speakers: int = 0
    speaker_switches: int = 0


def _even(x: float) -> int:
    v = int(round(x))
    return v - (v % 2)


def smooth_camera(times: np.ndarray, centers: np.ndarray, src_w: int, crop_w: int,
                  dead_zone: float = 0.10, ease: float = 0.45,
                  cut_threshold: float = 0.5) -> list[tuple[float, float]]:
    """Turn noisy subject centres into a stable camera path of (t, crop_x) keyframes."""
    max_x = max(0, src_w - crop_w)
    if len(times) == 0:
        return [(0.0, max_x / 2)]
    valid = ~np.isnan(centers)
    if not valid.any():
        return [(0.0, max_x / 2)]
    # Fill gaps by holding the nearest known position.
    filled = centers.copy()
    idx = np.where(valid, np.arange(len(centers)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = filled[idx]
    first_valid = int(np.argmax(valid))
    filled[:first_valid] = filled[first_valid]
    # Median filter to kill detector blips.
    k = 2
    med = np.array([np.median(filled[max(0, i - k): i + k + 1]) for i in range(len(filled))])

    # Dead-zone with hysteresis: small drifts are ignored; once the subject leaves the
    # zone the camera eases all the way back onto it, then holds still again.
    cam = med[0]
    moving = False
    path: list[float] = []
    cuts: set[int] = set()
    for i, target in enumerate(med):
        err = target - cam
        if abs(err) > cut_threshold * crop_w:
            # Subject jumped (speaker change / camera cut): hard-cut instead of panning.
            cam, moving = target, False
            cuts.add(i)
        elif abs(err) > dead_zone * crop_w:
            moving = True
        if moving:
            cam += err * ease
            if abs(target - cam) < 0.02 * crop_w:
                moving = False
        path.append(cam)
    xs = np.clip(np.array(path) - crop_w / 2, 0, max_x)

    # Compress to keyframes. Linear interpolation runs between consecutive keyframes,
    # so a hold must be closed by a keyframe right before motion starts, and a cut
    # gets a near-zero-length transition.
    keyframes = [(float(times[0]), float(xs[0]))]
    for i in range(1, len(xs)):
        if abs(xs[i] - xs[i - 1]) < 1.0 and i != len(xs) - 1:
            continue
        t_prev, x_prev = float(times[i - 1]), float(xs[i - 1])
        if i in cuts:
            t_prev = float(times[i]) - 0.001
        if keyframes[-1][0] < t_prev:
            keyframes.append((t_prev, x_prev))
        keyframes.append((float(times[i]), float(xs[i])))
    return keyframes


def crop_x_expression(keyframes: list[tuple[float, float]]) -> str:
    """Piecewise-linear interpolation of keyframes as an ffmpeg expression in `t`."""
    if len(keyframes) == 1:
        return f"{keyframes[0][1]:.1f}"
    terms = []
    (t0, x0) = keyframes[0]
    terms.append(f"lt(t,{t0:.3f})*{x0:.1f}")
    for (ta, xa), (tb, xb) in zip(keyframes, keyframes[1:]):
        if tb <= ta:
            continue
        slope = (xb - xa) / (tb - ta)
        terms.append(f"between(t,{ta:.3f},{tb:.3f}-0.0001)*({xa:.1f}+{slope:.3f}*(t-{ta:.3f}))")
    tl, xl = keyframes[-1]
    terms.append(f"gte(t,{tl:.3f}-0.0001)*{xl:.1f}")
    return "+".join(terms)


def split_boxes(analysis: FaceAnalysis, out_w: int, out_h: int,
                track_ids: list[int]) -> list[tuple[int, int, int, int]]:
    """A static crop per person, framed head-and-shoulders, for each half of the frame."""
    aspect = out_w / (out_h / 2)
    by_id = {t.id: t for t in analysis.tracks}
    boxes = []
    for tid in track_ids:
        cx, cy, _fw, fh = by_id[tid].median_box()
        ch = min(analysis.src_h, max(fh * 3.0, analysis.src_h * 0.45))
        cw = ch * aspect
        if cw > analysis.src_w:
            cw, ch = analysis.src_w, analysis.src_w / aspect
        x = min(max(0.0, cx - cw / 2), analysis.src_w - cw)
        y = min(max(0.0, cy - 0.42 * ch), analysis.src_h - ch)
        boxes.append((_even(x), _even(y), _even(cw), _even(ch)))
    return boxes


def choose_split(analysis: FaceAnalysis, crop_w: int, min_visible: float = 0.6,
                 min_share: float = 0.15) -> list[int] | None:
    """Track ids [left, right] if this looks like a two-person conversation that
    can't fit in a single 9:16 crop, else None."""
    if len(analysis.tracks) < 2:
        return None
    ranked = sorted(analysis.tracks, key=lambda t: (analysis.speaking_share(t), t.median_box()[3]),
                    reverse=True)[:2]
    if any(analysis.visibility(t) < min_visible for t in ranked):
        return None
    if any(analysis.speaking_share(t) < min_share for t in ranked):
        return None
    (ax, *_), (bx, *_) = ranked[0].median_box(), ranked[1].median_box()
    if abs(ax - bx) < 0.6 * crop_w:  # close enough to share one frame
        return None
    left, right = sorted(ranked, key=lambda t: t.median_box()[0])
    return [left.id, right.id]


def plan_reframe(info: VideoInfo, start: float, end: float, layout: str,
                 out_w: int, out_h: int, loudness: np.ndarray | None = None,
                 speaker_detection: bool = True, analysis_fps: float = 6.0) -> ReframePlan:
    target_aspect = out_w / out_h
    src_aspect = info.width / info.height
    if src_aspect <= target_aspect + 0.01:
        return ReframePlan(layout="fit")

    crop_h = _even(info.height)
    crop_w = _even(info.height * target_aspect)
    if layout == "blur":
        return ReframePlan(layout="blur", crop_w=crop_w, crop_h=crop_h)

    analysis = analyze_faces(info.path, start, end, fps=analysis_fps,
                             loudness=loudness if speaker_detection else None)
    face_ratio = analysis.face_ratio
    pair: list[int] | None = None
    if layout == "auto":
        pair = choose_split(analysis, crop_w)
    elif layout == "split" and len(analysis.tracks) >= 2:
        two = sorted(analysis.tracks, key=lambda t: (analysis.speaking_share(t), t.median_box()[3]),
                     reverse=True)[:2]
        pair = [t.id for t in sorted(two, key=lambda t: t.median_box()[0])]
    if pair:
        return ReframePlan(layout="split", crop_w=crop_w, crop_h=crop_h, face_ratio=face_ratio,
                           split_boxes=split_boxes(analysis, out_w, out_h, pair), speakers=2)
    if layout == "auto" and face_ratio < 0.3:
        return ReframePlan(layout="blur", crop_w=crop_w, crop_h=crop_h, face_ratio=face_ratio)

    if speaker_detection:
        centers = analysis.centers()
        switches = int(np.sum(np.diff(analysis.active[analysis.active >= 0]) != 0))
    else:  # follow the largest face
        centers = np.full(len(analysis.times), np.nan)
        for i in range(len(analysis.times)):
            seen = [t.boxes[i] for t in analysis.tracks if i in t.boxes]
            if seen:
                centers[i] = max(seen, key=lambda f: f.w * f.h).cx
        switches = 0
    keyframes = smooth_camera(analysis.times, centers, analysis.src_w or info.width, crop_w)
    return ReframePlan(layout="crop", crop_w=crop_w, crop_h=crop_h, keyframes=keyframes,
                       face_ratio=face_ratio, speakers=len(analysis.tracks),
                       speaker_switches=switches)


def video_filter(plan: ReframePlan, out_w: int, out_h: int, out_label: str = "v") -> str:
    """ffmpeg filtergraph from input [0:v] to [out_label] for the plan."""
    if plan.layout == "crop":
        x = crop_x_expression(plan.keyframes)
        return (f"[0:v]crop=w={plan.crop_w}:h={plan.crop_h}:x='clip({x},0,iw-ow)':y=0,"
                f"scale={out_w}:{out_h}:flags=lanczos,setsar=1[{out_label}]")
    if plan.layout == "split":
        half = out_h // 2
        (ax, ay, aw, ah), (bx, by, bw, bh) = plan.split_boxes
        return (f"[0:v]split=2[sa][sb];"
                f"[sa]crop={aw}:{ah}:{ax}:{ay},scale={out_w}:{half}:flags=lanczos,setsar=1[top];"
                f"[sb]crop={bw}:{bh}:{bx}:{by},scale={out_w}:{out_h - half}:flags=lanczos,setsar=1[bot];"
                f"[top][bot]vstack=inputs=2[{out_label}]")
    if plan.layout == "blur":
        return (f"[0:v]split=2[bgsrc][fgsrc];"
                f"[bgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                f"crop={out_w}:{out_h},boxblur=luma_radius=40:luma_power=2,eq=brightness=-0.08[bg];"
                f"[fgsrc]scale={out_w}:-2:flags=lanczos[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[{out_label}]")
    # fit: already vertical-ish; scale and pad.
    return (f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[{out_label}]")
