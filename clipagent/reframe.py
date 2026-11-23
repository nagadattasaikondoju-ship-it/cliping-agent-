"""Reframe landscape video to 9:16.

We sample frames, detect faces, pick the main subject, and turn its path into a
smooth "virtual camera" (dead-zone + easing so the frame doesn't jitter). The
camera path is compiled into an ffmpeg `crop` expression. When no face is
found (screen recordings, b-roll, slides) we use a blurred-background layout
that keeps the whole frame visible.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .media import VideoInfo


@dataclass
class ReframePlan:
    layout: str  # "crop" | "blur" | "fit" (source is already vertical)
    crop_w: int = 0
    crop_h: int = 0
    keyframes: list[tuple[float, float]] = field(default_factory=list)  # (t, crop_x)
    face_ratio: float = 0.0


def _even(x: float) -> int:
    v = int(round(x))
    return v - (v % 2)


YUNET_URLS = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
)


def _yunet_model() -> Path | None:
    """Path to the YuNet face model, downloading it (~230 KB) on first use."""
    env = os.environ.get("CLIPAGENT_FACE_MODEL")
    if env:
        return Path(env)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "clipagent"
    path = cache / "face_detection_yunet_2023mar.onnx"
    if path.exists() and path.stat().st_size > 100_000:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    for url in YUNET_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            if len(data) > 100_000:  # guard against LFS pointers / error pages
                path.write_bytes(data)
                return path
        except Exception:
            continue
    return None


class FaceDetector:
    """YuNet (OpenCV >= 4.8) with a Haar-cascade fallback for older OpenCV builds."""

    def __init__(self):
        import cv2

        self.cv2 = cv2
        self.yunet = None
        self.haar = None
        model = _yunet_model() if hasattr(cv2, "FaceDetectorYN") else None
        if model is not None:
            self.yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.7, 0.3, 50)
        elif hasattr(cv2, "CascadeClassifier"):
            self.haar = cv2.CascadeClassifier(
                str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        else:
            raise RuntimeError("No face detector available (could not fetch the YuNet model).")

    def detect(self, image) -> list[tuple[float, float, float, float]]:
        """Return faces as (x, y, w, h) in `image` pixels."""
        cv2 = self.cv2
        if self.yunet is not None:
            h, w = image.shape[:2]
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(image)
            return [tuple(map(float, f[:4])) for f in (faces if faces is not None else [])]
        gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        side = max(24, int(image.shape[0] * 0.08))
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(side, side))
        return [tuple(map(float, f)) for f in faces]


def detect_subject_track(video: Path, start: float, end: float, sample_fps: float = 3.0,
                         analysis_width: int = 640) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (times relative to start, subject centre x in source pixels or NaN, src_width)."""
    import cv2  # imported lazily so the rest of the package works without OpenCV

    try:
        cv2.setLogLevel(2)  # silence OpenCV DNN warnings
    except Exception:
        pass
    detector = FaceDetector()
    cap = cv2.VideoCapture(str(video))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    times = np.arange(start, end, 1.0 / sample_fps)
    centers = np.full(len(times), np.nan)
    prev_center: float | None = None
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        scale = analysis_width / frame.shape[1]
        small = cv2.resize(frame, (analysis_width, int(frame.shape[0] * scale)))
        min_side = small.shape[0] * 0.06
        faces = [f for f in detector.detect(small) if f[3] >= min_side]
        if not faces:
            continue
        # Prefer the largest face, but stick with the current subject when sizes are close.
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        best = faces[0]
        if prev_center is not None and len(faces) > 1:
            largest = best[2] * best[3]
            close = [f for f in faces if f[2] * f[3] >= 0.6 * largest]
            best = min(close, key=lambda f: abs((f[0] + f[2] / 2) / scale - prev_center))
        cx = (best[0] + best[2] / 2) / scale
        centers[i] = cx
        prev_center = cx
    cap.release()
    return times - start, centers, src_w


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


def plan_reframe(info: VideoInfo, start: float, end: float, layout: str,
                 out_w: int, out_h: int) -> ReframePlan:
    target_aspect = out_w / out_h
    src_aspect = info.width / info.height
    if src_aspect <= target_aspect + 0.01:
        return ReframePlan(layout="fit")

    crop_h = _even(info.height)
    crop_w = _even(info.height * target_aspect)
    if layout == "blur":
        return ReframePlan(layout="blur", crop_w=crop_w, crop_h=crop_h)

    times, centers, src_w = detect_subject_track(info.path, start, end)
    face_ratio = float(np.mean(~np.isnan(centers))) if len(centers) else 0.0
    if layout == "auto" and face_ratio < 0.3:
        return ReframePlan(layout="blur", crop_w=crop_w, crop_h=crop_h, face_ratio=face_ratio)
    keyframes = smooth_camera(times, centers, src_w or info.width, crop_w)
    return ReframePlan(layout="crop", crop_w=crop_w, crop_h=crop_h,
                       keyframes=keyframes, face_ratio=face_ratio)


def video_filter(plan: ReframePlan, out_w: int, out_h: int) -> str:
    """ffmpeg filtergraph (input [0:v], output [v]) for the plan."""
    if plan.layout == "crop":
        x = crop_x_expression(plan.keyframes)
        return (f"[0:v]crop=w={plan.crop_w}:h={plan.crop_h}:x='clip({x},0,iw-ow)':y=0,"
                f"scale={out_w}:{out_h}:flags=lanczos,setsar=1[v]")
    if plan.layout == "blur":
        return (f"[0:v]split=2[bgsrc][fgsrc];"
                f"[bgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                f"crop={out_w}:{out_h},boxblur=luma_radius=40:luma_power=2,eq=brightness=-0.08[bg];"
                f"[fgsrc]scale={out_w}:-2:flags=lanczos[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
    # fit: already vertical-ish; scale and pad.
    return (f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v]")
