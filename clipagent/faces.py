"""Face detection, tracking and active-speaker detection.

The clip is decoded once at a low resolution. Faces are detected a few times
per second (YuNet), linked into tracks, and for every face we measure how much
its mouth moves between two frames ~80 ms apart. Mouth motion minus eye-area
motion (which cancels head movement and compression noise) is a robust
"is talking" signal. Combined with the audio track (only voiced moments count)
and a little hysteresis, this tells us who is speaking at each moment.
"""

from __future__ import annotations

import os
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .env import CACHE_DIR

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
    path = CACHE_DIR / "face_detection_yunet_2023mar.onnx"
    if path.exists() and path.stat().st_size > 100_000:
        return path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
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


@dataclass
class Face:
    x: float
    y: float
    w: float
    h: float
    # Landmarks (right eye, left eye, nose, right mouth corner, left mouth corner) or None.
    landmarks: np.ndarray | None = None

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


class FaceDetector:
    """YuNet (OpenCV >= 4.8) with a Haar-cascade fallback for older OpenCV builds."""

    def __init__(self):
        import cv2

        self.cv2 = cv2
        try:
            cv2.setLogLevel(2)  # silence OpenCV DNN backend warnings
        except Exception:
            pass
        self.yunet = None
        self.haar = None
        model = _yunet_model() if hasattr(cv2, "FaceDetectorYN") else None
        if model is not None:
            self.yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.7, 0.3, 50)
        elif hasattr(cv2, "CascadeClassifier"):
            self.haar = cv2.CascadeClassifier(
                str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        else:
            raise RuntimeError("No face detector available (could not fetch the YuNet model). "
                               "Run `clipagent setup` with internet access.")

    def detect(self, image) -> list[Face]:
        cv2 = self.cv2
        if self.yunet is not None:
            h, w = image.shape[:2]
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(image)
            if faces is None:
                return []
            return [Face(*map(float, f[:4]), landmarks=np.array(f[4:14], dtype=np.float32).reshape(5, 2))
                    for f in faces]
        gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        side = max(24, int(image.shape[0] * 0.08))
        found = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(side, side))
        return [Face(*map(float, f)) for f in found]


def _regions(face: Face) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """(mouth box, eye-band box) as (x0, y0, x1, y1)."""
    x, y, w, h = face.x, face.y, face.w, face.h
    if face.landmarks is not None:
        lm = face.landmarks
        eyes, nose, mouth = lm[0:2], lm[2], lm[3:5]
        my = float(mouth[:, 1].mean())
        mx0, mx1 = float(mouth[:, 0].min()), float(mouth[:, 0].max())
        pad = 0.25 * max(8.0, mx1 - mx0)
        drop = max(4.0, my - float(nose[1]))
        mouth_box = (mx0 - pad, float(nose[1]) + 0.35 * drop, mx1 + pad, my + 0.9 * drop)
        ey = float(eyes[:, 1].mean())
        eye_box = (x + 0.15 * w, ey - 0.12 * h, x + 0.85 * w, ey + 0.08 * h)
    else:
        mouth_box = (x + 0.25 * w, y + 0.68 * h, x + 0.75 * w, y + 0.95 * h)
        eye_box = (x + 0.15 * w, y + 0.28 * h, x + 0.85 * w, y + 0.48 * h)
    return (tuple(int(round(v)) for v in mouth_box),  # type: ignore[return-value]
            tuple(int(round(v)) for v in eye_box))


def _mad(a: np.ndarray, b: np.ndarray, box: tuple[int, int, int, int]) -> float:
    h, w = a.shape[:2]
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    return float(np.mean(np.abs(a[y0:y1, x0:x1].astype(np.int16) - b[y0:y1, x0:x1].astype(np.int16))))


def mouth_activity(gray_a: np.ndarray, gray_b: np.ndarray, face: Face) -> float:
    """How much the mouth moved between two frames, with head motion cancelled out."""
    mouth, eyes = _regions(face)
    size_norm = 100.0 / max(20.0, face.h) ** 0.5  # small faces get noisier diffs
    return max(0.0, _mad(gray_a, gray_b, mouth) - 0.8 * _mad(gray_a, gray_b, eyes)) * size_norm


@dataclass
class Track:
    id: int
    boxes: dict[int, Face] = field(default_factory=dict)  # sample index -> face (source px)
    activity: dict[int, float] = field(default_factory=dict)

    def last(self) -> tuple[int, Face]:
        i = max(self.boxes)
        return i, self.boxes[i]

    def median_box(self) -> tuple[float, float, float, float]:
        arr = np.array([(f.cx, f.cy, f.w, f.h) for f in self.boxes.values()])
        return tuple(np.median(arr, axis=0))  # type: ignore[return-value]


@dataclass
class FaceAnalysis:
    times: np.ndarray  # seconds relative to clip start
    src_w: int
    src_h: int
    tracks: list[Track]
    active: np.ndarray  # track id per sample, -1 = none
    voiced: np.ndarray  # bool per sample

    @property
    def face_ratio(self) -> float:
        if len(self.times) == 0:
            return 0.0
        seen = np.zeros(len(self.times), dtype=bool)
        for t in self.tracks:
            seen[list(t.boxes)] = True
        return float(seen.mean())

    def visibility(self, track: Track) -> float:
        return len(track.boxes) / max(1, len(self.times))

    def speaking_share(self, track: Track) -> float:
        mask = self.voiced & (self.active >= 0)
        if not mask.any():
            return 0.0
        return float(np.mean(self.active[mask] == track.id))

    def centers(self) -> np.ndarray:
        """x centre (source px) of the active speaker per sample, NaN when unknown."""
        out = np.full(len(self.times), np.nan)
        by_id = {t.id: t for t in self.tracks}
        for i, tid in enumerate(self.active):
            if tid >= 0 and i in by_id[tid].boxes:
                out[i] = by_id[tid].boxes[i].cx
        return out


def _read_frames(video: Path, start: float, end: float, fps: float, width: int):
    """Yield (index, bgr_frame) at `fps` from [start, end), decoded by ffmpeg."""
    import cv2  # noqa: F401  (ensures OpenCV is installed before we start)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_side_data=rotation", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True).stdout.strip().splitlines()
    sw, sh = (int(v) for v in probe[0].split(",")[:2])
    rot = abs(int(probe[1])) if len(probe) > 1 and probe[1].strip().lstrip("-").isdigit() else 0
    if rot in (90, 270):
        sw, sh = sh, sw
    height = int(round(sh * width / sw / 2) * 2)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
         "-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE)
    size = width * height * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield i, np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3), (sw, sh)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def _moving_average(values: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware centred moving average."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        win = values[max(0, i - k): i + k + 1]
        win = win[~np.isnan(win)]
        if len(win):
            out[i] = win.mean()
    return out


def choose_speakers(times: np.ndarray, tracks: list[Track], voiced: np.ndarray,
                    rate: float, switch_ratio: float = 1.3, min_hold: float = 0.8,
                    min_shot: float = 1.5) -> np.ndarray:
    """Active track id per sample (hysteresis so the camera doesn't ping-pong)."""
    n = len(times)
    active = np.full(n, -1, dtype=int)
    if not tracks:
        return active
    smooth = {}
    for t in tracks:
        series = np.full(n, np.nan)
        for i, a in t.activity.items():
            series[i] = a
        smooth[t.id] = _moving_average(series, max(1, int(rate * 0.6)))

    current, since, challenger, challenger_since, challenger_idx = -1, -1e9, -1, 0.0, 0
    for i in range(n):
        visible = [t.id for t in tracks if i in t.boxes]
        if not visible:
            active[i] = current if current >= 0 else -1
            continue
        if current not in visible:
            # Current speaker left the frame: take the most active visible face right away.
            current = max(visible, key=lambda tid: np.nan_to_num(smooth[tid][i], nan=-1))
            since, challenger = times[i], -1
        elif voiced[i] and len(visible) > 1:
            best = max(visible, key=lambda tid: np.nan_to_num(smooth[tid][i], nan=-1))
            cur_score = np.nan_to_num(smooth[current][i], nan=0.0)
            best_score = np.nan_to_num(smooth[best][i], nan=0.0)
            if best != current and best_score > cur_score * switch_ratio + 0.05:
                if challenger != best:
                    challenger, challenger_since, challenger_idx = best, times[i], i
                if (times[i] - challenger_since >= min_hold
                        and challenger_since - since >= min_shot):
                    # We analyse offline, so back-date the cut to when the new
                    # speaker actually started: stable *and* no lag.
                    active[challenger_idx:i] = best
                    current, since, challenger = best, challenger_since, -1
            else:
                challenger = -1
        active[i] = current
    return active


def analyze_faces(video: Path, start: float, end: float, fps: float = 6.0,
                  loudness: np.ndarray | None = None, hop: float = 0.1,
                  analysis_width: int = 640) -> FaceAnalysis:
    """Detect, track and score faces in [start, end) of `video`."""
    import cv2

    detector = FaceDetector()
    tracks: list[Track] = []
    prev_gray = None
    prev_faces: list[Face] = []
    prev_track_ids: list[int] = []
    src_w = src_h = 0
    n_samples = 0
    scale = 1.0
    # Decode at 2x the analysis rate: even frames get face detection, and each
    # odd frame is compared with the even one before it for mouth motion.
    for idx, frame, (sw, sh) in _read_frames(video, start, end, fps * 2, analysis_width):
        src_w, src_h = sw, sh
        scale = sw / frame.shape[1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sample = idx // 2
        if idx % 2 == 1:
            if prev_gray is not None:
                for face, tid in zip(prev_faces, prev_track_ids):
                    tracks[tid].activity[sample] = mouth_activity(prev_gray, gray, face)
            continue
        n_samples = sample + 1
        min_side = frame.shape[0] * 0.06
        faces = sorted((f for f in detector.detect(frame) if f.h >= min_side),
                       key=lambda f: f.w * f.h, reverse=True)
        prev_gray, prev_faces, prev_track_ids = gray, faces, []
        for face in faces:
            src_face = Face(face.x * scale, face.y * scale, face.w * scale, face.h * scale)
            best, best_d = None, None
            for t in tracks:
                last_i, last = t.last()
                if last_i == sample or sample - last_i > fps * 1.5:
                    continue
                d = np.hypot(last.cx - src_face.cx, last.cy - src_face.cy)
                if d < 0.6 * max(last.w, src_face.w) and 0.6 < src_face.h / max(1.0, last.h) < 1.6:
                    if best_d is None or d < best_d:
                        best, best_d = t, d
            if best is None:
                best = Track(id=len(tracks))
                tracks.append(best)
            best.boxes[sample] = src_face
            prev_track_ids.append(best.id)

    times = np.arange(n_samples) / fps
    # Drop short-lived tracks (false positives, people walking through).
    keep = [t for t in tracks if len(t.boxes) >= max(3, 0.1 * n_samples)]
    remap = {t.id: i for i, t in enumerate(keep)}
    for t in keep:
        t.id = remap[t.id]

    voiced = np.ones(n_samples, dtype=bool)
    if loudness is not None and len(loudness):
        speech = loudness[loudness > -60]
        threshold = (np.median(speech) - 12.0) if len(speech) else -45.0
        for i, t in enumerate(times):
            j = int((start + t) / hop)
            win = loudness[max(0, j - 2): j + 3]
            voiced[i] = bool(len(win)) and float(np.max(win)) > threshold

    active = choose_speakers(times, keep, voiced, fps)
    return FaceAnalysis(times=times, src_w=src_w, src_h=src_h, tracks=keep,
                        active=active, voiced=voiced)
