"""Build a synthetic two-person "podcast" video with a known speaker schedule.

Two real faces (cropped from a photo) sit left and right; only the current
speaker's mouth moves. Used to check active-speaker detection and the split
layout. Usage:

    python scripts/make_two_speaker_video.py photo.jpg speech.(wav|ogg|mp3) out.mp4
"""

from __future__ import annotations

import subprocess
import sys

import cv2
import numpy as np

# Speaker schedule: (start, end, speaker index 0=left 1=right)
SCHEDULE = [(0, 12, 0), (12, 24, 1), (24, 32, 0), (32, 40, 1)]
W, H, FPS, DUR = 1920, 1080, 25, 40


def speaker_at(t: float) -> int:
    for a, b, s in SCHEDULE:
        if a <= t < b:
            return s
    return SCHEDULE[-1][2]


def find_faces(img: np.ndarray, n: int = 2):
    from clipagent.faces import FaceDetector

    faces = FaceDetector().detect(img)
    faces = sorted(faces, key=lambda f: f.w * f.h, reverse=True)[:n]
    return sorted(faces, key=lambda f: f.cx)


def main(photo: str, audio: str, out: str) -> None:
    img = cv2.imread(photo)
    patches = []
    for f in find_faces(img):
        side = int(f.h * 2.2)
        x0, y0 = int(f.cx - side / 2), int(f.cy - side * 0.45)
        x0, y0 = max(0, x0), max(0, y0)
        patch = cv2.resize(img[y0:y0 + side, x0:x0 + side], (520, 520))
        s = 520 / side
        lm = (f.landmarks - np.array([x0, y0])) * s
        nose_y, mouth_y = lm[2, 1], lm[3:5, 1].mean()
        mx0, mx1 = lm[3:5, 0].min(), lm[3:5, 0].max()
        region = (int(mx0 - 0.4 * (mx1 - mx0)), int(nose_y + 0.4 * (mouth_y - nose_y)),
                  int(mx1 + 0.4 * (mx1 - mx0)), int(mouth_y + 1.4 * (mouth_y - nose_y)))
        patches.append((patch, region))

    proc = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-i", audio, "-t", str(DUR), "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", out,
    ], stdin=subprocess.PIPE)
    rng = np.random.default_rng(1)
    positions = [(200, 280), (1200, 280)]
    for k in range(DUR * FPS):
        t = k / FPS
        frame = np.full((H, W, 3), (48, 40, 32), dtype=np.uint8)
        talking = speaker_at(t)
        for i, ((patch, (x0, y0, x1, y1)), (px, py)) in enumerate(zip(patches, positions)):
            p = patch.copy()
            if i == talking:
                open_ = 1.0 + 0.45 * abs(np.sin(2 * np.pi * 3.7 * t + rng.normal(0, 0.4)))
                reg = p[y0:y1, x0:x1]
                stretched = cv2.resize(reg, (x1 - x0, int((y1 - y0) * open_)))
                p[y0:y1, x0:x1] = stretched[: y1 - y0]
            frame[py:py + 520, px:px + 520] = p
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    main(*sys.argv[1:4])
