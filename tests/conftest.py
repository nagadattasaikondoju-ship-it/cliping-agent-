import shutil
import subprocess

import pytest

from clipagent.models import Transcript, Word

SENTENCES = [
    "So um yeah we started the podcast last year.",
    "Why do most startups fail in their first year?",
    "Here's the secret nobody tells you about money.",
    "The reason is simple, you run out of cash before you find customers.",
    "I lost a million dollars learning this lesson and it was terrible.",
    "If you want to survive, you need to talk to users every single day.",
    "And then we went to lunch.",
    "It was fine I guess.",
    "Honestly the biggest mistake I made was hiring too fast!",
    "That's why I always tell founders to stay small until it hurts.",
    "Anyway let's move on.",
    "The weather was okay.",
]


def make_transcript(sentences=SENTENCES, wps=2.8, pause=0.4, start=1.0) -> Transcript:
    words, t = [], start
    for s in sentences:
        for tok in s.split():
            words.append(Word(tok, round(t, 3), round(t + 1 / wps * 0.85, 3)))
            t += 1 / wps
        t += pause
    return Transcript(words=words)


@pytest.fixture
def transcript():
    return make_transcript()


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("media") / "long.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=90",
        "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration=90",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return path
