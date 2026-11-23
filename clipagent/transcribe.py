"""Speech-to-text with word-level timestamps (faster-whisper)."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from .config import Settings
from .models import Transcript, Word


def load_transcript(path: str | Path) -> Transcript:
    return Transcript.from_dict(json.loads(Path(path).read_text()))


def save_transcript(transcript: Transcript, path: str | Path) -> None:
    Path(path).write_text(json.dumps(transcript.to_dict(), ensure_ascii=False))


def transcribe(audio_path: Path, settings: Settings, log=print) -> Transcript:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is required for transcription: pip install faster-whisper"
        ) from exc

    device = settings.whisper_device
    if device == "auto":
        try:
            import ctranslate2  # type: ignore

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute_type = settings.whisper_compute_type
    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "int8"

    log(f"Loading Whisper '{settings.whisper_model}' on {device} ({compute_type})...")
    model = WhisperModel(settings.whisper_model, device=device, compute_type=compute_type)
    # Decode the 16 kHz mono WAV ourselves: avoids faster-whisper's PyAV loader,
    # which breaks across PyAV releases.
    with wave.open(str(audio_path), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = model.transcribe(
        audio,
        language=settings.language,
        word_timestamps=True,
        vad_filter=True,
    )
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(text=text, start=float(w.start), end=float(w.end)))
    log(f"Transcribed {len(words)} words (language: {info.language}).")
    return Transcript(words=words, language=info.language)
