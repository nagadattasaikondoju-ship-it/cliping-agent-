"""Pipeline settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    # Clip selection
    num_clips: int = 5
    min_duration: float = 15.0
    max_duration: float = 60.0
    target_duration: float = 35.0
    max_overlap: float = 0.25  # max fraction of overlap allowed between chosen clips

    # Downloading (yt-dlp). YouTube often requires cookies when run from a server.
    cookies_file: str | None = None
    cookies_from_browser: str | None = None  # e.g. "chrome", "firefox"

    # Transcription (faster-whisper)
    whisper_model: str = "small"
    whisper_device: str = "auto"  # auto | cpu | cuda
    whisper_compute_type: str = "default"
    language: str | None = None

    # LLM highlight detection (Claude). Disabled automatically if no credentials.
    use_llm: bool = True
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    llm_weight: float = 0.6  # share of the final virality score that comes from the LLM

    # Output video
    width: int = 1080
    height: int = 1920
    layout: str = "auto"  # auto | crop | split | blur
    speaker_detection: bool = True  # follow whoever is talking, not just the biggest face
    analysis_fps: float = 6.0
    crf: int = 20
    preset: str = "medium"

    # Captions
    captions: bool = True
    caption_font: str = "Inter"
    caption_size: int = 88
    caption_color: str = "FFFFFF"  # RGB hex
    caption_highlight: str = "00E5FF"  # RGB hex for the word being spoken
    caption_max_words: int = 3
    caption_uppercase: bool = True
    show_title: bool = True  # overlay the clip's hook title during the first seconds
    emoji: bool = False  # pop an emoji above captions on words like "money", "fire", "love"

    # B-roll cutaways (Pexels with PEXELS_API_KEY, and/or a local folder)
    broll: bool = False
    broll_dir: str | None = None
    broll_max: int = 2
    broll_duration: float = 2.2

    # Fields a clip may override individually (from the web editor)
    @staticmethod
    def clip_option_keys() -> tuple[str, ...]:
        return ("layout", "captions", "show_title", "emoji", "broll", "caption_highlight",
                "caption_font", "caption_size", "caption_uppercase", "caption_max_words",
                "speaker_detection")
