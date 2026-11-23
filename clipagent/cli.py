"""Command-line interface: `clipagent <video-or-url> [options]`."""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    d = Settings()
    p = argparse.ArgumentParser(
        prog="clipagent",
        description="Turn a long video into ranked, captioned 9:16 short-form clips.",
    )
    p.add_argument("source", help="Path to a video file or a URL (YouTube, Vimeo, ...)")
    p.add_argument("-o", "--out", default="clips", help="Output directory (default: clips)")
    p.add_argument("-n", "--num-clips", type=int, default=d.num_clips)
    p.add_argument("--min", dest="min_duration", type=float, default=d.min_duration,
                   help="Minimum clip length in seconds")
    p.add_argument("--max", dest="max_duration", type=float, default=d.max_duration,
                   help="Maximum clip length in seconds")
    p.add_argument("--target", dest="target_duration", type=float, default=d.target_duration,
                   help="Preferred clip length in seconds")
    p.add_argument("--layout", choices=["auto", "crop", "blur"], default=d.layout,
                   help="auto = face-tracked crop if faces are found, else blurred background")
    p.add_argument("--cookies", help="cookies.txt for yt-dlp (needed for YouTube on servers)")
    p.add_argument("--cookies-from-browser", help="Read yt-dlp cookies from a browser, e.g. chrome")
    p.add_argument("--whisper-model", default=d.whisper_model,
                   help="tiny | base | small | medium | large-v3 | distil-large-v3 ...")
    p.add_argument("--device", dest="whisper_device", default=d.whisper_device)
    p.add_argument("--language", default=None, help="Force transcription language, e.g. en")
    p.add_argument("--transcript", default=None,
                   help="Use an existing transcript JSON instead of running Whisper")
    p.add_argument("--no-llm", action="store_true", help="Heuristic scoring only (no Claude)")
    p.add_argument("--llm-model", default=d.llm_model)
    p.add_argument("--effort", dest="llm_effort", default=d.llm_effort,
                   choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--no-title", action="store_true", help="Don't overlay the hook title")
    p.add_argument("--font", dest="caption_font", default=d.caption_font)
    p.add_argument("--font-size", dest="caption_size", type=int, default=d.caption_size)
    p.add_argument("--highlight-color", dest="caption_highlight", default=d.caption_highlight,
                   help="Hex RGB colour of the active word (default 00E5FF)")
    p.add_argument("--words-per-caption", dest="caption_max_words", type=int,
                   default=d.caption_max_words)
    p.add_argument("--no-uppercase", action="store_true")
    p.add_argument("--resolution", default=f"{d.width}x{d.height}",
                   help="Output WxH (default 1080x1920)")
    p.add_argument("--crf", type=int, default=d.crf)
    p.add_argument("--preset", default=d.preset)
    p.add_argument("--dry-run", action="store_true",
                   help="Find and score clips but don't render video")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    w, _, h = args.resolution.lower().partition("x")
    s = Settings(
        num_clips=args.num_clips, min_duration=args.min_duration,
        max_duration=args.max_duration, target_duration=args.target_duration,
        whisper_model=args.whisper_model, whisper_device=args.whisper_device,
        language=args.language, use_llm=not args.no_llm, llm_model=args.llm_model,
        llm_effort=args.llm_effort, width=int(w), height=int(h), layout=args.layout,
        crf=args.crf, preset=args.preset, captions=not args.no_captions,
        caption_font=args.caption_font, caption_size=args.caption_size,
        caption_highlight=args.caption_highlight, caption_max_words=args.caption_max_words,
        caption_uppercase=not args.no_uppercase, show_title=not args.no_title,
        cookies_file=args.cookies, cookies_from_browser=args.cookies_from_browser,
    )
    if s.min_duration >= s.max_duration:
        print("--min must be smaller than --max", file=sys.stderr)
        return 2
    s.target_duration = min(max(s.target_duration, s.min_duration), s.max_duration)
    try:
        clips = run_pipeline(args.source, args.out, s, transcript_path=args.transcript,
                             render=not args.dry_run)
    except Exception as exc:  # surface a clean message on the CLI
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"\nDone: {len(clips)} clips -> {args.out}/ (scores in {args.out}/clips.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
