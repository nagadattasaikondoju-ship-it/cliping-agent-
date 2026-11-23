"""Command-line interface.

    clipagent video.mp4 [options]        # same as `clipagent run video.mp4`
    clipagent doctor                     # check requirements
    clipagent setup                      # download models ahead of time
    clipagent serve                      # web UI + editor
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import Settings
from .env import load_env

COMMANDS = ("run", "doctor", "setup", "serve")


def add_run_args(p: argparse.ArgumentParser) -> None:
    d = Settings()
    p.add_argument("source", help="Path to a video file or a URL (YouTube, Vimeo, ...)")
    p.add_argument("-o", "--out", default="clips", help="Output directory (default: clips)")
    p.add_argument("-n", "--num-clips", type=int, default=d.num_clips)
    p.add_argument("--min", dest="min_duration", type=float, default=d.min_duration,
                   help="Minimum clip length in seconds")
    p.add_argument("--max", dest="max_duration", type=float, default=d.max_duration,
                   help="Maximum clip length in seconds")
    p.add_argument("--target", dest="target_duration", type=float, default=d.target_duration,
                   help="Preferred clip length in seconds")

    v = p.add_argument_group("video")
    v.add_argument("--layout", choices=["auto", "crop", "split", "blur"], default=d.layout,
                   help="auto: split-screen for two-person conversations, speaker-tracking crop "
                        "when there are faces, blurred background otherwise")
    v.add_argument("--no-speaker-detection", action="store_true",
                   help="Follow the largest face instead of whoever is talking")
    v.add_argument("--resolution", default=f"{d.width}x{d.height}", help="Output WxH")
    v.add_argument("--crf", type=int, default=d.crf)
    v.add_argument("--preset", default=d.preset)

    c = p.add_argument_group("captions")
    c.add_argument("--no-captions", action="store_true")
    c.add_argument("--no-title", action="store_true", help="Don't overlay the hook title")
    c.add_argument("--emoji", action="store_true", help="Pop emoji above captions")
    c.add_argument("--font", dest="caption_font", default=d.caption_font)
    c.add_argument("--font-size", dest="caption_size", type=int, default=d.caption_size)
    c.add_argument("--highlight-color", dest="caption_highlight", default=d.caption_highlight,
                   help="Hex RGB colour of the active word (default 00E5FF)")
    c.add_argument("--words-per-caption", dest="caption_max_words", type=int,
                   default=d.caption_max_words)
    c.add_argument("--no-uppercase", action="store_true")

    b = p.add_argument_group("b-roll")
    b.add_argument("--broll", action="store_true",
                   help="Cut away to stock footage (needs PEXELS_API_KEY or --broll-dir)")
    b.add_argument("--broll-dir", help="Folder of your own clips; file names are matched")
    b.add_argument("--broll-max", type=int, default=d.broll_max, help="Cutaways per clip")

    a = p.add_argument_group("analysis")
    a.add_argument("--whisper-model", default=d.whisper_model,
                   help="tiny | base | small | medium | large-v3 | distil-large-v3 ...")
    a.add_argument("--device", dest="whisper_device", default=d.whisper_device)
    a.add_argument("--language", default=None, help="Force transcription language, e.g. en")
    a.add_argument("--transcript", default=None,
                   help="Use an existing transcript JSON instead of running Whisper")
    a.add_argument("--no-llm", action="store_true", help="Heuristic scoring only (no AI model)")
    a.add_argument("--llm-provider", default=d.llm_provider, choices=["auto", "anthropic", "gemini"],
                   help="auto: Claude if ANTHROPIC_API_KEY is set, else Gemini if GEMINI_API_KEY is set")
    a.add_argument("--llm-model", default=d.llm_model,
                   help="Override the model, e.g. claude-sonnet-5-5 or gemini-2.5-pro")
    a.add_argument("--effort", dest="llm_effort", default=d.llm_effort,
                   choices=["low", "medium", "high", "xhigh", "max"], help="Claude only")

    dl = p.add_argument_group("download")
    dl.add_argument("--cookies", default=os.environ.get("YTDLP_COOKIES"),
                    help="cookies.txt for yt-dlp (YouTube on servers)")
    dl.add_argument("--cookies-from-browser", default=os.environ.get("YTDLP_COOKIES_FROM_BROWSER"),
                    help="Read yt-dlp cookies from a browser, e.g. chrome")
    p.add_argument("--dry-run", action="store_true", help="Find and score clips, don't render")


def settings_from_args(args) -> Settings:
    w, _, h = args.resolution.lower().partition("x")
    s = Settings(
        num_clips=args.num_clips, min_duration=args.min_duration,
        max_duration=args.max_duration, target_duration=args.target_duration,
        whisper_model=args.whisper_model, whisper_device=args.whisper_device,
        language=args.language, use_llm=not args.no_llm, llm_provider=args.llm_provider, llm_model=args.llm_model,
        llm_effort=args.llm_effort, width=int(w), height=int(h), layout=args.layout,
        speaker_detection=not args.no_speaker_detection,
        crf=args.crf, preset=args.preset, captions=not args.no_captions,
        caption_font=args.caption_font, caption_size=args.caption_size,
        caption_highlight=args.caption_highlight, caption_max_words=args.caption_max_words,
        caption_uppercase=not args.no_uppercase, show_title=not args.no_title,
        emoji=args.emoji, broll=args.broll or bool(args.broll_dir), broll_dir=args.broll_dir,
        broll_max=args.broll_max,
        cookies_file=args.cookies, cookies_from_browser=args.cookies_from_browser,
    )
    s.target_duration = min(max(s.target_duration, s.min_duration), s.max_duration)
    return s


def cmd_run(args) -> int:
    from .pipeline import run_pipeline

    if args.min_duration >= args.max_duration:
        print("--min must be smaller than --max", file=sys.stderr)
        return 2
    s = settings_from_args(args)
    try:
        clips = run_pipeline(args.source, args.out, s, transcript_path=args.transcript,
                             render=not args.dry_run)
    except Exception as exc:  # surface a clean message on the CLI
        print(f"error: {exc}", file=sys.stderr)
        return 1
    failed = [c for c in clips if c.status == "error"]
    print(f"\nDone: {len(clips) - len(failed)} clips -> {args.out}/ "
          f"(scores in {args.out}/clips.json)")
    return 1 if failed and not args.dry_run else 0


def cmd_doctor(args) -> int:
    from .doctor import print_report, run_checks

    checks = run_checks(args.whisper_model)
    if args.json:
        print(json.dumps([c.to_dict() for c in checks], indent=2))
        return 0
    return print_report(checks)


def cmd_setup(args) -> int:
    from .doctor import setup

    return setup(args.whisper_model)


def cmd_serve(args) -> int:
    try:
        from .web.server import serve
    except ImportError:
        print('The web UI needs extra packages: pip install -e ".[web]"', file=sys.stderr)
        return 1
    serve(host=args.host, port=args.port, data_dir=args.data_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clipagent",
        description="Turn a long video into ranked, captioned 9:16 short-form clips.")
    sub = p.add_subparsers(dest="command", required=True)

    add_run_args(sub.add_parser("run", help="Make clips from a video or URL"))

    d = sub.add_parser("doctor", help="Check that everything needed is installed and configured")
    d.add_argument("--whisper-model", default=Settings().whisper_model)
    d.add_argument("--json", action="store_true")

    s = sub.add_parser("setup", help="Download the Whisper and face models")
    s.add_argument("--whisper-model", default=Settings().whisper_model)

    w = sub.add_parser("serve", help="Start the web UI")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--data-dir", default=os.environ.get("CLIPAGENT_DATA_DIR", "clipagent-data"))

    return p


def main(argv: list[str] | None = None) -> int:
    load_env()
    argv = list(sys.argv[1:] if argv is None else argv)
    # `clipagent video.mp4 ...` is shorthand for `clipagent run video.mp4 ...`
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "run")
    args = build_parser().parse_args(argv)
    return {"run": cmd_run, "doctor": cmd_doctor, "setup": cmd_setup,
            "serve": cmd_serve}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
