"""End-to-end: long video in, ranked vertical clips out."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .captions import write_ass
from .config import Settings
from .highlights import heuristic_highlights, score_clip, select_top, snap_to_words
from .media import extract_audio, download, is_url, loudness_envelope, probe
from .models import Clip, ScoreBreakdown, Transcript
from .reframe import plan_reframe
from .render import render_clip
from .transcribe import load_transcript, save_transcript, transcribe


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s_-]+", "-", s)[:n].strip("-") or "clip"


def _blend(llm_clip: Clip, heur: ScoreBreakdown, w: float) -> ScoreBreakdown:
    l = llm_clip.scores
    mix = lambda a, b: round(w * a + (1 - w) * b, 1)  # noqa: E731
    return ScoreBreakdown(
        hook=mix(l.hook, heur.hook), flow=mix(l.flow, heur.flow), value=mix(l.value, heur.value),
        emotion=mix(l.emotion, heur.emotion), pacing=heur.pacing, trend=mix(l.trend, heur.trend),
        total=mix(l.total, heur.total),
    )


def find_clips(transcript: Transcript, loudness, settings: Settings, log=print) -> list[Clip]:
    heuristic = heuristic_highlights(transcript, loudness, settings)
    chosen: list[Clip] = []

    if settings.use_llm:
        from .llm import llm_highlights

        llm = llm_highlights(transcript, settings, log=log)
        for c in llm:
            snap_to_words(c, transcript)
            if not (settings.min_duration * 0.8 <= c.duration <= settings.max_duration * 1.15):
                continue
            heur = score_clip(c, transcript, loudness, settings)
            c.heuristic_score = heur.total
            c.scores = _blend(c, heur, settings.llm_weight)
        chosen = select_top([c for c in llm if c.text], settings.num_clips, settings.max_overlap)

    if len(chosen) < settings.num_clips:
        pool = select_top(heuristic, len(heuristic), settings.max_overlap)
        for c in pool:
            if len(chosen) >= settings.num_clips:
                break
            if all(min(c.end, o.end) - max(c.start, o.start) <= settings.max_overlap * min(c.duration, o.duration)
                   for o in chosen):
                if not c.title:
                    c.title = " ".join(c.text.split()[:7])
                    c.hook = c.title
                chosen.append(c)

    chosen.sort(key=lambda c: c.virality, reverse=True)
    return chosen


def run_pipeline(source: str, out_dir: str | Path, settings: Settings,
                 transcript_path: str | None = None, render: bool = True,
                 log=print) -> list[Clip]:
    out = Path(out_dir)
    work = out / ".work"
    work.mkdir(parents=True, exist_ok=True)

    if is_url(source):
        log(f"Downloading {source} ...")
        video = download(source, work, settings.cookies_file, settings.cookies_from_browser)
    else:
        video = Path(source)
        if not video.exists():
            raise FileNotFoundError(source)
    info = probe(video)
    log(f"Source: {video.name} {info.width}x{info.height} {info.duration:.1f}s @ {info.fps:.2f}fps")

    cache = work / f"{video.stem}.transcript.json"
    if transcript_path:
        transcript = load_transcript(transcript_path)
    elif cache.exists():
        log("Using cached transcript.")
        transcript = load_transcript(cache)
    else:
        if not info.has_audio:
            raise RuntimeError("The video has no audio track to transcribe.")
        audio = extract_audio(video, work / "audio.wav")
        transcript = transcribe(audio, settings, log=log)
        save_transcript(transcript, cache)
    if not transcript.words:
        raise RuntimeError("No speech found in the video.")

    loudness = loudness_envelope(video) if info.has_audio else None
    clips = find_clips(transcript, loudness, settings, log=log)
    log(f"Selected {len(clips)} clips.")

    for rank, clip in enumerate(clips, 1):
        name = f"{rank:02d}_{int(clip.virality)}_{_slug(clip.title or clip.text)}"
        if render:
            plan = plan_reframe(info, clip.start, clip.end, settings.layout,
                                settings.width, settings.height)
            clip.layout = plan.layout
            ass = None
            if settings.captions or (settings.show_title and clip.title):
                ass = write_ass(work / f"{name}.ass", transcript.words_between(clip.start, clip.end),
                                clip.start, clip.end, settings, title=clip.title)
            path = render_clip(info, clip.start, clip.end, plan, ass, out / f"{name}.mp4", settings)
            clip.output_path = str(path)
            log(f"  [{rank}] {clip.virality:5.1f}  {clip.duration:5.1f}s  {plan.layout:4s}  {path.name}")
        else:
            log(f"  [{rank}] {clip.virality:5.1f}  {clip.start:7.1f}-{clip.end:7.1f}  {clip.title}")

    manifest = {
        "source": str(source),
        "duration": info.duration,
        "clips": [c.to_dict() for c in clips],
    }
    (out / "clips.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return clips
