"""End-to-end: long video in, ranked vertical clips out.

`Project` keeps everything for one source video in a folder so clips can be
re-rendered later (the web editor uses this):

    out_dir/
      clips.json          manifest: settings, source info, every clip
      01_84_title.mp4     rendered clips
      .work/              source download, transcript, loudness, subtitles
"""

from __future__ import annotations

import dataclasses
import json
import re
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .captions import caption_layout, emoji_events, write_ass
from .config import Settings
from .highlights import heuristic_highlights, score_clip, select_top, snap_to_words
from .media import VideoInfo, download, extract_audio, is_url, loudness_envelope, probe
from .models import Clip, ScoreBreakdown, Transcript
from .transcribe import load_transcript, save_transcript, transcribe

Progress = Callable[[str, float], None]


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


class Project:
    MANIFEST = "clips.json"

    def __init__(self, out_dir: str | Path, settings: Settings, log=print,
                 progress: Optional[Progress] = None):
        self.out = Path(out_dir)
        self.work = self.out / ".work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.log = log
        self.progress = progress or (lambda stage, frac: None)
        self.source: str = ""
        self.video: Optional[Path] = None
        self.info: Optional[VideoInfo] = None
        self.transcript: Optional[Transcript] = None
        self.loudness: Optional[np.ndarray] = None
        self.clips: list[Clip] = []

    # ------------------------------------------------------------------ setup
    def prepare(self, source: str, transcript_path: str | None = None) -> None:
        s = self.settings
        self.source = source
        self.progress("download", 0.0)
        if is_url(source):
            self.log(f"Downloading {source} ...")
            self.video = download(source, self.work, s.cookies_file, s.cookies_from_browser,
                                  max_minutes=s.max_source_minutes)
        else:
            self.video = Path(source).resolve()
            if not self.video.exists():
                raise FileNotFoundError(source)
        self.info = probe(self.video)
        i = self.info
        if s.max_source_minutes and i.duration > s.max_source_minutes * 60:
            raise RuntimeError(f"This video is {i.duration / 60:.0f} minutes long; the limit here "
                               f"is {s.max_source_minutes:g} minutes.")
        self.log(f"Source: {self.video.name} {i.width}x{i.height} {i.duration:.1f}s @ {i.fps:.2f}fps")

        self.progress("transcribe", 0.1)
        cache = self.work / "transcript.json"
        if transcript_path:
            self.transcript = load_transcript(transcript_path)
            save_transcript(self.transcript, cache)
        elif cache.exists():
            self.log("Using cached transcript.")
            self.transcript = load_transcript(cache)
        else:
            if not i.has_audio:
                raise RuntimeError("The video has no audio track to transcribe.")
            audio = extract_audio(self.video, self.work / "audio.wav")
            self.transcript = transcribe(audio, s, log=self.log)
            save_transcript(self.transcript, cache)
            audio.unlink(missing_ok=True)
        if not self.transcript.words:
            raise RuntimeError("No speech found in the video.")

        loud_cache = self.work / "loudness.npy"
        if loud_cache.exists():
            self.loudness = np.load(loud_cache)
        elif i.has_audio:
            self.loudness = loudness_envelope(self.video)
            np.save(loud_cache, self.loudness)
        self.save()

    @classmethod
    def load(cls, out_dir: str | Path, settings: Settings | None = None, log=print,
             progress: Optional[Progress] = None) -> "Project":
        out = Path(out_dir)
        data = json.loads((out / cls.MANIFEST).read_text())
        base = Settings(**{k: v for k, v in data.get("settings", {}).items()
                           if k in Settings.__dataclass_fields__})
        p = cls(out, settings or base, log=log, progress=progress)
        p.source = data["source"]
        p.video = Path(data["video"])
        p.info = probe(p.video)
        p.transcript = load_transcript(p.work / "transcript.json")
        loud = p.work / "loudness.npy"
        p.loudness = np.load(loud) if loud.exists() else None
        p.clips = [Clip.from_dict(c) for c in data.get("clips", [])]
        return p

    def save(self) -> None:
        manifest = {
            "source": self.source,
            "video": str(self.video) if self.video else None,
            "duration": self.info.duration if self.info else None,
            "width": self.info.width if self.info else None,
            "height": self.info.height if self.info else None,
            "settings": dataclasses.asdict(self.settings),
            "clips": [c.to_dict() for c in self.clips],
        }
        tmp = self.out / (self.MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        tmp.replace(self.out / self.MANIFEST)

    # ------------------------------------------------------------------ steps
    def find(self) -> list[Clip]:
        self.progress("analyze", 0.3)
        self.clips = find_clips(self.transcript, self.loudness, self.settings, log=self.log)
        for c in self.clips:
            c.id = c.id or uuid.uuid4().hex[:8]
            if not c.description:
                c.description = c.title
        self.log(f"Selected {len(self.clips)} clips.")
        self.save()
        return self.clips

    def settings_for(self, clip: Clip) -> Settings:
        allowed = set(Settings.clip_option_keys())
        return dataclasses.replace(self.settings,
                                   **{k: v for k, v in clip.options.items() if k in allowed})

    def render(self, clip: Clip, rank: int) -> Clip:
        from .broll import prepare_broll
        from .emoji import EMOJI_SIZE, render_emoji
        from .reframe import plan_reframe
        from .render import Overlays, render_clip

        s = self.settings_for(clip)
        clip.status, clip.error = "rendering", ""
        name = f"{rank:02d}_{int(clip.virality)}_{_slug(clip.title or clip.text)}"
        plan = plan_reframe(self.info, clip.start, clip.end, s.layout, s.width, s.height,
                            loudness=self.loudness, speaker_detection=s.speaker_detection,
                            analysis_fps=s.analysis_fps)
        clip.layout, clip.speakers, clip.speaker_switches = (plan.layout, plan.speakers,
                                                             plan.speaker_switches)
        words = self.transcript.words_between(clip.start, clip.end)
        overlays = Overlays()
        if s.captions or (s.show_title and clip.title):
            overlays.ass_path = write_ass(self.work / f"{name}.ass", words, clip.start, clip.end, s,
                                          title=clip.title, layout=plan.layout)
        clip.emojis = []
        if s.emoji:
            _, _, caption_top = caption_layout(s, plan.layout)
            overlays.emoji_y = max(0, caption_top - EMOJI_SIZE - 24)
            for t0, t1, e in emoji_events(words, clip.start, clip.end, s):
                png = render_emoji(e)
                if png:
                    overlays.emoji.append((t0, t1, str(png)))
                    clip.emojis.append([t0, t1, e])
        clip.broll = []
        if s.broll:
            cuts = prepare_broll(clip, self.transcript, s.broll_max, s.broll_duration,
                                 s.broll_dir, log=self.log)
            overlays.broll = [(c.time, c.duration, c.path) for c in cuts]
            clip.broll = [c.to_dict() for c in cuts]

        out_path = self.out / f"{name}.mp4"
        if clip.output_path and Path(clip.output_path) != out_path:
            Path(clip.output_path).unlink(missing_ok=True)  # replaced by a re-render
        try:
            render_clip(self.info, clip.start, clip.end, plan, overlays, out_path, s)
        except Exception as exc:
            clip.status, clip.error = "error", str(exc)[-500:]
            raise
        clip.output_path = str(out_path)
        clip.status = "done"
        extras = "".join([f" +{len(clip.broll)} b-roll" if clip.broll else "",
                          f" +{len(clip.emojis)} emoji" if clip.emojis else ""])
        self.log(f"  [{rank}] {clip.virality:5.1f}  {clip.duration:5.1f}s  {plan.layout:5s}"
                 f"{extras}  {out_path.name}")
        return clip

    def render_all(self) -> None:
        n = len(self.clips)
        for i, clip in enumerate(self.clips, 1):
            self.progress("render", 0.35 + 0.65 * (i - 1) / max(1, n))
            try:
                self.render(clip, i)
            except Exception as exc:
                self.log(f"  [{i}] failed: {exc}")
            self.save()
        self.progress("done", 1.0)


def run_pipeline(source: str, out_dir: str | Path, settings: Settings,
                 transcript_path: str | None = None, render: bool = True,
                 log=print, progress: Optional[Progress] = None) -> list[Clip]:
    project = Project(out_dir, settings, log=log, progress=progress)
    project.prepare(source, transcript_path)
    clips = project.find()
    if render:
        project.render_all()
    else:
        for rank, c in enumerate(clips, 1):
            log(f"  [{rank}] {c.virality:5.1f}  {c.start:7.1f}-{c.end:7.1f}  {c.title}")
    return clips
