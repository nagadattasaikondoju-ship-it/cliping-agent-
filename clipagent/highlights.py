"""Candidate generation, heuristic virality scoring and clip selection.

This module works with no network access and no API keys. When Claude is
available (see llm.py) its judgement is blended into the final score.
"""

from __future__ import annotations

import re

import numpy as np

from .config import Settings
from .models import Clip, ScoreBreakdown, Sentence, Transcript, Word

_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*$")
PAUSE_BREAK = 0.9  # seconds of silence that we treat as a sentence boundary

HOOK_WORDS = {
    "secret", "secrets", "never", "always", "nobody", "everyone", "mistake", "mistakes",
    "truth", "actually", "crazy", "insane", "shocking", "biggest", "worst", "best",
    "why", "how", "stop", "wrong", "lie", "lied", "hack", "trick", "million", "billion",
    "free", "money", "rich", "broke", "fail", "failed", "quit", "fired", "warning",
    "imagine", "listen", "honestly", "seriously", "literally", "unbelievable", "first",
}
VALUE_PHRASES = (
    "the reason", "here's", "here is", "the key", "the trick", "the secret", "you need to",
    "you should", "you have to", "the lesson", "what i learned", "the problem", "the answer",
    "step one", "number one", "first thing", "the truth is", "the point is", "because",
    "that's why", "this is why", "the way to", "if you want",
)
EMOTION_WORDS = {
    "love", "hate", "scared", "afraid", "angry", "happy", "cried", "crying", "laugh",
    "amazing", "terrible", "incredible", "hurt", "proud", "shocked", "wow", "omg", "damn",
    "unbelievable", "insane", "crazy", "obsessed",
}
TREND_WORDS = {
    "ai", "chatgpt", "claude", "crypto", "bitcoin", "tiktok", "viral", "startup", "founder",
    "productivity", "fitness", "diet", "dating", "relationship", "mindset", "motivation",
    "investing", "stocks", "side", "hustle", "remote", "layoffs", "creator", "youtube",
}
WEAK_STARTS = {"and", "but", "so", "um", "uh", "like", "yeah", "because", "or", "also", "then"}


def _norm(token: str) -> str:
    return re.sub(r"[^\w']+", "", token.lower())


def split_sentences(words: list[Word]) -> list[Sentence]:
    sentences: list[Sentence] = []
    current: list[Word] = []
    for i, w in enumerate(words):
        current.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt.start - w.end) if nxt else 0.0
        if _SENTENCE_END.search(w.text) or gap >= PAUSE_BREAK or nxt is None:
            sentences.append(Sentence(current))
            current = []
    return sentences


def generate_candidates(transcript: Transcript, settings: Settings) -> list[Clip]:
    """Every window that starts and ends on a sentence boundary and fits the length band."""
    sentences = split_sentences(transcript.words)
    candidates: list[Clip] = []
    for i, first in enumerate(sentences):
        for j in range(i, len(sentences)):
            start, end = first.start, sentences[j].end
            dur = end - start
            if dur > settings.max_duration:
                break
            if dur >= settings.min_duration:
                text = " ".join(s.text for s in sentences[i : j + 1])
                candidates.append(Clip(start=start, end=end, text=text))
    if not candidates and transcript.words:
        # Very short or sentence-less input: fall back to fixed windows.
        total = transcript.words[-1].end
        step = settings.target_duration
        t = transcript.words[0].start
        while t < total:
            end = min(total, t + settings.target_duration)
            words = transcript.words_between(t, end)
            if words:
                candidates.append(Clip(start=words[0].start, end=words[-1].end,
                                       text=" ".join(w.text for w in words)))
            t += step
    return candidates


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def score_clip(clip: Clip, transcript: Transcript, loudness: np.ndarray | None,
               settings: Settings, hop: float = 0.1) -> ScoreBreakdown:
    words = transcript.words_between(clip.start, clip.end)
    if not words:
        return ScoreBreakdown()
    tokens = [_norm(w.text) for w in words]
    lower_text = clip.text.lower()
    dur = max(clip.duration, 1e-3)

    # --- Hook: how strong are the first ~3 seconds?
    hook_words = [w for w in words if w.start < clip.start + 3.5]
    hook_tokens = [_norm(w.text) for w in hook_words]
    hook_text = " ".join(w.text for w in hook_words)
    hook = 0.35
    hook += 0.12 * min(3, sum(t in HOOK_WORDS for t in hook_tokens))
    hook += 0.15 if "?" in hook_text else 0.0
    hook += 0.10 if any(t in {"you", "your", "you're"} for t in hook_tokens) else 0.0
    hook += 0.10 if any(any(c.isdigit() for c in t) for t in hook_tokens) else 0.0
    hook -= 0.25 if tokens[0] in WEAK_STARTS else 0.0

    # --- Flow / completeness: clean start, clean ending, sensible length.
    flow = 0.4
    flow += 0.3 if _SENTENCE_END.search(words[-1].text) else 0.0
    flow += 0.1 if tokens[0] not in WEAK_STARTS else -0.1
    flow += 0.2 * (1 - min(1.0, abs(dur - settings.target_duration) / settings.target_duration))

    # --- Value: does it teach, explain or deliver a payoff?
    value = 0.25 + 0.15 * min(4, sum(p in lower_text for p in VALUE_PHRASES))
    value += 0.1 if any(any(c.isdigit() for c in t) for t in tokens) else 0.0

    # --- Emotion: emotional words, exclamations, and loudness dynamics.
    emotion = 0.2 + 0.1 * min(3, sum(t in EMOTION_WORDS for t in tokens))
    emotion += 0.05 * min(3, clip.text.count("!"))
    if loudness is not None and len(loudness) > 0:
        a, b = int(clip.start / hop), max(int(clip.start / hop) + 1, int(clip.end / hop))
        seg = loudness[a:b]
        voiced = loudness[loudness > -60]
        if len(seg) and len(voiced):
            rel = (float(np.mean(seg)) - float(np.median(voiced))) / 6.0  # +6 dB => +1
            dyn = float(np.std(seg[seg > -60])) / 10.0 if np.any(seg > -60) else 0.0
            emotion += 0.25 * _clip01(0.5 + rel) + 0.15 * _clip01(dyn)

    # --- Pacing: ~2.3-3.6 words/sec is the sweet spot for short-form.
    wps = len(words) / dur
    pacing = 1.0 - min(1.0, max(0.0, 2.3 - wps, wps - 3.6) / 1.5)
    gaps = [b.start - a.end for a, b in zip(words, words[1:])]
    dead_air = sum(g for g in gaps if g > 0.7)
    pacing -= min(0.5, dead_air / dur * 2)

    # --- Trend: topical keywords.
    trend = 0.3 + 0.15 * min(4, len({t for t in tokens if t in TREND_WORDS}))

    parts = {k: round(100 * _clip01(v), 1) for k, v in dict(
        hook=hook, flow=flow, value=value, emotion=emotion, pacing=pacing, trend=trend).items()}
    total = (0.30 * parts["hook"] + 0.20 * parts["flow"] + 0.20 * parts["value"]
             + 0.12 * parts["emotion"] + 0.12 * parts["pacing"] + 0.06 * parts["trend"])
    return ScoreBreakdown(**parts, total=round(total, 1))


def overlap_fraction(a: Clip, b: Clip) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / max(1e-6, min(a.duration, b.duration))


def select_top(clips: list[Clip], n: int, max_overlap: float) -> list[Clip]:
    """Greedy non-maximum suppression on virality score."""
    chosen: list[Clip] = []
    for clip in sorted(clips, key=lambda c: c.virality, reverse=True):
        if all(overlap_fraction(clip, c) <= max_overlap for c in chosen):
            chosen.append(clip)
        if len(chosen) >= n:
            break
    return chosen


def heuristic_highlights(transcript: Transcript, loudness: np.ndarray | None,
                         settings: Settings) -> list[Clip]:
    candidates = generate_candidates(transcript, settings)
    for c in candidates:
        c.scores = score_clip(c, transcript, loudness, settings)
        c.heuristic_score = c.scores.total
    return candidates


def snap_to_words(clip: Clip, transcript: Transcript) -> Clip:
    """Move clip edges onto the nearest word boundaries and refresh its text."""
    if not transcript.words:
        return clip
    starts = np.array([w.start for w in transcript.words])
    ends = np.array([w.end for w in transcript.words])
    i = int(np.argmin(np.abs(starts - clip.start)))
    j = int(np.argmin(np.abs(ends - clip.end)))
    if j < i:
        j = i
    clip.start, clip.end = float(starts[i]), float(ends[j])
    clip.text = " ".join(w.text for w in transcript.words[i : j + 1])
    return clip
