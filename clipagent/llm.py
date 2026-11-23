"""Highlight detection and virality scoring with Claude.

Claude reads the time-stamped transcript and proposes self-contained moments
with a hook, a title and per-dimension scores. Everything here is optional:
if the SDK or credentials are missing the pipeline falls back to heuristics.
"""

from __future__ import annotations

import json

from .config import Settings
from .highlights import split_sentences
from .models import Clip, ScoreBreakdown, Transcript

SYSTEM_PROMPT = """You are a senior short-form video editor who has cut thousands of viral \
TikToks, Reels and YouTube Shorts from long podcasts, interviews, streams and talks.

You receive a transcript where every sentence is prefixed with its start time in seconds. \
Pick the moments most likely to perform as standalone vertical clips.

A great clip:
- opens with a hook in the first 3 seconds (a bold claim, a question, a surprising number, \
conflict, or a curiosity gap) - never with filler like "so", "and", "um", "yeah";
- is self-contained: a viewer with zero context understands it;
- has a payoff (punchline, insight, story resolution, actionable tip) and ends on a complete \
sentence, not mid-thought;
- carries emotion, tension, humour, or a contrarian take.

Rules:
- Use timestamps from the transcript. `start` must be the timestamp of the first sentence \
you include; `end` must be the end of the last sentence you include (use the next sentence's \
start time if needed).
- Each clip must be between {min_d:.0f} and {max_d:.0f} seconds long, ideally around \
{target:.0f}s.
- Clips must not overlap each other.
- Score every dimension from 0 to 100, honestly; most clips are not 90+.
  hook = strength of the first 3 seconds; flow = self-contained and cleanly ending; \
value = insight, payoff or entertainment; emotion = energy, tension, humour; \
trend = relevance to what audiences currently care about; virality = your overall \
prediction of how well it performs.
- title: a punchy on-screen title of at most 8 words, written in the transcript's language.
- hook: the opening line, quoted from the transcript.
- reason: one sentence on why it will work.
- description: a 1-2 sentence social caption for posting the clip.
- hashtags: 3-6 relevant hashtags without the # sign.
- broll: 0-2 moments inside the clip where a short stock-footage cutaway would \
illustrate what is being said. `time` is the timestamp where the word is spoken; `query` is \
an English stock-video search of 1-3 concrete, filmable words (e.g. "city skyline night", \
"counting cash"). Never in the first 3 seconds. Use an empty list if nothing is visual."""

CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "reason": {"type": "string"},
                    "hook_score": {"type": "number"},
                    "flow_score": {"type": "number"},
                    "value_score": {"type": "number"},
                    "emotion_score": {"type": "number"},
                    "trend_score": {"type": "number"},
                    "virality_score": {"type": "number"},
                    "description": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "broll": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"time": {"type": "number"}, "query": {"type": "string"}},
                            "required": ["time", "query"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "start", "end", "title", "hook", "reason", "hook_score", "flow_score",
                    "value_score", "emotion_score", "trend_score", "virality_score",
                    "description", "hashtags", "broll",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


def format_transcript(transcript: Transcript) -> str:
    lines = [f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in split_sentences(transcript.words)]
    return "\n".join(lines)


def _clamp(x: float) -> float:
    return max(0.0, min(100.0, float(x)))


def llm_highlights(transcript: Transcript, settings: Settings, log=print) -> list[Clip]:
    """Ask Claude for highlight clips. Returns [] (and logs why) if unavailable."""
    try:
        import anthropic
    except ImportError:
        log("anthropic SDK not installed - using heuristic scoring only.")
        return []

    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # missing credentials etc.
        log(f"Claude unavailable ({exc.__class__.__name__}) - using heuristic scoring only.")
        return []

    wanted = max(settings.num_clips + 3, int(settings.num_clips * 1.5))
    system = SYSTEM_PROMPT.format(
        min_d=settings.min_duration, max_d=settings.max_duration, target=settings.target_duration
    )
    user = (
        f"Find the {wanted} best clips in this transcript, best first.\n\n"
        f"<transcript>\n{format_transcript(transcript)}\n</transcript>"
    )

    log(f"Asking {settings.llm_model} to find highlights...")
    try:
        with client.beta.messages.stream(
            model=settings.llm_model,
            max_tokens=32000,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={
                "effort": settings.llm_effort,
                "format": {"type": "json_schema", "schema": CLIP_SCHEMA},
            },
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError:
        log("No valid Anthropic credentials - using heuristic scoring only.")
        return []
    except anthropic.RateLimitError:
        log("Claude rate limit hit - using heuristic scoring only.")
        return []
    except anthropic.APIStatusError as exc:
        log(f"Claude API error {exc.status_code} - using heuristic scoring only.")
        return []
    except anthropic.APIConnectionError:
        log("Could not reach the Claude API - using heuristic scoring only.")
        return []
    except TypeError as exc:  # credentials could not be resolved at request time
        log(f"Claude unavailable ({exc}) - using heuristic scoring only.")
        return []

    if response.stop_reason == "refusal":
        log("Claude declined this transcript - using heuristic scoring only.")
        return []
    if response.stop_reason == "max_tokens":
        log("Claude response was truncated - using heuristic scoring only.")
        return []

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log("Could not parse Claude's response - using heuristic scoring only.")
        return []

    clips: list[Clip] = []
    for item in data.get("clips", []):
        start, end = float(item["start"]), float(item["end"])
        if end <= start:
            continue
        scores = ScoreBreakdown(
            hook=_clamp(item["hook_score"]),
            flow=_clamp(item["flow_score"]),
            value=_clamp(item["value_score"]),
            emotion=_clamp(item["emotion_score"]),
            trend=_clamp(item["trend_score"]),
            total=_clamp(item["virality_score"]),
        )
        clips.append(Clip(
            start=start, end=end, text="", title=item["title"].strip(),
            hook=item["hook"].strip(), reason=item["reason"].strip(),
            scores=scores, llm_score=scores.total,
            description=item.get("description", "").strip(),
            hashtags=[h.strip().lstrip("#") for h in item.get("hashtags", []) if h.strip()][:8],
            broll_queries=[{"time": float(b["time"]), "query": b["query"]}
                           for b in item.get("broll", []) if start <= float(b["time"]) <= end],
        ))
    log(f"Claude proposed {len(clips)} clips.")
    return clips
