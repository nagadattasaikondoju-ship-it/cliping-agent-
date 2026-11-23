"""Highlight detection and virality scoring with an LLM (Claude or Gemini).

The model reads the time-stamped transcript and proposes self-contained
moments with a hook, a title and per-dimension scores. Set ANTHROPIC_API_KEY
for Claude or GEMINI_API_KEY for Gemini. Everything here is optional: without
a key, or if the call fails, the pipeline falls back to heuristics.
"""

from __future__ import annotations

import json
import os
import re
import time

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


# "auto" for Gemini = the newest Flash model this key can use (see _gemini_candidates).
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "gemini": "auto"}
GEMINI_FALLBACK = "gemini-2.5-flash"


class LLMUnavailable(Exception):
    """The model could not be used; the pipeline falls back to heuristics."""


def resolve_provider(settings: Settings) -> tuple[str | None, str]:
    """(provider, model) to use, or (None, reason) when no AI key is configured.

    `auto` prefers Claude when an Anthropic key is set, otherwise Gemini.
    """
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    provider = settings.llm_provider
    if provider == "auto":
        provider = "anthropic" if has_anthropic else "gemini" if has_gemini else None
        if provider is None:
            return None, "no ANTHROPIC_API_KEY or GEMINI_API_KEY set"
    elif provider == "anthropic" and not has_anthropic:
        return None, "llm_provider is anthropic but ANTHROPIC_API_KEY is not set"
    elif provider == "gemini" and not has_gemini:
        return None, "llm_provider is gemini but GEMINI_API_KEY is not set"
    elif provider not in DEFAULT_MODELS:
        return None, f"unknown llm_provider '{provider}'"
    env_model = os.environ.get("GEMINI_MODEL") if provider == "gemini" else None
    return provider, settings.llm_model or env_model or DEFAULT_MODELS[provider]


def _ask_claude(model: str, system: str, user: str, settings: Settings) -> str:
    import anthropic

    try:
        client = anthropic.Anthropic()
        with client.beta.messages.stream(
            model=model,
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
        raise LLMUnavailable("the Anthropic API key was rejected")
    except anthropic.RateLimitError:
        raise LLMUnavailable("Claude rate limit hit")
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"Claude API error {exc.status_code}")
    except anthropic.APIConnectionError:
        raise LLMUnavailable("could not reach the Claude API")
    except TypeError as exc:  # credentials could not be resolved at request time
        raise LLMUnavailable(str(exc))
    if response.stop_reason == "refusal":
        raise LLMUnavailable("Claude declined this transcript")
    if response.stop_reason == "max_tokens":
        raise LLMUnavailable("Claude's response was truncated")
    return next((b.text for b in response.content if b.type == "text"), "")


def _gemini_schema(schema):
    """Gemini's JSON-schema support doesn't need `additionalProperties`; drop it."""
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema


def _gemini_version(name: str) -> tuple[int, ...] | None:
    """(3, 8) for 'gemini-3.8-flash'; None for previews, lite, image or other variants."""
    m = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash", name)
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def _gemini_candidates(client, wanted: str) -> list[str]:
    """Models to try in order. An explicit model is used as-is; "auto" means the
    stable Flash models this key can call, newest first."""
    if wanted != "auto":
        return [wanted]
    try:
        names = [m.name.split("/", 1)[-1] for m in client.models.list()]
    except Exception:
        names = []
    flash = sorted((n for n in names if _gemini_version(n)), key=_gemini_version, reverse=True)
    if GEMINI_FALLBACK not in flash:
        flash.append(GEMINI_FALLBACK)
    return flash[:4]


def _ask_gemini(model: str, system: str, user: str, settings: Settings, log=print,
                retry_wait: float = 4.0) -> str:
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError:
        raise LLMUnavailable("google-genai is not installed (pip install google-genai)")

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=300_000))
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=_gemini_schema(CLIP_SCHEMA),
        max_output_tokens=32000,
    )
    last = "no Gemini model available"
    for name in _gemini_candidates(client, model):
        for attempt in range(2):
            try:
                log(f"  trying {name}...")
                response = client.models.generate_content(model=name, contents=user, config=config)
                text = response.text or ""
                if text:
                    return text
                last = f"{name} returned an empty response (blocked or truncated)"
                break
            except errors.APIError as exc:
                if exc.code in (400, 401, 403) and "key" in str(exc).lower():
                    raise LLMUnavailable("the Gemini API key was rejected")
                if exc.code == 404:
                    last = f"Gemini model '{name}' not found - set GEMINI_MODEL to one your key can use"
                    break
                if exc.code in (429, 500, 503, 504):  # busy or quota: retry, then try an older model
                    last = f"{name} is busy or over quota ({exc.code})"
                    if attempt == 0:
                        time.sleep(retry_wait)
                    continue
                raise LLMUnavailable(f"Gemini API error {exc.code}")
            except Exception as exc:  # network errors, timeouts
                last = f"could not reach Gemini ({exc.__class__.__name__})"
                break
    raise LLMUnavailable(last)


def llm_highlights(transcript: Transcript, settings: Settings, log=print) -> list[Clip]:
    """Ask Claude or Gemini for highlight clips. Returns [] (and logs why) if unavailable."""
    provider, model = resolve_provider(settings)
    if provider is None:
        log(f"AI highlight detection off ({model}) - using heuristic scoring only.")
        return []

    wanted = max(settings.num_clips + 3, int(settings.num_clips * 1.5))
    system = SYSTEM_PROMPT.format(
        min_d=settings.min_duration, max_d=settings.max_duration, target=settings.target_duration
    )
    user = (
        f"Find the {wanted} best clips in this transcript, best first.\n\n"
        f"<transcript>\n{format_transcript(transcript)}\n</transcript>"
    )

    label = "the newest Gemini Flash model" if model == "auto" else model
    log(f"Asking {label} to find highlights...")
    try:
        if provider == "anthropic":
            text = _ask_claude(model, system, user, settings)
        else:
            text = _ask_gemini(model, system, user, settings, log=log)
        data = json.loads(text)
    except LLMUnavailable as exc:
        log(f"{exc} - using heuristic scoring only.")
        return []
    except json.JSONDecodeError:
        log(f"Could not parse {label}'s response - using heuristic scoring only.")
        return []

    clips: list[Clip] = []
    for item in data.get("clips", []):
        try:
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
                start=start, end=end, text="", title=str(item["title"]).strip(),
                hook=str(item["hook"]).strip(), reason=str(item["reason"]).strip(),
                scores=scores, llm_score=scores.total,
                description=str(item.get("description", "")).strip(),
                hashtags=[h.strip().lstrip("#") for h in item.get("hashtags", []) if h.strip()][:8],
                broll_queries=[{"time": float(b["time"]), "query": b["query"]}
                               for b in item.get("broll", []) if start <= float(b["time"]) <= end],
            ))
        except (KeyError, TypeError, ValueError):
            continue  # skip a malformed item rather than losing the whole answer
    log(f"{label} proposed {len(clips)} clips.")
    return clips
