"""Animated word-by-word captions as an ASS subtitle file (burned in by ffmpeg/libass)."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .models import Word


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_color(rgb_hex: str) -> str:
    rgb = rgb_hex.lstrip("#")
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    return f"&H00{b}{g}{r}".upper()


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def chunk_words(words: list[Word], max_words: int, max_chars: int = 18,
                max_gap: float = 0.6) -> list[list[Word]]:
    """Group words into short on-screen phrases, breaking on punctuation and pauses."""
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            text_len = sum(len(x.text) + 1 for x in cur) + len(w.text)
            if (len(cur) >= max_words or text_len > max_chars
                    or w.start - cur[-1].end > max_gap
                    or cur[-1].text[-1:] in ".!?,;:"):
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def caption_layout(settings: Settings, layout: str) -> tuple[int, int, int]:
    """(ASS alignment, MarginV, y of the caption's top edge in output pixels)."""
    H, size = settings.height, settings.caption_size
    if layout == "split":  # sit on the seam between the two speakers
        return 5, 0, H // 2 - int(size * 0.6)
    margin_v = int(H * 0.26)
    return 2, margin_v, H - margin_v - int(size * 1.2)


def emoji_events(words: list[Word], clip_start: float, clip_end: float, settings: Settings,
                 min_gap: float = 2.5, max_events: int = 8) -> list[tuple[float, float, str]]:
    """(start, end, emoji) relative to the clip, one per caption phrase that has one."""
    from .emoji import pick_emoji

    rel = [Word(w.text, w.start - clip_start, w.end - clip_start) for w in words]
    out: list[tuple[float, float, str]] = []
    for chunk in chunk_words(rel, settings.caption_max_words):
        emoji = pick_emoji(" ".join(w.text for w in chunk))
        if not emoji:
            continue
        start = max(0.0, chunk[0].start)
        if out and start - out[-1][0] < min_gap:
            continue
        end = min(clip_end - clip_start, max(chunk[-1].end + 0.3, start + 0.9), start + 2.0)
        out.append((round(start, 2), round(end, 2), emoji))
        if len(out) >= max_events:
            break
    return out


def build_ass(words: list[Word], clip_start: float, clip_end: float, settings: Settings,
              title: str = "", title_duration: float = 3.0, layout: str = "crop") -> str:
    W, H = settings.width, settings.height
    base = _ass_color(settings.caption_color)
    hi = _ass_color(settings.caption_highlight)
    size = settings.caption_size
    align, margin_v, _ = caption_layout(settings, layout)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{settings.caption_font},{size},{base},{base},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,{align},60,60,{margin_v},1
Style: Title,{settings.caption_font},{int(size * 0.72)},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,18,0,8,80,80,{int(H * 0.12)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    duration = clip_end - clip_start

    if title and settings.show_title:
        t = _escape(title.upper() if settings.caption_uppercase else title)
        events.append(
            f"Dialogue: 1,{_ass_time(0)},{_ass_time(min(title_duration, duration))},Title,,0,0,0,,"
            f"{{\\fad(150,250)}}{t}"
        )

    if settings.captions:
        rel = [Word(w.text, w.start - clip_start, w.end - clip_start) for w in words]
        chunks = chunk_words(rel, settings.caption_max_words)
        for ci, chunk in enumerate(chunks):
            chunk_end = chunks[ci + 1][0].start if ci + 1 < len(chunks) else chunk[-1].end + 0.3
            chunk_end = min(chunk_end, chunk[-1].end + 0.8, duration)
            for wi, w in enumerate(chunk):
                start = w.start
                end = chunk[wi + 1].start if wi + 1 < len(chunk) else chunk_end
                if end <= start:
                    end = start + 0.05
                parts = []
                for k, other in enumerate(chunk):
                    txt = _escape(other.text.upper() if settings.caption_uppercase else other.text)
                    if k == wi:
                        parts.append(f"{{\\c{hi}\\fscx112\\fscy112}}{txt}{{\\c{base}\\fscx100\\fscy100}}")
                    else:
                        parts.append(txt)
                pop = "{\\fscx90\\fscy90\\t(0,80,\\fscx100\\fscy100)}" if wi == 0 else ""
                events.append(
                    f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{pop}"
                    + " ".join(parts)
                )
    return header + "\n".join(events) + "\n"


def write_ass(path: Path, *args, **kwargs) -> Path:
    path.write_text(build_ass(*args, **kwargs), encoding="utf-8")
    return path
