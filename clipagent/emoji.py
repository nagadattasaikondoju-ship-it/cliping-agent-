"""Emoji for captions: pick an emoji for a phrase and render it to a PNG.

libass can't draw colour emoji, so we render each emoji to a transparent PNG
(Noto Color Emoji via Pillow, or a Twemoji download as a fallback) and
overlay it with ffmpeg.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .env import CACHE_DIR

EMOJI_SIZE = 160

# Lower-case word (or word stem) -> emoji. Keep it to words that are clearly visual.
EMOJI_MAP: dict[str, str] = {
    # money & business
    "money": "💰", "cash": "💵", "dollar": "💵", "dollars": "💵", "million": "💰", "millions": "💰",
    "billion": "🤑", "rich": "🤑", "broke": "💸", "debt": "💸",
    "invest": "📈", "investing": "📈", "investment": "📈", "stocks": "📈", "growth": "📈",
    "profit": "📈", "crash": "📉", "loss": "📉", "business": "💼", "company": "🏢", "startup": "🚀",
    "startups": "🚀", "launch": "🚀", "founder": "🧑‍💼", "boss": "😎",
    "career": "💼", "sales": "🤝", "deal": "🤝", "customers": "🛍️", "price": "🏷️", "bitcoin": "🪙",
    "crypto": "🪙", "bank": "🏦", "tax": "🧾", "taxes": "🧾",
    # tech
    "ai": "🤖", "robot": "🤖", "computer": "💻", "laptop": "💻", "code": "👨‍💻", "coding": "👨‍💻",
    "phone": "📱", "app": "📱", "internet": "🌐", "online": "🌐", "data": "📊", "email": "📧",
    "video": "🎥", "camera": "📷", "youtube": "▶️", "tiktok": "🎵", "instagram": "📸",
    # mind & emotion
    "love": "❤️", "heart": "❤️", "hate": "😤", "angry": "😡", "happy": "😄",
    "sad": "😢", "cry": "😭", "cried": "😭", "crying": "😭", "laugh": "😂", "funny": "😂",
    "joke": "😂", "scared": "😱", "afraid": "😨", "fear": "😨", "shocked": "😱", "shocking": "😱",
    "crazy": "🤪", "insane": "🤯", "mind": "🧠", "brain": "🧠", "idea": "💡",
    "ideas": "💡", "smart": "🧠", "genius": "🧠", "secret": "🤫", "secrets": "🤫", "truth": "💯",
    "honest": "💯", "lie": "🤥", "lied": "🤥", "wow": "😮", "amazing": "🤩", "incredible": "🤩",
    "proud": "🥹", "stress": "😩", "tired": "😴", "sleep": "😴", "dream": "💭", "dreams": "💭",
    "confidence": "😎", "cool": "😎", "nervous": "😬", "awkward": "😬", "pain": "🤕", "hurt": "🤕",
    # success & failure
    "win": "🏆", "winning": "🏆", "won": "🏆", "success": "🏆", "successful": "🏆",
    "champion": "🏆", "goal": "🎯", "goals": "🎯", "focus": "🎯", "target": "🎯",
    "fail": "❌", "failed": "❌", "failure": "❌", "mistake": "⚠️", "mistakes": "⚠️",
    "wrong": "❌", "never": "🚫",
    "stop": "🛑", "warning": "⚠️", "danger": "⚠️", "problem": "🧩", "solution": "🔑",
    "fire": "🔥", "hot": "🔥", "viral": "🔥", "boom": "💥", "explode": "💥", "power": "⚡",
    "energy": "⚡", "fast": "⚡", "speed": "⚡", "quick": "⚡", "slow": "🐢",
    "deadline": "⏰", "clock": "⏰", "future": "🔮",
    "rocket": "🚀", "grow": "🌱", "learn": "📚", "learned": "📚", "lesson": "📚",
    "study": "📚", "school": "🏫", "college": "🎓", "university": "🎓", "book": "📖",
    "books": "📖", "teach": "🧑‍🏫", "question": "❓",
    # life
    "family": "👨‍👩‍👧", "kids": "🧒", "baby": "👶", "mom": "👩", "dad": "👨", "friend": "🤝",
    "friends": "🤝", "dating": "💘", "relationship": "💑", "marriage": "💍", "married": "💍",
    "wedding": "💍", "party": "🎉", "celebrate": "🎉", "birthday": "🎂", "music": "🎵",
    "song": "🎶", "food": "🍔", "eating": "🍽️", "coffee": "☕", "pizza": "🍕",
    "diet": "🥗", "healthy": "🥗", "health": "🩺", "doctor": "🩺", "gym": "🏋️", "workout": "💪",
    "strong": "💪", "fitness": "💪", "running": "🏃", "travel": "✈️", "flight": "✈️",
    "car": "🚗", "house": "🏠", "home": "🏠", "city": "🏙️", "world": "🌍",
    "earth": "🌍", "sun": "☀️", "rain": "🌧️", "weather": "🌦️", "water": "💧", "ocean": "🌊",
    "game": "🎮", "games": "🎮", "sport": "⚽", "football": "🏈", "soccer": "⚽", "basketball": "🏀",
    "dog": "🐶", "cat": "🐱", "god": "🙏", "pray": "🙏", "thanks": "🙏", "thank": "🙏",
    "magic": "✨", "beautiful": "✨", "difficult": "😤",
    "huge": "🐘", "tiny": "🤏",
    "eyes": "👀", "listen": "👂",
    "kill": "💀", "dead": "💀", "die": "💀", "death": "💀", "war": "⚔️", "fight": "🥊",
    "police": "🚓", "law": "⚖️", "government": "🏛️", "president": "🏛️", "vote": "🗳️",
}

_TOKEN = re.compile(r"[^\w']+")


def pick_emoji(text: str) -> str | None:
    """Most specific emoji for a phrase (the longest matching word wins)."""
    words = [w for w in (_TOKEN.sub("", t.lower()) for t in text.split()) if w]
    best: tuple[int, str] | None = None
    for w in words:
        e = EMOJI_MAP.get(w) or (EMOJI_MAP.get(w[:-1]) if w.endswith("s") else None)
        if e and (best is None or len(w) > best[0]):
            best = (len(w), e)
    return best[1] if best else None


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "C:/Windows/Fonts/seguiemj.ttf",
]


def _emoji_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    if shutil.which("fc-match"):
        out = subprocess.run(["fc-match", "-f", "%{file}", "Noto Color Emoji"],
                             capture_output=True, text=True).stdout.strip()
        if "emoji" in out.lower() and Path(out).exists():
            return out
    return None


def _twemoji_name(emoji: str) -> str:
    return "-".join(f"{ord(c):x}" for c in emoji if c != "\ufe0f")


def _render_with_font(emoji: str, out: Path) -> bool:
    font_path = _emoji_font()
    if not font_path:
        return False
    try:
        from PIL import Image, ImageDraw, ImageFont

        size = 160 if font_path.endswith(".ttc") else 109  # bitmap emoji fonts need their native size
        font = ImageFont.truetype(font_path, size)
        canvas = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).text((size // 2, size // 3), emoji, font=font, embedded_color=True)
        bbox = canvas.getbbox()
        if not bbox:
            return False
        img = canvas.crop(bbox)
        scale = EMOJI_SIZE / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.LANCZOS)
        img.save(out)
        return True
    except Exception:
        return False


def _render_with_twemoji(emoji: str, out: Path) -> bool:
    url = f"https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{_twemoji_name(emoji)}.png"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = resp.read()
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(data)).convert("RGBA").resize((EMOJI_SIZE, EMOJI_SIZE), Image.LANCZOS)
        img.save(out)
        return True
    except Exception:
        return False


def render_emoji(emoji: str) -> Path | None:
    """Cached transparent PNG for `emoji`, or None if it can't be rendered."""
    cache = CACHE_DIR / "emoji"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{_twemoji_name(emoji)}.png"
    if out.exists():
        return out
    if _render_with_font(emoji, out) or _render_with_twemoji(emoji, out):
        return out
    return None


def emoji_backend() -> str | None:
    font = _emoji_font()
    if font:
        return f"font: {Path(font).name}"
    return "Twemoji download (cdn.jsdelivr.net)" if render_emoji("🔥") else None
