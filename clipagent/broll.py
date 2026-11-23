"""B-roll: short stock-footage cutaways over the speaker at moments that mention
something visual ("money", "the ocean", "a rocket launch").

Footage comes from a local folder (file names are matched against the query),
then Pexels (PEXELS_API_KEY) and Pixabay (PIXABAY_API_KEY); both keys are free. Moments come from Claude
(it suggests a time and a search query per clip) or, without Claude, from
concrete nouns in the transcript.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .env import CACHE_DIR
from .models import Clip, Transcript

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}

# Concrete, filmable nouns used when Claude isn't available.
VISUAL_WORDS = {
    "money", "cash", "dollars", "bitcoin", "crypto", "stocks", "bank", "office", "city", "cities",
    "car", "cars", "plane", "airport", "travel", "beach", "ocean", "sea", "mountain", "mountains",
    "forest", "desert", "snow", "rain", "storm", "weather", "sun", "space", "rocket", "moon",
    "computer", "laptop", "phone", "smartphone", "robot", "ai", "code", "server", "factory",
    "gym", "workout", "running", "food", "coffee", "pizza", "kitchen", "restaurant", "doctor",
    "hospital", "school", "students", "classroom", "university", "books", "library", "family",
    "kids", "baby", "dog", "cat", "horse", "farm", "music", "concert", "guitar", "piano",
    "party", "wedding", "football", "soccer", "basketball", "stadium", "crowd", "traffic",
    "street", "house", "home", "construction", "fire", "water", "river", "tree", "garden",
    "meeting", "team", "handshake", "interview", "camera", "studio", "map", "world", "earth",
}


@dataclass
class BRollCut:
    time: float  # seconds from clip start
    duration: float
    query: str
    path: str = ""
    credit: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "broll"


def heuristic_moments(clip: Clip, transcript: Transcript) -> list[dict]:
    out = []
    for w in transcript.words_between(clip.start, clip.end):
        token = re.sub(r"[^\w]", "", w.text.lower())
        if token in VISUAL_WORDS:
            out.append({"time": w.start, "query": token})
    return out


def plan_cuts(clip: Clip, transcript: Transcript, max_cuts: int, duration: float,
              skip_start: float = 3.0, min_gap: float = 5.0) -> list[BRollCut]:
    """Pick when to cut away. Never during the hook; cuts are spaced out."""
    moments = list(clip.broll_queries) or heuristic_moments(clip, transcript)
    cuts: list[BRollCut] = []
    for m in moments:
        t = float(m["time"]) - clip.start
        if t < skip_start or t + duration > clip.duration - 1.0:
            continue
        if any(abs(t - c.time) < min_gap for c in cuts):
            continue
        cuts.append(BRollCut(time=round(t, 2), duration=duration, query=str(m["query"]).strip()))
        if len(cuts) >= max_cuts:
            break
    return sorted(cuts, key=lambda c: c.time)


def find_local(query: str, folder: str | Path) -> Path | None:
    words = {w for w in re.split(r"[^a-z0-9]+", query.lower()) if len(w) > 2}
    best, best_hits = None, 0
    for f in sorted(Path(folder).rglob("*")):
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        tokens = set(re.split(r"[^a-z0-9]+", f.stem.lower()))
        hits = len(words & tokens)
        if hits > best_hits:
            best, best_hits = f, hits
    return best


def search_pexels(query: str, api_key: str, out_w: int = 1080, out_h: int = 1920,
                  opener=urllib.request.urlopen) -> tuple[str, str] | None:
    """(download URL, credit) of the best portrait clip for `query`, or None."""
    url = "https://api.pexels.com/videos/search?" + urllib.parse.urlencode(
        {"query": query, "orientation": "portrait", "size": "medium", "per_page": 8})
    req = urllib.request.Request(url, headers={"Authorization": api_key,
                                               "User-Agent": "clipagent/0.2"})
    with opener(req, timeout=20) as resp:
        data = json.loads(resp.read())
    best = None
    for video in data.get("videos", []):
        if video.get("duration", 0) < 3:
            continue
        for f in video.get("video_files", []):
            w, h = f.get("width") or 0, f.get("height") or 0
            if f.get("file_type") != "video/mp4" or h < w or h < out_h * 0.5:
                continue
            score = abs(h - out_h)  # closest to the output size, no needless 4K
            if best is None or score < best[0]:
                credit = f"{video.get('user', {}).get('name', 'Pexels')} / Pexels"
                best = (score, f["link"], credit)
    return (best[1], best[2]) if best else None


def search_pixabay(query: str, api_key: str, out_h: int = 1920, max_bytes: int = 60_000_000,
                   opener=urllib.request.urlopen) -> tuple[str, str] | None:
    """(download URL, credit) of the best clip on Pixabay for `query`, or None.

    Pixabay is mostly landscape, which we crop to 9:16, so prefer portrait clips,
    then ones tall enough to crop without blurring, without downloading huge 4K files.
    """
    url = "https://pixabay.com/api/videos/?" + urllib.parse.urlencode(
        {"key": api_key, "q": query[:100], "per_page": 10, "safesearch": "true"})
    req = urllib.request.Request(url, headers={"User-Agent": "clipagent/0.2"})
    with opener(req, timeout=20) as resp:
        data = json.loads(resp.read())
    best = None
    for rank, hit in enumerate(data.get("hits", [])):
        if hit.get("duration", 0) < 3:
            continue
        for variant in (hit.get("videos") or {}).values():
            w, h = variant.get("width") or 0, variant.get("height") or 0
            if not variant.get("url") or not h or (variant.get("size") or 0) > max_bytes:
                continue
            portrait = h >= w
            # Landscape is cropped to a strip h*9/16 wide, so its height is what matters.
            sharp = h >= out_h * (0.5 if portrait else 0.75)
            score = (not portrait, not sharp, rank, abs(h - out_h))
            if best is None or score < best[0]:
                best = (score, variant["url"], f"{hit.get('user', 'Pixabay')} / Pixabay")
    return (best[1], best[2]) if best else None


SOURCES = {"pexels": ("PEXELS_API_KEY", search_pexels), "pixabay": ("PIXABAY_API_KEY", search_pixabay)}


def configured_sources() -> list[str]:
    return [name for name, (env, _) in SOURCES.items() if os.environ.get(env)]


def fetch(cut: BRollCut, broll_dir: str | None, sources: list[str], log=print) -> BRollCut | None:
    """Find footage for `cut`: your folder first, then each stock site in turn."""
    if broll_dir:
        local = find_local(cut.query, broll_dir)
        if local:
            cut.path, cut.credit = str(local), "local"
            return cut
    if not sources:
        return None
    cache = CACHE_DIR / "broll"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{_slug(cut.query)}.mp4"
    meta = target.with_suffix(".json")
    if target.exists() and meta.exists():
        cut.path, cut.credit = str(target), json.loads(meta.read_text()).get("credit", "")
        return cut
    for name in sources:
        env, search = SOURCES[name]
        try:
            hit = search(cut.query, os.environ[env])
        except urllib.error.HTTPError as exc:
            hint = " - check the key" if exc.code in (400, 401, 403) else ""
            log(f"  b-roll: {name} returned HTTP {exc.code} for '{cut.query}'{hint}")
            continue
        except Exception as exc:
            log(f"  b-roll: {name} lookup failed for '{cut.query}' ({exc.__class__.__name__})")
            continue
        if not hit:
            log(f"  b-roll: nothing on {name} for '{cut.query}'")
            continue
        link, credit = hit
        try:
            req = urllib.request.Request(link, headers={"User-Agent": "clipagent/0.2"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                target.write_bytes(resp.read())
        except Exception as exc:
            log(f"  b-roll: download from {name} failed ({exc.__class__.__name__})")
            target.unlink(missing_ok=True)
            continue
        meta.write_text(json.dumps({"query": cut.query, "credit": credit, "url": link}))
        cut.path, cut.credit = str(target), credit
        return cut
    return None


def check_key(name: str, opener=urllib.request.urlopen) -> tuple[bool, str]:
    """Make one real search to see whether a stock-footage key works."""
    env, search = SOURCES[name]
    key = os.environ.get(env)
    if not key:
        return False, f"{env} not set"
    try:
        hit = search("city", key, opener=opener)
        return True, "key works" if hit else "key works (no result for test search)"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: the key was rejected" if exc.code in (400, 401, 403) \
            else f"HTTP {exc.code}"
    except Exception as exc:
        return False, f"could not connect ({exc.__class__.__name__})"


def prepare_broll(clip: Clip, transcript: Transcript, max_cuts: int, duration: float,
                  broll_dir: str | None = None, log=print) -> list[BRollCut]:
    sources = configured_sources()
    if not broll_dir and not sources:
        log("  b-roll: skipped (set PEXELS_API_KEY or PIXABAY_API_KEY, or pass --broll-dir)")
        return []
    cuts = []
    for cut in plan_cuts(clip, transcript, max_cuts, duration):
        got = fetch(cut, broll_dir, sources, log=log)
        if got:
            cuts.append(got)
    return cuts
