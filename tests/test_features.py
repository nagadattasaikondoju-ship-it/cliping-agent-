import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from clipagent import broll
from clipagent.broll import BRollCut, check_key, fetch, find_local, plan_cuts, search_pexels, search_pixabay
from clipagent.captions import emoji_events
from clipagent.config import Settings
from clipagent.emoji import pick_emoji, render_emoji
from clipagent.env import parse_env
from clipagent.faces import Face, Track, choose_speakers
from clipagent.media import probe
from clipagent.models import Clip, Word
from clipagent.pipeline import Project, run_pipeline
from clipagent.reframe import choose_split
from clipagent.transcribe import save_transcript
from tests.conftest import SENTENCES, make_transcript


# ---------------------------------------------------------------- env / emoji
def test_parse_env():
    env = parse_env('# c\nA=1\nexport B="two words"\nC=x # note\nEMPTY=\nD=   # comment only\nbad line\n')
    assert env == {"A": "1", "B": "two words", "C": "x", "EMPTY": "", "D": ""}


def test_pick_emoji_prefers_specific_words():
    assert pick_emoji("we made a MILLION dollars") == "💰"
    assert pick_emoji("the weather today") == "🌦️"
    assert pick_emoji("and then we went home.") == "🏠"
    assert pick_emoji("nothing visual here") is None


def test_emoji_events_are_spaced():
    words = [Word(w, i * 0.4, i * 0.4 + 0.3) for i, w in
             enumerate("money money money fire fire love love love car car".split())]
    events = emoji_events(words, 0.0, 5.0, Settings(caption_max_words=1), min_gap=1.0)
    starts = [e[0] for e in events]
    assert all(b - a >= 1.0 for a, b in zip(starts, starts[1:]))
    assert events[0][2] == "💰"


def test_render_emoji_png():
    path = render_emoji("🔥")
    if path is None:
        pytest.skip("no emoji font and no network")
    from PIL import Image

    img = Image.open(path)
    assert img.mode == "RGBA" and max(img.size) == 160


# ---------------------------------------------------------------- speakers
def _track(tid, n, cx, active_ranges):
    t = Track(id=tid)
    for i in range(n):
        t.boxes[i] = Face(cx - 50, 300, 100, 120)
        t.activity[i] = 8.0 if any(a <= i < b for a, b in active_ranges) else 0.5
    return t


def test_choose_speakers_switches_without_lag_or_flicker():
    n, fps = 120, 6.0
    times = np.arange(n) / fps
    a = _track(0, n, 400, [(0, 60)])
    b = _track(1, n, 1500, [(60, 120)])
    b.activity[20] = 30.0  # a single noisy spike must not cause a cut
    active = choose_speakers(times, [a, b], np.ones(n, dtype=bool), fps)
    assert set(active[:57]) == {0}
    assert set(active[63:]) == {1}
    assert int(np.sum(np.diff(active) != 0)) == 1


def test_choose_split_needs_two_talking_people_far_apart():
    n = 60
    analysis = SimpleNamespace(
        tracks=[_track(0, n, 400, [(0, 30)]), _track(1, n, 1500, [(30, 60)])],
        active=np.array([0] * 30 + [1] * 30), voiced=np.ones(n, dtype=bool), times=np.arange(n))
    from clipagent.faces import FaceAnalysis

    fa = FaceAnalysis(times=np.arange(n) / 6, src_w=1920, src_h=1080, tracks=analysis.tracks,
                      active=analysis.active, voiced=analysis.voiced)
    assert choose_split(fa, crop_w=608) == [0, 1]
    fa.active = np.zeros(n, dtype=int)  # only one person talks -> no split
    assert choose_split(fa, crop_w=608) is None


# ---------------------------------------------------------------- b-roll
def test_plan_cuts_skips_hook_and_spaces_cuts(transcript):
    clip = Clip(start=0.0, end=40.0, text="")
    clip.broll_queries = [{"time": 1.0, "query": "early"}, {"time": 8.0, "query": "cash"},
                          {"time": 10.0, "query": "too close"}, {"time": 20.0, "query": "city"}]
    cuts = plan_cuts(clip, transcript, max_cuts=3, duration=2.0)
    assert [c.query for c in cuts] == ["cash", "city"]


def test_find_local_matches_filename(tmp_path):
    (tmp_path / "counting-cash-closeup.mp4").write_bytes(b"x")
    (tmp_path / "city.mov").write_bytes(b"x")
    assert find_local("cash", tmp_path).name == "counting-cash-closeup.mp4"
    assert find_local("night city skyline", tmp_path).name == "city.mov"
    assert find_local("ocean", tmp_path) is None


def test_search_pexels_picks_portrait_file():
    payload = {"videos": [{"duration": 10, "user": {"name": "Ann"}, "video_files": [
        {"file_type": "video/mp4", "width": 1920, "height": 1080, "link": "landscape"},
        {"file_type": "video/mp4", "width": 2160, "height": 3840, "link": "4k"},
        {"file_type": "video/mp4", "width": 1080, "height": 1920, "link": "hd-portrait"}]}]}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode()

    seen = {}

    def opener(req, timeout):
        seen["auth"] = req.headers.get("Authorization")
        seen["url"] = req.full_url
        return Resp()

    link, credit = search_pexels("cash", "KEY", opener=opener)
    assert link == "hd-portrait" and credit == "Ann / Pexels"
    assert seen["auth"] == "KEY" and "orientation=portrait" in seen["url"]


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_search_pixabay_prefers_portrait_then_sharp():
    payload = {"hits": [
        {"duration": 12, "user": "Bo", "videos": {
            "large": {"url": "4k", "width": 3840, "height": 2160, "size": 90_000_000},  # too big
            "medium": {"url": "hd", "width": 1920, "height": 1080, "size": 9_000_000},
            "small": {"url": "sd", "width": 1280, "height": 720, "size": 4_000_000}}},
        {"duration": 2, "user": "Short", "videos": {  # too short
            "medium": {"url": "short-portrait", "width": 1080, "height": 1920, "size": 1}}},
    ]}
    seen = {}

    def opener(req, timeout):
        seen["url"] = req.full_url
        return _Resp(payload)

    assert search_pixabay("cash", "K", opener=opener) == ("hd", "Bo / Pixabay")
    assert "key=K" in seen["url"] and "q=cash" in seen["url"]
    payload["hits"].append({"duration": 9, "user": "Vee", "videos": {
        "medium": {"url": "portrait", "width": 1080, "height": 1920, "size": 5_000_000}}})
    assert search_pixabay("cash", "K", opener=opener)[0] == "portrait"


def test_fetch_falls_back_to_pixabay_when_pexels_rejects_key(tmp_path, monkeypatch):
    import urllib.error

    monkeypatch.setattr(broll, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("PEXELS_API_KEY", "bad")
    monkeypatch.setenv("PIXABAY_API_KEY", "good")

    def pexels(*a, **k):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    monkeypatch.setitem(broll.SOURCES, "pexels", ("PEXELS_API_KEY", pexels))
    monkeypatch.setitem(broll.SOURCES, "pixabay", ("PIXABAY_API_KEY", lambda q, k, **kw: ("http://x/v.mp4", "Bo / Pixabay")))

    class Dl:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"video-bytes"

    monkeypatch.setattr(broll.urllib.request, "urlopen", lambda req, timeout: Dl())
    logs = []
    cut = fetch(BRollCut(time=5, duration=2, query="cash"), None, ["pexels", "pixabay"], log=logs.append)
    assert cut and cut.credit == "Bo / Pixabay" and (tmp_path / "broll" / "cash.mp4").read_bytes() == b"video-bytes"
    assert any("HTTP 401" in l and "check the key" in l for l in logs)


def test_check_key_reports_rejection(monkeypatch):
    import urllib.error

    monkeypatch.setenv("PIXABAY_API_KEY", "nope")

    def opener(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, None)

    ok, detail = check_key("pixabay", opener=opener)
    assert not ok and "rejected" in detail
    monkeypatch.delenv("PIXABAY_API_KEY")
    assert check_key("pixabay") == (False, "PIXABAY_API_KEY not set")


# ---------------------------------------------------------------- end to end
def _color_clip(path, color):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"color=c={color}:s=360x640:d=4", "-c:v", "libx264", "-preset", "ultrafast",
                    str(path)], check=True)


def test_end_to_end_with_broll_and_emoji(sample_video, tmp_path):
    broll = tmp_path / "broll"
    broll.mkdir()
    _color_clip(broll / "money.mp4", "red")
    _color_clip(broll / "cash.mp4", "red")
    transcript = make_transcript(SENTENCES * 3)
    tpath = tmp_path / "t.json"
    save_transcript(transcript, tpath)
    s = Settings(num_clips=1, min_duration=14, max_duration=25, target_duration=20, use_llm=False,
                 layout="blur", preset="ultrafast", width=360, height=640, caption_size=30,
                 emoji=True, broll=True, broll_dir=str(broll), broll_duration=1.5)
    out = tmp_path / "out"
    clips = run_pipeline(str(sample_video), out, s, transcript_path=str(tpath), log=lambda *_: None)
    clip = clips[0]
    assert clip.status == "done", clip.error
    assert clip.broll, "expected a b-roll cut on 'money'/'cash'"
    assert clip.emojis
    info = probe(clip.output_path)
    assert (info.width, info.height) == (360, 640)

    # The frame in the middle of the cutaway is the (red) b-roll, not the test pattern.
    t = clip.broll[0]["time"] + 0.7
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", clip.output_path,
                          "-frames:v", "1", "-vf", "scale=36:64", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    px = np.frombuffer(raw, dtype=np.uint8).reshape(64, 36, 3)[5:15].reshape(-1, 3).mean(axis=0)
    assert px[0] > 150 and px[1] < 80 and px[2] < 80, px

    # Re-render from disk with a per-clip override, as the web editor does.
    project = Project.load(out, log=lambda *_: None)
    c = project.clips[0]
    c.options = {"emoji": False, "broll": False, "caption_highlight": "FFD400"}
    c.end -= 2
    project.render(c, 1)
    assert c.status == "done" and not c.emojis and not c.broll
    assert abs(probe(c.output_path).duration - c.duration) < 0.5


# --- YouTube cookies and bot checks -------------------------------------------------

COOKIES = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tabc\n"


def test_cookie_file_from_env_contents(tmp_path, monkeypatch):
    import base64

    from clipagent.media import _cookie_file

    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("YTDLP_COOKIES_TXT", raising=False)
    assert _cookie_file(tmp_path, None) is None
    assert _cookie_file(tmp_path, "/given.txt") == "/given.txt"

    monkeypatch.setenv("YTDLP_COOKIES_TXT", COOKIES)
    path = _cookie_file(tmp_path, None)
    assert open(path).read().strip() == COOKIES.strip()

    monkeypatch.setenv("YTDLP_COOKIES_TXT", base64.b64encode(COOKIES.encode()).decode())
    assert "\tSID\tabc" in open(_cookie_file(tmp_path, None)).read()


def test_youtube_bot_check_gives_a_clear_error(tmp_path, monkeypatch):
    import yt_dlp

    from clipagent.media import MediaError, download

    class Blocked:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=True):
            raise yt_dlp.utils.DownloadError(
                "ERROR: [youtube] x: Sign in to confirm you’re not a bot.")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", Blocked)
    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("YTDLP_COOKIES_TXT", raising=False)
    with pytest.raises(MediaError, match="upload the file instead"):
        download("https://youtu.be/x", tmp_path)
    monkeypatch.setenv("YTDLP_COOKIES_TXT", COOKIES)
    with pytest.raises(MediaError, match="expired or were rejected"):
        download("https://youtu.be/x", tmp_path)
