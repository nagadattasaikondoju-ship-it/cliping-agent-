"""Auto-posting finished clips to YouTube Shorts, TikTok and Instagram Reels.

Each platform needs credentials from its own developer program; see
.env.example and the README. Nothing is posted unless you run
`clipagent publish ...` or press Publish in the web UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PLATFORMS = ("youtube", "tiktok", "instagram")


@dataclass
class PostRequest:
    video: Path
    title: str
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    privacy: str = "private"  # private | unlisted | public

    def caption(self, limit: int = 2200) -> str:
        tags = " ".join(h if h.startswith("#") else f"#{h}" for h in self.hashtags)
        parts = [p for p in (self.title, self.description, tags) if p]
        return "\n\n".join(parts)[:limit]


@dataclass
class PostResult:
    platform: str
    ok: bool
    id: str = ""
    url: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return dict(platform=self.platform, ok=self.ok, id=self.id, url=self.url, detail=self.detail)


class PublishError(RuntimeError):
    pass


def _module(platform: str):
    if platform == "youtube":
        from . import youtube as m
    elif platform == "tiktok":
        from . import tiktok as m
    elif platform == "instagram":
        from . import instagram as m
    else:
        raise PublishError(f"Unknown platform '{platform}'. Choose from {', '.join(PLATFORMS)}")
    return m


def publisher_status() -> dict[str, tuple[bool, str]]:
    return {p: _module(p).status() for p in PLATFORMS}


def publish(platform: str, req: PostRequest, log=print) -> PostResult:
    if not req.video.exists():
        return PostResult(platform, False, detail=f"file not found: {req.video}")
    try:
        return _module(platform).upload(req, log=log)
    except PublishError as exc:
        return PostResult(platform, False, detail=str(exc))
    except Exception as exc:  # network errors etc.
        return PostResult(platform, False, detail=f"{exc.__class__.__name__}: {exc}")
