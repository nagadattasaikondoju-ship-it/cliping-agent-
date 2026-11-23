"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Sentence:
    """A run of words that ends in sentence punctuation (or a long pause)."""

    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()


@dataclass
class Transcript:
    words: list[Word]
    language: str = "en"

    def to_dict(self) -> dict:
        return {"language": self.language, "words": [asdict(w) for w in self.words]}

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        return cls(
            words=[Word(**w) for w in data["words"]],
            language=data.get("language", "en"),
        )

    def words_between(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if w.start >= start - 1e-6 and w.end <= end + 1e-6]


@dataclass
class ScoreBreakdown:
    """All component scores are on a 0-100 scale."""

    hook: float = 0.0
    flow: float = 0.0
    value: float = 0.0
    emotion: float = 0.0
    pacing: float = 0.0
    trend: float = 0.0
    total: float = 0.0


@dataclass
class Clip:
    start: float
    end: float
    text: str
    id: str = ""
    title: str = ""
    hook: str = ""
    reason: str = ""
    scores: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    heuristic_score: float = 0.0
    llm_score: Optional[float] = None
    output_path: Optional[str] = None
    layout: Optional[str] = None
    # Claude's B-roll ideas: [{"time": absolute seconds, "query": "city skyline"}]
    broll_queries: list = field(default_factory=list)
    broll: list = field(default_factory=list)  # cuts actually used (see broll.BRollCut)
    emojis: list = field(default_factory=list)  # [(start, end, emoji)] relative to clip
    options: dict = field(default_factory=dict)  # per-clip Settings overrides (web editor)
    speakers: int = 0
    speaker_switches: int = 0
    hashtags: list = field(default_factory=list)
    description: str = ""
    status: str = "pending"  # pending | rendering | done | error
    error: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def virality(self) -> float:
        return self.scores.total

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        d["virality"] = round(self.virality, 1)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        names = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in d.items() if k in names}
        data["scores"] = ScoreBreakdown(**d.get("scores", {}))
        return cls(**data)
