import json
from types import SimpleNamespace

import anthropic

from clipagent.config import Settings
from clipagent.pipeline import find_clips
from tests.conftest import SENTENCES, make_transcript


class _FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def _fake_client(payload, stop_reason="end_turn", calls=None):
    def stream(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        block = SimpleNamespace(type="text", text=json.dumps(payload))
        return _FakeStream(SimpleNamespace(stop_reason=stop_reason, content=[block]))

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))


def _clip(start, end, virality, title):
    return dict(start=start, end=end, title=title, hook="h", reason="r", hook_score=90,
                flow_score=80, value_score=85, emotion_score=70, trend_score=60,
                virality_score=virality, description="desc", hashtags=["#startup", "money"],
                broll=[{"time": start + 5, "query": "counting cash"}, {"time": end + 50, "query": "x"}])


def test_llm_clips_are_snapped_blended_and_ranked(monkeypatch):
    t = make_transcript(SENTENCES * 2)
    calls = []
    payload = {"clips": [_clip(40.0, 58.0, 70, "Second"), _clip(4.0, 22.5, 95, "Why startups die")]}
    monkeypatch.setattr(anthropic, "Anthropic", lambda: _fake_client(payload, calls=calls))
    s = Settings(num_clips=2, min_duration=8, max_duration=25, target_duration=15)
    clips = find_clips(t, None, s, log=lambda *_: None)

    assert [c.title for c in clips] == ["Why startups die", "Second"]
    top = clips[0]
    assert top.llm_score == 95
    assert top.scores.total == round(0.6 * 95 + 0.4 * top.heuristic_score, 1)
    assert any(abs(top.start - w.start) < 1e-9 for w in t.words)  # snapped to a word
    assert top.hashtags == ["startup", "money"]
    assert [b["query"] for b in top.broll_queries] == ["counting cash"]  # out-of-clip idea dropped
    req = calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert "<transcript>" in req["messages"][0]["content"]


def test_refusal_falls_back_to_heuristics(monkeypatch):
    t = make_transcript(SENTENCES * 2)
    monkeypatch.setattr(anthropic, "Anthropic", lambda: _fake_client({"clips": []}, "refusal"))
    s = Settings(num_clips=2, min_duration=8, max_duration=25, target_duration=15)
    clips = find_clips(t, None, s, log=lambda *_: None)
    assert len(clips) == 2 and all(c.llm_score is None for c in clips)
