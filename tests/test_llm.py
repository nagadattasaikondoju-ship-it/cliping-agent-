import json
from types import SimpleNamespace

import anthropic
import pytest

from clipagent.config import Settings
from clipagent.llm import resolve_provider
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


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "GEMINI_MODEL"):
        monkeypatch.delenv(k, raising=False)


def test_llm_clips_are_snapped_blended_and_ranked(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    t = make_transcript(SENTENCES * 2)
    monkeypatch.setattr(anthropic, "Anthropic", lambda: _fake_client({"clips": []}, "refusal"))
    s = Settings(num_clips=2, min_duration=8, max_duration=25, target_duration=15)
    clips = find_clips(t, None, s, log=lambda *_: None)
    assert len(clips) == 2 and all(c.llm_score is None for c in clips)


def test_provider_resolution(monkeypatch):
    s = Settings()
    assert resolve_provider(s)[0] is None
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert resolve_provider(s) == ("gemini", "gemini-flash-latest")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    assert resolve_provider(s) == ("gemini", "gemini-2.5-pro")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert resolve_provider(s) == ("anthropic", "claude-opus-5-5")  # Claude wins when both set
    assert resolve_provider(Settings(llm_provider="gemini")) == ("gemini", "gemini-2.5-pro")
    assert resolve_provider(Settings(llm_model="claude-sonnet-5-5")) == ("anthropic", "claude-sonnet-5-5")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert resolve_provider(Settings(llm_provider="gemini"))[0] is None


class _FakeModels:
    def __init__(self, payload=None, error=None, calls=None):
        self.payload, self.error, self.calls = payload, error, calls

    def generate_content(self, **kwargs):
        if self.calls is not None:
            self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=json.dumps(self.payload))


def _fake_genai(monkeypatch, **kw):
    from google import genai

    seen = {}

    def client(api_key=None):
        seen["api_key"] = api_key
        return SimpleNamespace(models=_FakeModels(**kw))

    monkeypatch.setattr(genai, "Client", client)
    return seen


def test_gemini_provider(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    calls = []
    payload = {"clips": [_clip(4.0, 22.5, 88, "Gemini pick"), {"start": "bad"}]}
    seen = _fake_genai(monkeypatch, payload=payload, calls=calls)
    s = Settings(num_clips=1, min_duration=8, max_duration=25, target_duration=15)
    clips = find_clips(make_transcript(SENTENCES * 2), None, s, log=lambda *_: None)

    assert seen["api_key"] == "gem-key"
    assert clips[0].title == "Gemini pick" and clips[0].llm_score == 88
    req = calls[0]
    assert req["model"] == "gemini-flash-latest"
    cfg = req["config"]
    assert cfg.response_mime_type == "application/json"
    assert "additionalProperties" not in json.dumps(cfg.response_json_schema)
    assert "<transcript>" in req["contents"]


def test_gemini_bad_model_falls_back(monkeypatch):
    pytest.importorskip("google.genai")
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    _fake_genai(monkeypatch, error=errors.ClientError(404, {"error": {"message": "not found"}}))
    logs = []
    s = Settings(num_clips=2, min_duration=8, max_duration=25, target_duration=15)
    clips = find_clips(make_transcript(SENTENCES * 2), None, s, log=logs.append)
    assert len(clips) == 2 and all(c.llm_score is None for c in clips)
    assert any("not found" in line for line in logs)
