from clipagent.config import Settings
from clipagent.highlights import (
    generate_candidates, heuristic_highlights, overlap_fraction, select_top, snap_to_words,
    split_sentences,
)
from clipagent.models import Clip


def test_split_sentences(transcript):
    sents = split_sentences(transcript.words)
    assert len(sents) == 12
    assert sents[1].text == "Why do most startups fail in their first year?"


def test_candidates_respect_length_band(transcript):
    s = Settings(min_duration=8, max_duration=20, target_duration=12)
    cands = generate_candidates(transcript, s)
    assert cands
    assert all(8 <= c.duration <= 20 for c in cands)


def test_hooky_clip_beats_filler(transcript):
    s = Settings(min_duration=8, max_duration=20, target_duration=12, use_llm=False)
    clips = heuristic_highlights(transcript, None, s)
    best = select_top(clips, 1, 0.25)[0]
    assert "startups fail" in best.text or "secret" in best.text or "mistake" in best.text
    filler = [c for c in clips if c.text.startswith("And then we went to lunch")]
    assert filler and best.virality > max(c.virality for c in filler)


def test_scores_in_range(transcript):
    s = Settings(min_duration=8, max_duration=20, target_duration=12)
    for c in heuristic_highlights(transcript, None, s):
        for v in vars(c.scores).values():
            assert 0 <= v <= 100


def test_select_top_removes_overlaps():
    a = Clip(0, 30, "a"); a.scores.total = 90
    b = Clip(5, 35, "b"); b.scores.total = 80
    c = Clip(40, 70, "c"); c.scores.total = 70
    chosen = select_top([a, b, c], 3, 0.25)
    assert [x.text for x in chosen] == ["a", "c"]
    assert overlap_fraction(a, b) > 0.8


def test_snap_to_words(transcript):
    clip = Clip(start=transcript.words[3].start + 0.1, end=transcript.words[10].end - 0.05, text="")
    snap_to_words(clip, transcript)
    assert clip.start == transcript.words[3].start
    assert clip.end == transcript.words[10].end
    assert clip.text.split()[0] == transcript.words[3].text
