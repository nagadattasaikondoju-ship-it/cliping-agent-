import math

import numpy as np

from clipagent.captions import build_ass, chunk_words
from clipagent.config import Settings
from clipagent.models import Word
from clipagent.reframe import crop_x_expression, smooth_camera


def test_chunk_words_breaks_on_punctuation():
    words = [Word("Hello", 0, .3), Word("there.", .3, .6), Word("How", .7, .9),
             Word("are", .9, 1.1), Word("you", 1.1, 1.3), Word("doing", 1.3, 1.6)]
    chunks = chunk_words(words, max_words=3)
    assert [len(c) for c in chunks] == [2, 3, 1]


def test_ass_has_event_per_word_and_title():
    words = [Word("one", 10.0, 10.4), Word("two", 10.4, 10.8), Word("three", 10.8, 11.2)]
    ass = build_ass(words, 10.0, 12.0, Settings(), title="Big title")
    events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(events) == 4
    assert "BIG TITLE" in events[0]
    assert "0:00:00.00" in events[1]  # times are relative to the clip
    assert "PlayResX: 1080" in ass


def _eval_expr(expr: str, t: float) -> float:
    py = expr.replace("lt(t,", "_lt(t,").replace("gte(t,", "_gte(t,").replace("between(t,", "_between(t,")
    env = dict(t=t, _lt=lambda a, b: float(a < b), _gte=lambda a, b: float(a >= b),
               _between=lambda x, a, b: float(a <= x <= b))
    return eval(py, {}, env)


def test_crop_expression_interpolates():
    kf = [(0.0, 100.0), (2.0, 300.0), (4.0, 300.0)]
    expr = crop_x_expression(kf)
    assert math.isclose(_eval_expr(expr, 0.0), 100.0, abs_tol=0.5)
    assert math.isclose(_eval_expr(expr, 1.0), 200.0, abs_tol=0.5)
    assert math.isclose(_eval_expr(expr, 3.0), 300.0, abs_tol=0.5)
    assert math.isclose(_eval_expr(expr, 9.0), 300.0, abs_tol=0.5)


def test_smooth_camera_ignores_jitter_and_follows_moves():
    times = np.arange(0, 10, 1 / 3)
    centers = np.where(times < 5, 640.0, 1500.0) + np.random.default_rng(0).normal(0, 8, len(times))
    centers[3] = np.nan
    kf = smooth_camera(times, centers, src_w=1920, crop_w=608)
    xs = [x for _, x in kf]
    assert abs(xs[0] - (640 - 304)) < 20       # starts centred on the subject
    assert abs(xs[-1] - (1500 - 304)) < 40     # ends on the new position
    assert all(0 <= x <= 1920 - 608 for x in xs)
    early = [x for t, x in kf if t < 4.5]
    assert max(early) - min(early) < 5         # no jitter while subject is still


def test_camera_holds_then_hard_cuts():
    times = np.arange(0, 20, 1 / 3)
    centers = np.where(times < 10, 400.0, 1500.0)
    kf = smooth_camera(times, centers, src_w=1920, crop_w=608)
    expr = crop_x_expression(kf)
    # Still locked on the first subject just before the switch (no slow drift)...
    assert math.isclose(_eval_expr(expr, 9.5), 400 - 304, abs_tol=1)
    # ...and on the second subject right after it (hard cut, no pan).
    assert math.isclose(_eval_expr(expr, 10.1), 1500 - 304, abs_tol=1)
