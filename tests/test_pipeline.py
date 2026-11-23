import json
import subprocess

import pytest

from clipagent.config import Settings
from clipagent.media import probe
from clipagent.pipeline import run_pipeline
from clipagent.transcribe import save_transcript
from tests.conftest import SENTENCES, make_transcript


@pytest.mark.parametrize("layout", ["blur", "crop"])
def test_end_to_end(sample_video, tmp_path, layout):
    transcript = make_transcript(SENTENCES * 3)
    tpath = tmp_path / "t.json"
    save_transcript(transcript, tpath)
    s = Settings(num_clips=2, min_duration=10, max_duration=25, target_duration=15,
                 use_llm=False, layout=layout, preset="ultrafast", width=540, height=960,
                 caption_size=44)
    out = tmp_path / "out"
    clips = run_pipeline(str(sample_video), out, s, transcript_path=str(tpath), log=lambda *_: None)

    assert len(clips) == 2
    manifest = json.loads((out / "clips.json").read_text())
    assert len(manifest["clips"]) == 2
    assert manifest["clips"][0]["virality"] >= manifest["clips"][1]["virality"]
    for c in clips:
        info = probe(c.output_path)
        assert (info.width, info.height) == (540, 960)
        assert info.has_audio
        assert abs(info.duration - c.duration) < 0.5
        assert c.layout == layout
