# clipagent

An open-source AI clipping agent that turns long videos (podcasts, interviews, streams, talks) into short vertical clips ready to post. It's an open-source alternative to Opus Clip.

```
long video / URL ──► transcribe (Whisper, word timestamps)
                 ──► find highlights (Claude + heuristics)
                 ──► score virality 0-100 (hook · flow · value · emotion · pacing · trend)
                 ──► reframe to 9:16 (face-tracked virtual camera, or blurred background)
                 ──► burn in animated word-by-word captions + hook title
                 ──► clips/01_87_why-startups-fail.mp4 … + clips.json
```

## Features

- **Highlight detection.** Claude reads the timestamped transcript and picks moments that stand on their own: a strong hook, a payoff, a clean ending. A heuristic scorer (hook words, questions, payoff phrases, loudness, speaking pace, dead air, topical keywords) runs alongside it and takes over completely when no API key is set.
- **Virality score.** Each clip gets a 0–100 score with a breakdown (`hook`, `flow`, `value`, `emotion`, `pacing`, `trend`). With Claude enabled, the score is 60% Claude and 40% heuristics.
- **9:16 reframing.** Faces are detected with OpenCV YuNet. A smoothed "virtual camera" uses a dead-zone, so it doesn't jitter, eases onto the subject when they move, and hard-cuts when the active subject jumps (for example, a speaker change). Footage without faces (screen recordings, slides) gets a blurred-background layout so nothing is cropped away.
- **Captions.** Bold, outlined, 1–3 words at a time. The word being spoken is highlighted and pops in. Font, size, color and case are all configurable. The hook title is overlaid for the first 3 seconds.
- **Clean cuts.** Clips start and end on sentence and word boundaries. Audio is loudness-normalised to −14 LUFS, the usual target for social platforms.
- **Any source.** Local files, or any URL that yt-dlp supports (YouTube, Vimeo, X, Twitch VODs, …).

## Requirements

| What | Why | Required? |
|---|---|---|
| Python ≥ 3.10 | runtime | yes |
| **ffmpeg** with libass (standard builds include it) | cutting, reframing, captions, encoding | yes |
| `pip install` deps (numpy, opencv, faster-whisper, anthropic, yt-dlp) | see `pyproject.toml` | yes |
| **`ANTHROPIC_API_KEY`** | Claude highlight detection and scoring (much better clip picks) | optional, but recommended |
| NVIDIA GPU + CUDA | 5–10× faster transcription | optional |
| YouTube cookies (`--cookies` / `--cookies-from-browser`) | YouTube blocks downloads from servers and datacenter IPs | only for YouTube on servers |

The first run downloads the Whisper model (`small` ≈ 480 MB) from Hugging Face and the YuNet face model (≈ 230 KB) from the OpenCV Zoo. Both are cached.

## Install

```bash
# macOS: brew install ffmpeg   |   Ubuntu: sudo apt install ffmpeg
git clone <this repo> && cd cliping-agent-
python -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...   # optional
```

## Usage

```bash
clipagent podcast.mp4                         # 5 clips into ./clips
clipagent "https://youtu.be/..." -n 8 --cookies-from-browser chrome
clipagent talk.mp4 --min 20 --max 45 --layout blur --highlight-color FFD400
clipagent talk.mp4 --dry-run                  # just find and score, no rendering
clipagent talk.mp4 --no-llm                   # fully offline, heuristics only
clipagent talk.mp4 --whisper-model large-v3 --device cuda
```

Run `clipagent --help` for all options (caption font and size, words per caption, resolution, CRF, Claude model and effort, …).

Output:

```
clips/
  01_84_the-biggest-mistake-i-made.mp4
  02_79_why-most-startups-fail.mp4
  ...
  clips.json      # start/end, title, hook, reason, score breakdown, layout per clip
```

### As a library

```python
from clipagent import Settings, run_pipeline

clips = run_pipeline("talk.mp4", "out", Settings(num_clips=3, layout="crop"))
for c in clips:
    print(c.virality, c.title, c.output_path)
```

## How the scoring works

`clipagent/highlights.py` builds every candidate window that starts and ends on a sentence boundary within the length band. It scores each one, then keeps the best non-overlapping set (greedy non-max suppression). `clipagent/llm.py` sends the transcript to Claude (`claude-opus-5-5`, structured JSON output, adaptive thinking) and asks for self-contained clips with per-dimension scores. Claude's picks are snapped to word boundaries, re-scored by the heuristics and blended. If Claude returns too few clips, heuristic picks fill the remaining slots.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests cover segmentation, scoring, overlap suppression, caption timing, the camera path and crop expression, the Claude integration (with a mocked client and the refusal fallback), and a full end-to-end render of a synthetic video in both layouts.

## Roadmap

- Active-speaker detection (lip movement + audio) for multi-person shots. Right now the largest or most consistent face wins.
- Split-screen layout for two-person podcasts
- Web UI and job queue (FastAPI) with an in-browser trim editor
- B-roll, emoji and keyword-emphasis caption styles, background music
- Direct publishing to TikTok, YouTube Shorts and Instagram

## License

MIT
