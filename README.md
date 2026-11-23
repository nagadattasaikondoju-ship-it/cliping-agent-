# clipagent

An open-source AI clipping agent that turns long videos (podcasts, interviews, streams, talks) into short vertical clips ready to post. It's an open-source alternative to Opus Clip.

```
long video / URL ──► transcribe (Whisper, word timestamps)
                 ──► find highlights + score virality (Claude + heuristics)
                 ──► reframe to 9:16: follow the active speaker · split-screen · blurred full frame
                 ──► animated captions, hook title, emoji, B-roll cutaways
                 ──► review & edit in the web UI ──► post to YouTube Shorts / TikTok / Instagram Reels
```

## Features

| | |
|---|---|
| **Highlight detection** | Claude reads the timestamped transcript and picks self-contained moments with a hook and a payoff. It also writes a title, a posting caption, hashtags and B-roll ideas. Without an API key, a heuristic scorer takes over. |
| **Virality score** | 0–100 per clip, broken down into hook, flow, value, emotion, pacing and trend. |
| **Active-speaker detection** | Faces are tracked through the clip, and the system measures how much each mouth moves compared with the rest of the face, during moments with speech. The camera follows whoever is talking and cuts at the moment the speaker changes. |
| **Split-screen** | When two people take turns talking and don't fit in one vertical frame, each gets half of the screen, stacked top and bottom. |
| **Blur layout** | Screen recordings, slides and other footage without faces keep the whole frame, placed over a blurred copy of itself. |
| **Captions** | 1–3 words at a time. The word being spoken is highlighted and pops in. Font, size, color and case are all adjustable, and the hook title shows for the first 3 seconds. |
| **Emoji** | An emoji pops above the caption on visual words ("money" 💰, "fire" 🔥, "love" ❤️, …). |
| **B-roll** | Short stock-footage cutaways from Pexels (free API key), or from your own folder of clips. |
| **Web UI + editor** | Upload a file or paste a link, watch progress, browse clips by score, trim on the transcript, change title, layout and styling, re-render, download and post. |
| **Auto-posting** | YouTube Shorts (Data API), TikTok (Content Posting API) and Instagram Reels (Graph API). Posts are private by default. |

## What you need

| Item | Needed? | How |
|---|---|---|
| Python 3.10+ and **ffmpeg** | yes | the installer handles both, or use Docker |
| **`ANTHROPIC_API_KEY`** | strongly recommended | [console.anthropic.com](https://console.anthropic.com). Without it, clip picks come from heuristics only. |
| An NVIDIA GPU | optional | Transcription runs 5–10× faster. Use `install.sh --gpu` or the GPU Docker image. |
| `PEXELS_API_KEY` | only for stock B-roll | free at [pexels.com/api](https://www.pexels.com/api/). `--broll-dir` works without it. |
| YouTube cookies | only for YouTube links on servers | `--cookies-from-browser chrome`, or `YTDLP_COOKIES` |
| Platform credentials | only for auto-posting | see [Posting](#posting) |
| Internet on first run | yes | Downloads the Whisper model (`small` ≈ 480 MB) and the face model (230 KB) once, then caches them. |

Run `clipagent doctor` at any time to see what's set up and what's missing.

## Install

**Installer (macOS / Linux)**: installs ffmpeg, creates `.venv`, installs everything, downloads the models and runs `doctor`:

```bash
git clone <this repo> && cd cliping-agent-
./scripts/install.sh            # add --gpu on Linux with an NVIDIA card
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

**Docker:**

```bash
cp .env.example .env            # add your keys
docker compose up --build       # http://localhost:8000
docker compose --profile gpu up --build clipagent-gpu   # NVIDIA GPU (needs nvidia-container-toolkit)
```

**Manual:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"         # core + web UI + posting
cp .env.example .env            # add your keys
clipagent setup                 # download models now instead of on first run
```

Keys are read from the environment, from `./.env`, or from `~/.config/clipagent/.env`.

## Use it

### Web UI

```bash
clipagent serve                 # http://localhost:8000
```

To run it on a server, set `CLIPAGENT_WEB_TOKEN` in `.env` to password-protect it, then start it with `clipagent serve --host 0.0.0.0`.

### Command line

```bash
clipagent podcast.mp4                                   # 5 clips into ./clips
clipagent "https://youtu.be/..." -n 8 --cookies-from-browser chrome
clipagent talk.mp4 --emoji --broll                      # emoji + Pexels B-roll
clipagent talk.mp4 --broll-dir ~/stock                  # B-roll from your own clips
clipagent interview.mp4 --layout split                  # force split-screen
clipagent talk.mp4 --min 20 --max 45 --highlight-color FFD400
clipagent talk.mp4 --dry-run                            # find and score only
clipagent talk.mp4 --no-llm                             # fully offline
```

Output:

```
clips/
  01_84_the-biggest-mistake-i-made.mp4
  02_79_why-most-startups-fail.mp4
  clips.json   # timings, title, caption, hashtags, scores, layout, B-roll, emoji per clip
```

### As a library

```python
from clipagent import Settings, run_pipeline

clips = run_pipeline("talk.mp4", "out", Settings(num_clips=3, emoji=True))
```

## Posting

Everything is private by default (`--privacy` changes it), so you can check a clip before it goes live.

```bash
clipagent publish clips/ --to youtube,tiktok,instagram   # all clips, with their generated captions
clipagent publish clips/ --to youtube --clips 1,3 --privacy public
clipagent publish clips/01_84_title.mp4 --to tiktok --title "..." --hashtags ai,startups
```

| Platform | Setup |
|---|---|
| **YouTube Shorts** | In Google Cloud Console, enable *YouTube Data API v3*, create an OAuth client of type *Desktop app* and download `client_secret.json`. Set `YOUTUBE_CLIENT_SECRETS=/path/client_secret.json`, then run `clipagent auth youtube` once. Until your Google app passes verification, uploads may be locked to private. |
| **TikTok** | Create an app at developers.tiktok.com with the Content Posting API and get a user access token (scope `video.upload`). Set `TIKTOK_ACCESS_TOKEN`. By default clips go to the TikTok inbox, and you finish the post in the app. `TIKTOK_MODE=direct` posts directly (scope `video.publish`). Unaudited apps can only post privately. |
| **Instagram Reels** | Use an Instagram professional account linked to a Facebook app with `instagram_content_publish`. Set `IG_USER_ID` and `IG_ACCESS_TOKEN`. |

## How it works

- `highlights.py` scores every sentence-aligned window and keeps the best set that doesn't overlap. `llm.py` sends the transcript to Claude (`claude-opus-5-5`, structured JSON output). Claude's picks are snapped to word boundaries, re-scored by the heuristics, and the two scores are blended 60/40.
- `faces.py` decodes the clip once at 640 px. It runs YuNet face detection 6×/s, links detections into tracks, and measures mouth motion minus eye-area motion (which cancels head movement) for each face. Only moments with speech count. A hysteresis step picks the speaker so the camera doesn't flicker. Because the analysis is offline, each cut is moved back to the moment the new speaker started, so there is no lag.
- `reframe.py` turns the speaker track into a smooth virtual camera: it stays still for small movements, eases toward the subject when it moves, and hard-cuts on a speaker change. It can also decide on the split-screen layout. The result is compiled into an ffmpeg `crop` expression.
- `render.py` stacks the layers in this order: reframed video → B-roll → captions and title (libass) → emoji. Audio is normalized to −14 LUFS.
- `web/` is a FastAPI app with one worker thread for heavy jobs and one for uploads. Each project is a folder (`clipagent-data/jobs/<id>/`) that can be re-rendered at any time.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The 37 tests cover scoring, captions, emoji, the camera path, speaker selection, split detection, B-roll planning and the Pexels lookup, posting to all three platforms (against fake HTTP sessions), the web API (a full job, edits, re-render, password protection), and end-to-end renders.

`scripts/make_two_speaker_video.py` builds a two-person test video with a known speaker schedule. On it, speaker detection agrees with the ground truth on 98.8% of frames.

## Roadmap

- Bring in the final web design
- Audio-visual speaker models (e.g. TalkNet) for crowded or low-light shots
- Caption style presets, background music, filler-word removal
- Scheduled posting and per-platform analytics

## License

MIT
