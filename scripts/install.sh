#!/usr/bin/env bash
# clipagent installer for macOS and Linux.
#   ./scripts/install.sh          # CPU
#   ./scripts/install.sh --gpu    # also install CUDA libraries for Whisper (Linux + NVIDIA)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=0
for arg in "$@"; do
  case "$arg" in
    --gpu) GPU=1 ;;
    -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
  esac
done

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"

say "Checking ffmpeg"
if ! command -v ffmpeg >/dev/null; then
  if [ "$(uname)" = "Darwin" ]; then
    command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
    brew install ffmpeg
  elif command -v apt-get >/dev/null; then
    $SUDO apt-get update && $SUDO apt-get install -y ffmpeg fonts-noto-color-emoji
  elif command -v dnf >/dev/null; then
    $SUDO dnf install -y ffmpeg google-noto-emoji-color-fonts || {
      echo "On Fedora/RHEL enable RPM Fusion for ffmpeg: https://rpmfusion.org/Configuration"; exit 1; }
  elif command -v pacman >/dev/null; then
    $SUDO pacman -S --noconfirm ffmpeg noto-fonts-emoji
  else
    echo "Please install ffmpeg with your package manager, then re-run."; exit 1
  fi
fi
ffmpeg -version | head -1

say "Checking Python (3.10+)"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3,10))'; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.10+ not found. macOS: brew install python · Ubuntu: sudo apt install python3 python3-venv"; exit 1; }
"$PY" --version

say "Creating virtual environment in .venv"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -e ".[all]"

if [ "$GPU" = 1 ]; then
  say "Installing CUDA 12 / cuDNN 9 libraries for Whisper"
  pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10"
  LIBS=$(python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
  echo "export LD_LIBRARY_PATH=\"$LIBS:\${LD_LIBRARY_PATH:-}\"" >> .venv/bin/activate
  export LD_LIBRARY_PATH="$LIBS:${LD_LIBRARY_PATH:-}"
fi

if [ ! -f .env ]; then
  cp .env.example .env
  say "Created .env - add your ANTHROPIC_API_KEY (and other keys) there"
fi

say "Downloading models (Whisper 'small' ~480 MB, face model 230 KB)"
clipagent setup || echo "Model download failed - it will be retried on first use."

say "Checking everything"
clipagent doctor || true

cat <<'MSG'

Done. Next:
  source .venv/bin/activate
  clipagent serve                 # web UI at http://localhost:8000
  clipagent my-video.mp4          # or straight from the command line
MSG
