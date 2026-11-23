"""Load settings from .env files without overriding real environment variables."""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "clipagent"
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "clipagent"


def parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            out[key] = value
    return out


def load_env(paths: list[Path] | None = None) -> list[Path]:
    """Load ./.env then ~/.config/clipagent/.env. Existing variables win."""
    loaded = []
    for path in paths or [Path.cwd() / ".env", CONFIG_DIR / ".env"]:
        if path.is_file():
            for k, v in parse_env(path.read_text()).items():
                if v and not os.environ.get(k):
                    os.environ[k] = v
            loaded.append(path)
    return loaded
