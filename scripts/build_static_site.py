"""Build the public website (landing page) as plain static files, e.g. for Vercel.

    python scripts/build_static_site.py https://your-site.vercel.app [out_dir]

Vercel runs this automatically (see vercel.json at the repo root).

The studio itself needs ffmpeg, Whisper and background jobs, so it can't run on
a static host. On the static site, /app is a page explaining how to run it
locally (`clipagent serve`).
"""

from __future__ import annotations

import html
import os
import shutil
import sys
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "clipagent" / "web" / "static"
ASSETS = ["theme.css", "landing.css", "landing.js", "favicon.svg", "og.jpg"]


def build(base_url: str, out: Path, studio_url: str = "") -> Path:
    """`studio_url`: where a hosted studio runs (e.g. on Modal). When set, every
    "Open studio" link points there instead of the run-it-locally page."""
    base = html.escape(base_url.rstrip("/"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    landing = (STATIC / "index.html").read_text().replace("__BASE_URL__", base)
    if studio_url:
        landing = landing.replace('href="/app"', f'href="{html.escape(studio_url.rstrip("/"))}/app"')
    (out / "index.html").write_text(landing)
    (out / "app.html").write_text((STATIC / "run-locally.html").read_text().replace("__BASE_URL__", base))
    for name in ASSETS:
        shutil.copy(STATIC / name, out / name)
    (out / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\nDisallow: /app\n\nSitemap: {base}/sitemap.xml\n")
    (out / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{base}/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>\n"
        "</urlset>\n")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = build(sys.argv[1], Path(sys.argv[2] if len(sys.argv) > 2 else "site"),
                 studio_url=os.environ.get("CLIPAGENT_STUDIO_URL", ""))
    print(f"Built {path}/ - deploy that folder as a static site.")
