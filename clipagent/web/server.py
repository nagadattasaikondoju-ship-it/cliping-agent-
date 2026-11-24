"""FastAPI app for the clipagent web UI.

    clipagent serve                 # http://127.0.0.1:8000
    clipagent serve --host 0.0.0.0  # on a server (set CLIPAGENT_WEB_TOKEN!)
"""

from __future__ import annotations

import dataclasses
import html
import threading
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import Settings
from ..media import is_url, probe
from .jobs import JobStore, Workers, new_clip_dict

STATIC = Path(__file__).parent / "static"
UPLOAD_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mp3", ".wav", ".m4a"}
# Settings the browser may set when creating a job.
JOB_SETTING_KEYS = {
    "num_clips", "min_duration", "max_duration", "target_duration", "layout", "speaker_detection",
    "whisper_model", "language", "use_llm", "llm_effort", "captions", "show_title", "emoji",
    "broll", "broll_max", "caption_highlight", "caption_font", "caption_size", "caption_uppercase",
    "caption_max_words", "width", "height",
}
CLIP_EDIT_KEYS = {"start", "end", "title", "description", "hashtags", "options"}
RENDER_KEYS = {"start", "end", "title", "options"}


def _clean_settings(raw: dict) -> dict:
    defaults = dataclasses.asdict(Settings())
    out = {}
    for k, v in raw.items():
        if k not in JOB_SETTING_KEYS or v is None or v == "":
            continue
        kind = type(defaults[k]) if defaults.get(k) is not None else str
        try:
            out[k] = (v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")) \
                if kind is bool else kind(v)
        except (TypeError, ValueError):
            raise HTTPException(400, f"bad value for {k}")
    if out.get("layout", "auto") not in ("auto", "crop", "split", "blur"):
        raise HTTPException(400, "layout must be auto, crop, split or blur")
    if out.get("caption_highlight") and not re.fullmatch(r"[0-9A-Fa-f]{6}", out["caption_highlight"]):
        raise HTTPException(400, "caption_highlight must be a 6-digit hex colour")
    return out


@dataclasses.dataclass
class PublicLimits:
    """Limits for a studio that anyone on the internet can use (CLIPAGENT_PUBLIC=1)."""

    enabled: bool = False
    max_upload_mb: int = 500
    max_minutes: float = 30
    max_clips: int = 5
    job_ttl_hours: float = 24
    max_queue: int = 8
    whisper_models: tuple = ("base", "small")

    @classmethod
    def from_env(cls) -> "PublicLimits":
        e = os.environ.get
        if e("CLIPAGENT_PUBLIC", "").lower() not in ("1", "true", "yes"):
            return cls()
        return cls(enabled=True,
                   max_upload_mb=int(e("CLIPAGENT_MAX_UPLOAD_MB", 500)),
                   max_minutes=float(e("CLIPAGENT_MAX_MINUTES", 30)),
                   max_clips=int(e("CLIPAGENT_MAX_CLIPS", 5)),
                   job_ttl_hours=float(e("CLIPAGENT_JOB_TTL_HOURS", 24)),
                   max_queue=int(e("CLIPAGENT_MAX_QUEUE", 8)))


def _feature(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return default if not value else value in ("1", "true", "yes", "on")


@dataclasses.dataclass
class Features:
    """Parts of the studio that can be switched off. Both are off by default on a
    public studio: links (YouTube often blocks servers) and the clip editor."""

    links: bool = True      # start a project from a URL
    editing: bool = True    # trim, retitle, restyle and re-render clips; add custom clips

    @classmethod
    def from_env(cls, public: bool) -> "Features":
        return cls(links=_feature("CLIPAGENT_LINKS", not public),
                   editing=_feature("CLIPAGENT_EDITING", not public))


VISITOR_COOKIE = "clipagent_visitor"


def create_app(data_dir: str | Path = "clipagent-data", start_workers: bool = True,
               limits: PublicLimits | None = None, features: Features | None = None) -> FastAPI:
    limits = limits or PublicLimits.from_env()
    features = features or Features.from_env(limits.enabled)
    store = JobStore(data_dir)
    workers = Workers(store) if start_workers else None
    if workers:
        for job_id in store.recover():
            workers.process(job_id)
    if limits.enabled and start_workers:
        def janitor() -> None:
            while True:
                removed = store.cleanup(limits.job_ttl_hours * 3600)
                if removed:
                    print(f"cleanup: removed {removed} expired job(s)")
                time.sleep(1800)
        threading.Thread(target=janitor, daemon=True).start()
    token = os.environ.get("CLIPAGENT_WEB_TOKEN", "")
    app = FastAPI(title="clipagent", docs_url="/api/docs")
    app.state.store, app.state.workers = store, workers
    status_cache: dict = {}

    @app.middleware("http")
    async def auth(request: Request, call_next):
        path = request.url.path
        if token and (path.startswith("/api/") or path.startswith("/files/")) \
                and path not in ("/api/login", "/api/auth"):
            given = request.cookies.get("clipagent_token") or \
                request.headers.get("authorization", "").removeprefix("Bearer ")
            if not secrets.compare_digest(given, token):
                return JSONResponse({"detail": "login required"}, status_code=401)
        # Public mode: an anonymous, random per-browser id. Visitors only ever
        # see the jobs they created.
        visitor = request.cookies.get(VISITOR_COOKIE, "")
        fresh = not re.fullmatch(r"[A-Za-z0-9_-]{20,64}", visitor)
        if fresh:
            visitor = secrets.token_urlsafe(24)
        request.state.visitor = visitor
        response = await call_next(request)
        if fresh and limits.enabled:
            response.set_cookie(VISITOR_COOKIE, visitor, httponly=True, samesite="lax",
                                secure=request.url.scheme == "https", max_age=30 * 24 * 3600)
        return response

    def _owner(request: Request) -> str:
        return request.state.visitor if limits.enabled else ""

    def _job(job_id: str, request: Request) -> dict:
        try:
            job = store.get(job_id)
        except KeyError:
            raise HTTPException(404, "job not found")
        if limits.enabled and not secrets.compare_digest(job.get("owner", ""), _owner(request)):
            raise HTTPException(404, "job not found")  # don't reveal other people's jobs
        return job

    def _clip_urls(job_id: str, clip: dict) -> dict:
        if clip.get("output_path"):
            name = Path(clip["output_path"]).name
            clip["url"] = f"/files/{job_id}/{name}?v={clip.get('version', 0)}"
        return clip

    # ----------------------------------------------------------------- auth
    @app.get("/api/auth")
    def auth_info():
        return {"required": bool(token)}

    @app.post("/api/login")
    def login(password: str = Body(..., embed=True)):
        if not token or not secrets.compare_digest(password, token):
            raise HTTPException(401, "wrong password")
        resp = JSONResponse({"ok": True})
        resp.set_cookie("clipagent_token", token, httponly=True, samesite="strict",
                        max_age=30 * 24 * 3600)
        return resp

    # ----------------------------------------------------------------- info
    @app.get("/api/status")
    def status():
        if not status_cache or time.time() - status_cache["at"] > 60:
            from ..doctor import run_checks

            status_cache.update(at=time.time(), checks=[c.to_dict() for c in run_checks()])
        return {"checks": status_cache["checks"]}

    @app.get("/api/defaults")
    def defaults():
        d = dataclasses.asdict(Settings())
        if limits.enabled:
            d.update(whisper_model=limits.whisper_models[0], num_clips=min(3, limits.max_clips))
        return {k: d[k] for k in sorted(JOB_SETTING_KEYS)}

    @app.get("/api/config")
    def config():
        return {"public": limits.enabled,
                "limits": dataclasses.asdict(limits) if limits.enabled else None,
                "features": dataclasses.asdict(features)}

    def _require_editing() -> None:
        if not features.editing:
            raise HTTPException(403, "Editing clips is turned off on this studio.")

    # ----------------------------------------------------------------- jobs
    @app.get("/api/jobs")
    def list_jobs(request: Request):
        out = []
        owner = _owner(request)
        for j in store.list():
            if limits.enabled and j.get("owner") != owner:
                continue
            m = store.manifest(j["id"])
            out.append({k: j[k] for k in ("id", "created", "source_name", "status", "stage",
                                          "progress", "error")}
                       | {"clips": len(m["clips"]) if m else 0})
        return out

    def _apply_limits(cfg: dict) -> dict:
        cfg = dict(cfg)
        cfg["num_clips"] = max(1, min(int(cfg.get("num_clips", 3)), limits.max_clips))
        if cfg.get("whisper_model") not in limits.whisper_models:
            cfg["whisper_model"] = limits.whisper_models[0]
        cfg["broll_max"] = min(int(cfg.get("broll_max", 2)), 2)
        cfg["max_source_minutes"] = limits.max_minutes
        return cfg

    @app.post("/api/jobs")
    async def create_job(request: Request, url: str = Form(""), settings: str = Form("{}"),
                         file: Optional[UploadFile] = File(None)):
        try:
            cfg = _clean_settings(json.loads(settings or "{}"))
        except json.JSONDecodeError:
            raise HTTPException(400, "settings must be JSON")
        owner = _owner(request)
        if limits.enabled:
            cfg = _apply_limits(cfg)
            jobs = store.list()
            if any(j.get("owner") == owner and j["status"] in ("queued", "running") for j in jobs):
                raise HTTPException(429, "You already have a video processing. Wait for it to finish.")
            if sum(j["status"] in ("queued", "running") for j in jobs) >= limits.max_queue:
                raise HTTPException(503, "The studio is busy right now. Please try again in a few minutes.")
        if file is not None and file.filename:
            ext = Path(file.filename).suffix.lower()
            if ext not in UPLOAD_EXTS:
                raise HTTPException(400, f"unsupported file type {ext}")
            job = store.create("", file.filename, cfg, owner=owner)
            dest = store.dir(job["id"]) / ".work" / f"upload{ext}"
            max_bytes = limits.max_upload_mb * 1024 * 1024 if limits.enabled else None
            written = 0
            with dest.open("wb") as fh:
                while chunk := await file.read(8 * 1024 * 1024):
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        fh.close()
                        store.delete(job["id"])
                        raise HTTPException(413, f"Files are limited to {limits.max_upload_mb} MB here.")
                    fh.write(chunk)
            if limits.enabled:
                try:
                    minutes = probe(dest).duration / 60
                except Exception:
                    store.delete(job["id"])
                    raise HTTPException(400, "That file doesn't look like a video ffmpeg can read.")
                if minutes > limits.max_minutes:
                    store.delete(job["id"])
                    raise HTTPException(413, f"Videos are limited to {limits.max_minutes:g} minutes "
                                             f"here (this one is {minutes:.0f}).")
            job = store.update(job["id"], source=str(dest), upload=True)
        elif url.strip():
            if not features.links:
                raise HTTPException(400, "Links are turned off on this studio. Upload a video file instead.")
            if not is_url(url.strip()):
                raise HTTPException(400, "url must start with http:// or https://")
            job = store.create(url.strip(), url.strip(), cfg, owner=owner)
        else:
            raise HTTPException(400, "upload a file or paste a URL" if features.links else "upload a video file")
        if workers:
            workers.process(job["id"])
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        job = _job(job_id, request)
        job.pop("owner", None)
        m = store.manifest(job_id)
        job["clips"] = [_clip_urls(job_id, c) for c in (m["clips"] if m else [])]
        job["source_info"] = ({k: m.get(k) for k in ("duration", "width", "height")} if m else None)
        job["source_url"] = f"/api/jobs/{job_id}/source" if m and m.get("video") else None
        return job

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str, request: Request):
        job = _job(job_id, request)
        if job["status"] == "running":
            raise HTTPException(409, "job is running")
        store.delete(job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/source")
    def source_video(job_id: str, request: Request):
        _job(job_id, request)
        m = store.manifest(job_id)
        if not m or not m.get("video") or not Path(m["video"]).exists():
            raise HTTPException(404, "source not available")
        return FileResponse(m["video"])

    @app.get("/api/jobs/{job_id}/transcript")
    def transcript(job_id: str, request: Request):
        _job(job_id, request)
        return _load_transcript(job_id)

    def _load_transcript(job_id: str) -> dict:
        path = store.dir(job_id) / ".work" / "transcript.json"
        if not path.exists():
            raise HTTPException(404, "no transcript yet")
        return json.loads(path.read_text())

    # ----------------------------------------------------------------- clips
    def _find(m: dict, clip_id: str) -> dict:
        for c in m["clips"]:
            if c["id"] == clip_id:
                return c
        raise HTTPException(404, "clip not found")

    def _text_between(job_id: str, start: float, end: float) -> str:
        words = _load_transcript(job_id)["words"]
        return " ".join(w["text"] for w in words if w["start"] >= start - 1e-6 and w["end"] <= end + 1e-6)

    def _check_range(m: dict, start: float, end: float) -> None:
        if not (0 <= start < end <= (m.get("duration") or 1e9)):
            raise HTTPException(400, "invalid start/end")
        if end - start < 1 or end - start > 180:
            raise HTTPException(400, "clips must be 1-180 seconds long")

    @app.patch("/api/jobs/{job_id}/clips/{clip_id}")
    def edit_clip(job_id: str, clip_id: str, request: Request, body: dict = Body(...),
                  render: bool = True):
        _job(job_id, request)
        _require_editing()
        changes = {k: v for k, v in body.items() if k in CLIP_EDIT_KEYS}
        if "options" in changes:
            allowed = set(Settings.clip_option_keys())
            changes["options"] = {k: v for k, v in (changes["options"] or {}).items() if k in allowed}
            if "layout" in changes["options"] and changes["options"]["layout"] not in (
                    "auto", "crop", "split", "blur"):
                raise HTTPException(400, "bad layout")
        if "hashtags" in changes:
            changes["hashtags"] = [str(h).strip().lstrip("#") for h in changes["hashtags"] if str(h).strip()]

        def apply(m: dict) -> dict:
            c = _find(m, clip_id)
            start = float(changes.get("start", c["start"]))
            end = float(changes.get("end", c["end"]))
            _check_range(m, start, end)
            if "start" in changes or "end" in changes:
                changes["start"], changes["end"] = round(start, 3), round(end, 3)
                c["text"] = _text_between(job_id, start, end)
                c["duration"] = round(end - start, 2)
            needs_render = any(k in changes and changes[k] != c.get(k) for k in RENDER_KEYS)
            c.update(changes)
            return {"clip": dict(c), "needs_render": needs_render}

        try:
            result = store.edit_manifest(job_id, apply)
        except KeyError:
            raise HTTPException(409, "clips are not ready yet")
        if render and result["needs_render"] and workers:
            workers.render(job_id, clip_id)
        return _clip_urls(job_id, result["clip"])

    @app.post("/api/jobs/{job_id}/clips")
    def add_clip(job_id: str, request: Request, body: dict = Body(...)):
        _job(job_id, request)
        _require_editing()
        start, end = float(body.get("start", 0)), float(body.get("end", 0))
        title = str(body.get("title", "")).strip() or "Custom clip"

        def apply(m: dict) -> dict:
            _check_range(m, start, end)
            clip = new_clip_dict(start, end, title, _text_between(job_id, start, end))
            m["clips"].append(clip)
            return clip

        clip = store.edit_manifest(job_id, apply)
        if workers:
            workers.render(job_id, clip["id"])
        return clip

    @app.delete("/api/jobs/{job_id}/clips/{clip_id}")
    def delete_clip(job_id: str, clip_id: str, request: Request):
        _job(job_id, request)
        def apply(m: dict) -> None:
            c = _find(m, clip_id)
            if c.get("output_path"):
                Path(c["output_path"]).unlink(missing_ok=True)
            m["clips"].remove(c)
        store.edit_manifest(job_id, apply)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/clips/{clip_id}/render")
    def rerender(job_id: str, clip_id: str, request: Request):
        _job(job_id, request)
        _require_editing()
        _find(store.manifest(job_id) or {"clips": []}, clip_id)
        if workers:
            workers.render(job_id, clip_id)
        return {"ok": True}

    # ----------------------------------------------------------------- files
    @app.get("/files/{job_id}/{name}")
    def files(job_id: str, name: str, request: Request):
        _job(job_id, request)
        if "/" in name or name.startswith(".") or not name.endswith(".mp4"):
            raise HTTPException(404)
        path = store.dir(job_id) / name
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path, media_type="video/mp4",
                            headers={"Cache-Control": "no-cache"})

    # ----------------------------------------------------------------- pages / SEO
    def base_url(request: Request) -> str:
        """Public URL for canonical/OG tags: CLIPAGENT_PUBLIC_URL, else the request's host."""
        return (os.environ.get("CLIPAGENT_PUBLIC_URL") or str(request.base_url)).rstrip("/")

    landing_html = (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def landing(request: Request):
        return HTMLResponse(landing_html.replace("__BASE_URL__", html.escape(base_url(request))))

    @app.get("/app", include_in_schema=False)
    def studio():
        return FileResponse(STATIC / "app.html", media_type="text/html",
                            headers={"X-Robots-Tag": "noindex"})

    @app.get("/robots.txt", include_in_schema=False)
    def robots(request: Request):
        return PlainTextResponse(
            "User-agent: *\nAllow: /$\nDisallow: /app\nDisallow: /api/\nDisallow: /files/\n\n"
            f"Sitemap: {base_url(request)}/sitemap.xml\n")

    @app.get("/sitemap.xml", include_in_schema=False)
    def sitemap(request: Request):
        xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
               '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
               f"  <url><loc>{html.escape(base_url(request))}/</loc><changefreq>weekly</changefreq>"
               "<priority>1.0</priority></url>\n</urlset>\n")
        return Response(xml, media_type="application/xml")

    app.mount("/", StaticFiles(directory=STATIC, html=False), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8000, data_dir: str = "clipagent-data") -> None:
    import uvicorn

    if (host not in ("127.0.0.1", "localhost") and not os.environ.get("CLIPAGENT_WEB_TOKEN")
            and not PublicLimits.from_env().enabled):
        print("warning: serving on a public interface without CLIPAGENT_WEB_TOKEN - anyone who "
              "can reach this port can use it.")
    print(f"clipagent web UI: http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}")
    uvicorn.run(create_app(data_dir), host=host, port=port, log_level="warning")
