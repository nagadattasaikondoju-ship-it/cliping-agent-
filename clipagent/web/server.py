"""FastAPI app for the clipagent web UI.

    clipagent serve                 # http://127.0.0.1:8000
    clipagent serve --host 0.0.0.0  # on a server (set CLIPAGENT_WEB_TOKEN!)
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings
from ..media import is_url
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


def create_app(data_dir: str | Path = "clipagent-data", start_workers: bool = True) -> FastAPI:
    store = JobStore(data_dir)
    workers = Workers(store) if start_workers else None
    if workers:
        for job_id in store.recover():
            workers.process(job_id)
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
        return await call_next(request)

    def _job(job_id: str) -> dict:
        try:
            return store.get(job_id)
        except KeyError:
            raise HTTPException(404, "job not found")

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
        return {k: d[k] for k in sorted(JOB_SETTING_KEYS)}

    # ----------------------------------------------------------------- jobs
    @app.get("/api/jobs")
    def list_jobs():
        out = []
        for j in store.list():
            m = store.manifest(j["id"])
            out.append({k: j[k] for k in ("id", "created", "source_name", "status", "stage",
                                          "progress", "error")}
                       | {"clips": len(m["clips"]) if m else 0})
        return out

    @app.post("/api/jobs")
    async def create_job(url: str = Form(""), settings: str = Form("{}"),
                         file: Optional[UploadFile] = File(None)):
        try:
            cfg = _clean_settings(json.loads(settings or "{}"))
        except json.JSONDecodeError:
            raise HTTPException(400, "settings must be JSON")
        if file is not None and file.filename:
            ext = Path(file.filename).suffix.lower()
            if ext not in UPLOAD_EXTS:
                raise HTTPException(400, f"unsupported file type {ext}")
            job = store.create("", file.filename, cfg)
            dest = store.dir(job["id"]) / ".work" / f"upload{ext}"
            with dest.open("wb") as fh:
                while chunk := await file.read(8 * 1024 * 1024):
                    fh.write(chunk)
            job = store.update(job["id"], source=str(dest), upload=True)
        elif url.strip():
            if not is_url(url.strip()):
                raise HTTPException(400, "url must start with http:// or https://")
            job = store.create(url.strip(), url.strip(), cfg)
        else:
            raise HTTPException(400, "upload a file or paste a URL")
        if workers:
            workers.process(job["id"])
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = _job(job_id)
        m = store.manifest(job_id)
        job["clips"] = [_clip_urls(job_id, c) for c in (m["clips"] if m else [])]
        job["source_info"] = ({k: m.get(k) for k in ("duration", "width", "height")} if m else None)
        job["source_url"] = f"/api/jobs/{job_id}/source" if m and m.get("video") else None
        return job

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job = _job(job_id)
        if job["status"] == "running":
            raise HTTPException(409, "job is running")
        store.delete(job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/source")
    def source_video(job_id: str):
        _job(job_id)
        m = store.manifest(job_id)
        if not m or not m.get("video") or not Path(m["video"]).exists():
            raise HTTPException(404, "source not available")
        return FileResponse(m["video"])

    @app.get("/api/jobs/{job_id}/transcript")
    def transcript(job_id: str):
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
        words = transcript(job_id)["words"]
        return " ".join(w["text"] for w in words if w["start"] >= start - 1e-6 and w["end"] <= end + 1e-6)

    def _check_range(m: dict, start: float, end: float) -> None:
        if not (0 <= start < end <= (m.get("duration") or 1e9)):
            raise HTTPException(400, "invalid start/end")
        if end - start < 1 or end - start > 180:
            raise HTTPException(400, "clips must be 1-180 seconds long")

    @app.patch("/api/jobs/{job_id}/clips/{clip_id}")
    def edit_clip(job_id: str, clip_id: str, body: dict = Body(...), render: bool = True):
        _job(job_id)
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
    def add_clip(job_id: str, body: dict = Body(...)):
        _job(job_id)
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
    def delete_clip(job_id: str, clip_id: str):
        def apply(m: dict) -> None:
            c = _find(m, clip_id)
            if c.get("output_path"):
                Path(c["output_path"]).unlink(missing_ok=True)
            m["clips"].remove(c)
        store.edit_manifest(job_id, apply)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/clips/{clip_id}/render")
    def rerender(job_id: str, clip_id: str):
        _job(job_id)
        _find(store.manifest(job_id) or {"clips": []}, clip_id)
        if workers:
            workers.render(job_id, clip_id)
        return {"ok": True}

    # ----------------------------------------------------------------- files
    @app.get("/files/{job_id}/{name}")
    def files(job_id: str, name: str):
        if "/" in name or name.startswith(".") or not name.endswith(".mp4"):
            raise HTTPException(404)
        path = store.dir(job_id) / name
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path, media_type="video/mp4",
                            headers={"Cache-Control": "no-cache"})

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8000, data_dir: str = "clipagent-data") -> None:
    import uvicorn

    if host not in ("127.0.0.1", "localhost") and not os.environ.get("CLIPAGENT_WEB_TOKEN"):
        print("warning: serving on a public interface without CLIPAGENT_WEB_TOKEN - anyone who "
              "can reach this port can use it.")
    print(f"clipagent web UI: http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}")
    uvicorn.run(create_app(data_dir), host=host, port=port, log_level="warning")
