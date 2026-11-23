"""Job storage and background workers for the web UI.

Every job is a folder: data/jobs/<id>/job.json plus a clipagent Project
(clips.json, rendered clips, .work/). Heavy work (transcription, rendering)
runs one task at a time on a worker thread.
"""

from __future__ import annotations

import dataclasses
import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..models import Clip


class JobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._job_locks: dict[str, threading.RLock] = {}

    # --------------------------------------------------------------- jobs
    def lock(self, job_id: str) -> threading.RLock:
        with self._lock:
            return self._job_locks.setdefault(job_id, threading.RLock())

    def dir(self, job_id: str) -> Path:
        if not job_id.isalnum():
            raise KeyError(job_id)
        return self.jobs_dir / job_id

    def create(self, source: str, source_name: str, settings: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        (self.dir(job_id) / ".work").mkdir(parents=True)
        job = {"id": job_id, "created": time.time(), "updated": time.time(), "source": source,
               "source_name": source_name, "status": "queued", "stage": "queued",
               "progress": 0.0, "error": "", "log": [], "settings": settings}
        self._write(job)
        return job

    def _write(self, job: dict) -> None:
        path = self.dir(job["id"]) / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False))
        tmp.replace(path)

    def get(self, job_id: str) -> dict:
        path = self.dir(job_id) / "job.json"
        if not path.exists():
            raise KeyError(job_id)
        return json.loads(path.read_text())

    def list(self) -> list[dict]:
        jobs = []
        for p in self.jobs_dir.glob("*/job.json"):
            try:
                jobs.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def update(self, job_id: str, **fields: Any) -> dict:
        with self.lock(job_id):
            job = self.get(job_id)
            job.update(fields, updated=time.time())
            self._write(job)
            return job

    def log(self, job_id: str, line: str) -> None:
        with self.lock(job_id):
            job = self.get(job_id)
            job["log"] = (job["log"] + [line])[-300:]
            self._write(job)

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self.dir(job_id), ignore_errors=True)

    # --------------------------------------------------------------- manifest
    def manifest_path(self, job_id: str) -> Path:
        return self.dir(job_id) / "clips.json"

    def manifest(self, job_id: str) -> dict | None:
        p = self.manifest_path(job_id)
        return json.loads(p.read_text()) if p.exists() else None

    def edit_manifest(self, job_id: str, fn: Callable[[dict], Any]) -> Any:
        """Read-modify-write clips.json under the job lock."""
        with self.lock(job_id):
            data = self.manifest(job_id)
            if data is None:
                raise KeyError("clips not ready")
            result = fn(data)
            tmp = self.manifest_path(job_id).with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            tmp.replace(self.manifest_path(job_id))
            return result

    def recover(self) -> list[str]:
        """After a restart: requeue queued jobs, flag ones that were mid-run."""
        requeue = []
        for job in self.list():
            if job["status"] == "queued":
                requeue.append(job["id"])
            elif job["status"] == "running":
                self.update(job["id"], status="error", error="Interrupted by a server restart")
        return requeue


def settings_from_dict(data: dict) -> Settings:
    names = {f.name for f in dataclasses.fields(Settings)}
    return Settings(**{k: v for k, v in data.items() if k in names})


# Fields a render writes back to a clip.
RENDER_FIELDS = ("output_path", "layout", "status", "error", "emojis", "broll", "speakers",
                 "speaker_switches")


class Workers:
    def __init__(self, store: JobStore):
        self.store = store
        self.heavy: "queue.Queue[tuple]" = queue.Queue()
        threading.Thread(target=self._loop, args=(self.heavy,), daemon=True).start()

    def _loop(self, q: "queue.Queue[tuple]") -> None:
        while True:
            task = q.get()
            try:
                getattr(self, f"_do_{task[0]}")(*task[1:])
            except Exception:  # keep the worker alive no matter what
                traceback.print_exc()
            finally:
                q.task_done()

    # --------------------------------------------------------------- submit
    def process(self, job_id: str) -> None:
        self.heavy.put(("process", job_id))

    def render(self, job_id: str, clip_id: str) -> None:
        self.store.edit_manifest(job_id, lambda m: _set_status(m, clip_id, "queued"))
        self.heavy.put(("render", job_id, clip_id))

    # --------------------------------------------------------------- tasks
    def _do_process(self, job_id: str) -> None:
        from ..pipeline import Project

        store = self.store
        job = store.update(job_id, status="running", stage="starting", progress=0.0, error="")
        project = Project(store.dir(job_id), settings_from_dict(job["settings"]),
                          log=lambda line: store.log(job_id, str(line)),
                          progress=lambda stage, frac: store.update(job_id, stage=stage,
                                                                    progress=round(frac, 3)))
        try:
            project.prepare(job["source"])
            project.find()
            if job.get("upload"):
                store.update(job_id, source_video=str(project.video))
            n = len(project.clips)
            for i, clip in enumerate(project.clips, 1):
                store.update(job_id, stage=f"rendering clip {i}/{n}",
                             progress=round(0.35 + 0.65 * (i - 1) / max(1, n), 3))
                try:
                    project.render(clip, i)
                except Exception as exc:
                    store.log(job_id, f"clip {i} failed: {exc}")
                with store.lock(job_id):
                    project.save()
            store.update(job_id, status="done", stage="done", progress=1.0)
        except Exception as exc:
            store.log(job_id, f"error: {exc}")
            store.update(job_id, status="error", stage="error", error=str(exc)[-800:])

    def _do_render(self, job_id: str, clip_id: str) -> None:
        from ..pipeline import Project

        store = self.store
        with store.lock(job_id):
            project = Project.load(store.dir(job_id), log=lambda l: store.log(job_id, str(l)))
        idx = next((i for i, c in enumerate(project.clips) if c.id == clip_id), None)
        if idx is None:
            return
        clip = project.clips[idx]
        store.edit_manifest(job_id, lambda m: _set_status(m, clip_id, "rendering"))
        try:
            project.render(clip, idx + 1)
        except Exception as exc:
            store.log(job_id, f"re-render failed: {exc}")
        rendered = clip.to_dict()

        def apply(m: dict) -> None:
            for c in m["clips"]:
                if c["id"] == clip_id:
                    c.update({k: rendered[k] for k in RENDER_FIELDS})
                    c["version"] = c.get("version", 0) + 1
        store.edit_manifest(job_id, apply)

def _set_status(manifest: dict, clip_id: str, status: str) -> None:
    for c in manifest["clips"]:
        if c["id"] == clip_id:
            c["status"] = status


def new_clip_dict(start: float, end: float, title: str, transcript_text: str) -> dict:
    clip = Clip(start=start, end=end, text=transcript_text, id=uuid.uuid4().hex[:8], title=title,
                hook=title, description=title)
    return clip.to_dict()
