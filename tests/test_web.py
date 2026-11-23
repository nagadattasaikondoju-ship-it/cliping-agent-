import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from clipagent.web.server import create_app  # noqa: E402
from tests.conftest import SENTENCES, make_transcript  # noqa: E402


def _wait(client, job_id, cond, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if cond(job):
            return job
        time.sleep(0.5)
    raise AssertionError(f"timed out: {job['status']} {job['stage']} {job['log'][-5:]}")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIPAGENT_WEB_TOKEN", raising=False)
    return TestClient(create_app(tmp_path / "data"))


def test_index_and_defaults(client):
    assert "clipagent" in client.get("/").text
    assert client.get("/app.js").status_code == 200
    d = client.get("/api/defaults").json()
    assert d["layout"] == "auto" and "num_clips" in d


def test_rejects_bad_input(client):
    assert client.post("/api/jobs", data={"url": ""}).status_code == 400
    assert client.post("/api/jobs", data={"url": "file:///etc/passwd"}).status_code == 400
    r = client.post("/api/jobs", data={"url": "https://x.y/v", "settings": json.dumps({"layout": "evil"})})
    assert r.status_code == 400
    r = client.post("/api/jobs", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
    assert client.get("/api/jobs/../../etc").status_code == 404
    assert client.get("/files/abc/..%2Fjob.json").status_code == 404


def test_full_job_edit_and_rerender(client, sample_video, tmp_path, monkeypatch):
    # Stub Whisper so the test is fast and deterministic.
    monkeypatch.setattr("clipagent.pipeline.transcribe",
                        lambda audio, settings, log=print: make_transcript(SENTENCES * 3))
    data = {"num_clips": 2, "min_duration": 10, "max_duration": 25, "target_duration": 15,
            "use_llm": False, "layout": "blur", "width": 360, "height": 640, "caption_size": 30}
    with open(sample_video, "rb") as fh:
        r = client.post("/api/jobs", data={"settings": json.dumps(data)},
                        files={"file": ("talk.mp4", fh, "video/mp4")})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    job = _wait(client, job_id, lambda j: j["status"] in ("done", "error"))
    assert job["status"] == "done", job["error"]
    assert len(job["clips"]) == 2
    clip = job["clips"][0]
    assert clip["status"] == "done" and clip["url"].startswith(f"/files/{job_id}/")

    # Video files support HTTP range requests (needed for seeking in the browser).
    r = client.get(clip["url"].split("?")[0], headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert client.get(f"/api/jobs/{job_id}/source").status_code == 200
    assert len(client.get(f"/api/jobs/{job_id}/transcript").json()["words"]) > 10

    # Edit: trim, retitle, change highlight colour -> re-render.
    r = client.patch(f"/api/jobs/{job_id}/clips/{clip['id']}", json={
        "start": clip["start"] + 1, "end": clip["end"] - 1, "title": "New title",
        "hashtags": ["#one", "two"], "options": {"caption_highlight": "FFD400", "hacker": 1}})
    assert r.status_code == 200, r.text
    edited = r.json()
    assert edited["title"] == "New title" and edited["hashtags"] == ["one", "two"]
    assert edited["options"] == {"caption_highlight": "FFD400"}
    job = _wait(client, job_id, lambda j: next(c for c in j["clips"] if c["id"] == clip["id"])
                .get("version", 0) >= 1 and next(c for c in j["clips"] if c["id"] == clip["id"])["status"] == "done")
    again = next(c for c in job["clips"] if c["id"] == clip["id"])
    assert again["url"].endswith("?v=1")

    assert client.patch(f"/api/jobs/{job_id}/clips/{clip['id']}", json={"start": 50, "end": 10}).status_code == 400

    # Add a custom clip from the editor, then delete it.
    r = client.post(f"/api/jobs/{job_id}/clips", json={"start": 2, "end": 14, "title": "Mine"})
    assert r.status_code == 200, r.text
    new_id = r.json()["id"]
    job = _wait(client, job_id, lambda j: any(c["id"] == new_id and c["status"] == "done" for c in j["clips"]))
    assert client.delete(f"/api/jobs/{job_id}/clips/{new_id}").status_code == 200
    assert len(client.get(f"/api/jobs/{job_id}").json()["clips"]) == 2

    # Posting without credentials reports a clean per-platform failure.
    r = client.post(f"/api/jobs/{job_id}/clips/{clip['id']}/publish", json={"platforms": ["tiktok"]})
    task = r.json()["task"]
    for _ in range(40):
        st = client.get(f"/api/publish/{task}").json()
        if st["status"] == "done":
            break
        time.sleep(0.2)
    assert st["results"][0]["ok"] is False

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_password_protection(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPAGENT_WEB_TOKEN", "s3cret")
    c = TestClient(create_app(tmp_path / "d", start_workers=False))
    assert c.get("/api/jobs").status_code == 401
    assert c.get("/").status_code == 200  # the page itself loads and shows a login box
    assert c.post("/api/login", json={"password": "nope"}).status_code == 401
    assert c.post("/api/login", json={"password": "s3cret"}).status_code == 200
    assert c.get("/api/jobs").status_code == 200  # cookie set
