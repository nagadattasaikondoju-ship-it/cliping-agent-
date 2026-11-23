"""Posting is tested against fake HTTP sessions: no real accounts are touched."""

import json

import pytest

from clipagent.cli import main
from clipagent.publish import PostRequest, publish
from clipagent.publish import instagram, tiktok


class FakeResp:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self._data = data if data is not None else {}
        self.text = json.dumps(self._data)

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, routes):
        self.routes = routes  # list of (method, url-substring, response)
        self.calls = []

    def _handle(self, method, url, **kw):
        self.calls.append((method, url, kw))
        for i, (m, frag, resp) in enumerate(self.routes):
            if m == method and frag in url:
                if isinstance(resp, list):
                    return resp.pop(0) if len(resp) > 1 else resp[0]
                return resp
        raise AssertionError(f"unexpected {method} {url}")

    def post(self, url, **kw):
        return self._handle("POST", url, **kw)

    def put(self, url, **kw):
        return self._handle("PUT", url, **kw)

    def get(self, url, **kw):
        return self._handle("GET", url, **kw)


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\0" * 1234)
    return p


def test_tiktok_chunk_plan():
    MB = 1024 * 1024
    assert tiktok.chunk_plan(3 * MB) == (3 * MB, 1)
    assert tiktok.chunk_plan(60 * MB) == (60 * MB, 1)
    size, count = tiktok.chunk_plan(150 * MB)
    assert size == 10 * MB and count == 15  # last chunk absorbs the remainder


def test_tiktok_inbox_upload(video, monkeypatch):
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok")
    monkeypatch.delenv("TIKTOK_MODE", raising=False)
    ok = {"error": {"code": "ok"}}
    s = FakeSession([
        ("POST", "/inbox/video/init/", FakeResp(200, {"data": {"publish_id": "p1", "upload_url": "https://up/1"}, **ok})),
        ("PUT", "https://up/1", FakeResp(201)),
        ("POST", "/status/fetch/", FakeResp(200, {"data": {"status": "SEND_TO_USER_INBOX"}, **ok})),
    ])
    res = tiktok.upload(PostRequest(video, "Title", hashtags=["a"]), log=lambda *_: None, session=s)
    assert res.ok and res.id == "p1" and "inbox" in res.detail
    init = s.calls[0][2]
    assert init["headers"]["Authorization"] == "Bearer tok"
    assert init["json"]["source_info"] == {"source": "FILE_UPLOAD", "video_size": 1234,
                                           "chunk_size": 1234, "total_chunk_count": 1}
    put = s.calls[1][2]
    assert put["headers"]["Content-Range"] == "bytes 0-1233/1234"


def test_tiktok_direct_respects_allowed_privacy(video, monkeypatch):
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("TIKTOK_MODE", "direct")
    ok = {"error": {"code": "ok"}}
    s = FakeSession([
        ("POST", "/creator_info/query/", FakeResp(200, {"data": {"privacy_level_options": ["SELF_ONLY"]}, **ok})),
        ("POST", "/post/publish/video/init/", FakeResp(200, {"data": {"publish_id": "p2", "upload_url": "https://up/2"}, **ok})),
        ("PUT", "https://up/2", FakeResp(201)),
        ("POST", "/status/fetch/", FakeResp(200, {"data": {"status": "PUBLISH_COMPLETE"}, **ok})),
    ])
    res = tiktok.upload(PostRequest(video, "T", privacy="public"), log=lambda *_: None, session=s)
    assert res.ok
    assert s.calls[1][2]["json"]["post_info"]["privacy_level"] == "SELF_ONLY"


def test_tiktok_error_is_reported(video, monkeypatch):
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok")
    monkeypatch.delenv("TIKTOK_MODE", raising=False)
    s = FakeSession([("POST", "/inbox/video/init/",
                      FakeResp(401, {"error": {"code": "access_token_invalid", "message": "bad token"}}))])
    monkeypatch.setattr("requests.Session", lambda: s)
    res = publish("tiktok", PostRequest(video, "T"), log=lambda *_: None)
    assert not res.ok and "access_token_invalid" in res.detail


def test_instagram_resumable_reel(video, monkeypatch):
    monkeypatch.setenv("IG_USER_ID", "42")
    monkeypatch.setenv("IG_ACCESS_TOKEN", "igt")
    s = FakeSession([
        ("POST", "/42/media_publish", FakeResp(200, {"id": "media9"})),
        ("POST", "/42/media", FakeResp(200, {"id": "c1", "uri": "https://rupload.facebook.com/ig-api-upload/v23.0/c1"})),
        ("POST", "rupload.facebook.com", FakeResp(200, {"success": True})),
        ("GET", "/c1", [FakeResp(200, {"status_code": "IN_PROGRESS"}), FakeResp(200, {"status_code": "FINISHED"})]),
    ])
    res = instagram.upload(PostRequest(video, "Big title", "desc", ["reels"]), log=lambda *_: None,
                           session=s, poll_seconds=0)
    assert res.ok and res.id == "media9"
    create = s.calls[0][2]["data"]
    assert create["media_type"] == "REELS" and create["upload_type"] == "resumable"
    assert create["caption"] == "Big title\n\ndesc\n\n#reels"
    up = s.calls[1][2]["headers"]
    assert up == {"Authorization": "OAuth igt", "offset": "0", "file_size": "1234"}


def test_unconfigured_platform_fails_cleanly(video, monkeypatch):
    monkeypatch.delenv("IG_USER_ID", raising=False)
    monkeypatch.delenv("IG_ACCESS_TOKEN", raising=False)
    res = publish("instagram", PostRequest(video, "T"))
    assert not res.ok and "IG_USER_ID" in res.detail


def test_cli_publish_uses_manifest_metadata(tmp_path, video, monkeypatch, capsys):
    (tmp_path / "clips.json").write_text(json.dumps({"clips": [
        {"output_path": str(video), "title": "From manifest", "description": "d", "hashtags": ["x"]}]}))
    seen = []

    def fake_publish(platform, req, log=print):
        from clipagent.publish import PostResult
        seen.append((platform, req))
        return PostResult(platform, True, id="1", detail="ok")

    monkeypatch.setattr("clipagent.publish.publish", fake_publish)
    assert main(["publish", str(tmp_path), "--to", "youtube,tiktok"]) == 0
    assert [p for p, _ in seen] == ["youtube", "tiktok"]
    assert seen[0][1].title == "From manifest" and seen[0][1].privacy == "private"
    assert main(["publish", str(tmp_path), "--to", "myspace"]) == 2


def test_youtube_upload_marks_short(video, monkeypatch):
    pytest.importorskip("googleapiclient")
    from clipagent.publish import youtube

    captured = {}

    class Req:
        def __init__(self):
            self.n = 0

        def next_chunk(self):
            self.n += 1
            return (None, {"id": "vid123"}) if self.n > 1 else (None, None)

    class Videos:
        def insert(self, part, body, media_body):
            captured.update(part=part, body=body)
            return Req()

    class YT:
        def videos(self):
            return Videos()

    monkeypatch.setattr(youtube, "_credentials", lambda: object())
    monkeypatch.setattr("googleapiclient.discovery.build", lambda *a, **k: YT())
    res = youtube.upload(PostRequest(video, "My clip", "about it", ["ai"], privacy="unlisted"),
                         log=lambda *_: None)
    assert res.ok and res.url == "https://youtube.com/shorts/vid123"
    snippet = captured["body"]["snippet"]
    assert "#Shorts" in snippet["description"] and "Shorts" in snippet["tags"]
    assert captured["body"]["status"]["privacyStatus"] == "unlisted"
