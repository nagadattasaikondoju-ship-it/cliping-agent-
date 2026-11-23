"""TikTok via the Content Posting API.

Default mode uploads to the creator's TikTok inbox as a draft (scope
`video.upload`); they finish and post it in the app. Set TIKTOK_MODE=direct to
post directly (scope `video.publish`; unaudited apps may only post with
SELF_ONLY privacy). Provide a user access token in TIKTOK_ACCESS_TOKEN.
"""

from __future__ import annotations

import math
import os
import time

from . import PostRequest, PostResult, PublishError

API = "https://open.tiktokapis.com/v2"
MIN_CHUNK = 5 * 1024 * 1024
MAX_CHUNK = 64 * 1024 * 1024


def status() -> tuple[bool, str]:
    if os.environ.get("TIKTOK_ACCESS_TOKEN"):
        return True, f"access token set (mode: {os.environ.get('TIKTOK_MODE', 'inbox')})"
    return False, "not configured (TIKTOK_ACCESS_TOKEN)"


def chunk_plan(size: int, chunk: int = 10 * 1024 * 1024) -> tuple[int, int]:
    """(chunk_size, total_chunk_count) following TikTok's rules: a file up to 64 MB
    goes in one chunk; otherwise chunks are 5-64 MB and the last absorbs the rest."""
    if size <= MAX_CHUNK:
        return size, 1
    chunk = max(MIN_CHUNK, min(MAX_CHUNK, chunk))
    return chunk, max(1, math.floor(size / chunk))


def _check(resp) -> dict:
    try:
        data = resp.json()
    except ValueError:
        raise PublishError(f"TikTok HTTP {resp.status_code}: {resp.text[:200]}")
    err = data.get("error", {})
    if resp.status_code >= 400 or (err.get("code") not in (None, "", "ok")):
        raise PublishError(f"TikTok error {err.get('code')}: {err.get('message', resp.text[:200])}")
    return data.get("data", {})


def upload(req: PostRequest, log=print, session=None) -> PostResult:
    import requests

    token = os.environ.get("TIKTOK_ACCESS_TOKEN")
    if not token:
        raise PublishError("Set TIKTOK_ACCESS_TOKEN")
    http = session or requests.Session()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"}
    size = req.video.stat().st_size
    chunk_size, count = chunk_plan(size)
    source = {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk_size,
              "total_chunk_count": count}
    mode = os.environ.get("TIKTOK_MODE", "inbox")

    if mode == "direct":
        info = _check(http.post(f"{API}/post/publish/creator_info/query/", headers=headers, json={}))
        options = info.get("privacy_level_options") or ["SELF_ONLY"]
        wanted = {"public": "PUBLIC_TO_EVERYONE", "unlisted": "MUTUAL_FOLLOW_FRIENDS",
                  "private": "SELF_ONLY"}[req.privacy]
        privacy = wanted if wanted in options else options[-1]
        body = {"post_info": {"title": req.caption(2200), "privacy_level": privacy,
                              "disable_comment": False, "disable_duet": False,
                              "disable_stitch": False},
                "source_info": source}
        init = _check(http.post(f"{API}/post/publish/video/init/", headers=headers, json=body))
    else:
        init = _check(http.post(f"{API}/post/publish/inbox/video/init/", headers=headers,
                                json={"source_info": source}))
    publish_id, upload_url = init["publish_id"], init["upload_url"]

    with open(req.video, "rb") as fh:
        for i in range(count):
            start = i * chunk_size
            length = chunk_size if i < count - 1 else size - start
            fh.seek(start)
            data = fh.read(length)
            resp = http.put(upload_url, data=data, headers={
                "Content-Type": "video/mp4", "Content-Length": str(length),
                "Content-Range": f"bytes {start}-{start + length - 1}/{size}"})
            if resp.status_code not in (200, 201, 206):
                raise PublishError(f"TikTok chunk {i + 1}/{count} failed: HTTP {resp.status_code}")
            log(f"  tiktok: chunk {i + 1}/{count}")

    state = ""
    for _ in range(30):
        st = _check(http.post(f"{API}/post/publish/status/fetch/", headers=headers,
                              json={"publish_id": publish_id}))
        state = st.get("status", "")
        if state in ("SEND_TO_USER_INBOX", "PUBLISH_COMPLETE"):
            break
        if state == "FAILED":
            raise PublishError(f"TikTok processing failed: {st.get('fail_reason', 'unknown')}")
        time.sleep(2)
    where = "sent to TikTok inbox - open TikTok to post it" if mode != "direct" else "posted"
    return PostResult("tiktok", True, id=publish_id, detail=f"{where} ({state or 'processing'})")
