"""Instagram Reels via the Instagram Graph API (resumable upload).

Needs an Instagram professional (business/creator) account linked to a
Facebook app with `instagram_content_publish`: set IG_USER_ID and
IG_ACCESS_TOKEN.
"""

from __future__ import annotations

import os
import time

from . import PostRequest, PostResult, PublishError


def _graph() -> str:
    return f"https://graph.facebook.com/{os.environ.get('GRAPH_API_VERSION', 'v23.0')}"


def status() -> tuple[bool, str]:
    if os.environ.get("IG_USER_ID") and os.environ.get("IG_ACCESS_TOKEN"):
        return True, "IG_USER_ID and IG_ACCESS_TOKEN set"
    return False, "not configured (IG_USER_ID, IG_ACCESS_TOKEN)"


def _json(resp) -> dict:
    try:
        data = resp.json()
    except ValueError:
        raise PublishError(f"Instagram HTTP {resp.status_code}: {resp.text[:200]}")
    if resp.status_code >= 400 or "error" in data:
        err = data.get("error", {})
        raise PublishError(f"Instagram error: {err.get('message', resp.text[:200])}")
    return data


def upload(req: PostRequest, log=print, session=None, poll_seconds: float = 5.0) -> PostResult:
    import requests

    user, token = os.environ.get("IG_USER_ID"), os.environ.get("IG_ACCESS_TOKEN")
    if not (user and token):
        raise PublishError("Set IG_USER_ID and IG_ACCESS_TOKEN")
    http = session or requests.Session()
    graph = _graph()

    container = _json(http.post(f"{graph}/{user}/media", data={
        "media_type": "REELS", "upload_type": "resumable", "caption": req.caption(2200),
        "share_to_feed": "true", "access_token": token}))
    cid = container["id"]
    upload_uri = container.get("uri") or f"https://rupload.facebook.com/ig-api-upload/{graph.rsplit('/', 1)[-1]}/{cid}"

    size = req.video.stat().st_size
    with open(req.video, "rb") as fh:
        resp = http.post(upload_uri, data=fh.read(), headers={
            "Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size)})
    _json(resp)
    log("  instagram: uploaded, waiting for processing...")

    for _ in range(60):
        st = _json(http.get(f"{graph}/{cid}", params={"fields": "status_code,status",
                                                      "access_token": token}))
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise PublishError(f"Instagram processing failed: {st.get('status', code)}")
        time.sleep(poll_seconds)
    else:
        raise PublishError("Instagram is still processing the video; try publishing again later")

    media = _json(http.post(f"{graph}/{user}/media_publish",
                            data={"creation_id": cid, "access_token": token}))
    return PostResult("instagram", True, id=media["id"], detail="published as a Reel")
